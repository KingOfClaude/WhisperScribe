// WhisperScribe - desktop GUI for whisper.cpp (+ optional NVIDIA Parakeet engine via sherpa-onnx)
// Dear ImGui + GLFW front end, miniaudio (+ optional ffmpeg) for audio decoding.
// Produces timestamped segments and exports SRT / VTT / TXT.

#ifdef _WIN32
  #ifndef NOMINMAX
    #define NOMINMAX            // keep std::min / std::max working
  #endif
  #ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
  #endif
  #include <windows.h>
#endif

#include <atomic>
#include <algorithm>
#include <cfloat>
#include <filesystem>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "misc/cpp/imgui_stdlib.h"
#include <GLFW/glfw3.h>
#ifdef _WIN32
  #define GLFW_EXPOSE_NATIVE_WIN32
  #include <GLFW/glfw3native.h>
#endif
#include <nfd.h>

#include "whisper.h"
#include "downloader.h"

#ifdef WITH_PARAKEET
  #include "parakeet.h"
#endif

#define MA_NO_DEVICE_IO
#define MA_NO_ENCODING
#define MA_NO_GENERATION
#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"

// ----------------------------------------------------------------------------
// Data types
// ----------------------------------------------------------------------------
struct Segment {
    int64_t t0;  // centiseconds (whisper units: 1 = 10 ms)
    int64_t t1;
    std::string text;
};

struct Settings {
    std::string model;
    std::string audio;
    int  langIdx   = 0;
    int  threads   = 4;
    int  beam      = 5;      // 1 = greedy
    bool translate = false;  // translate to English
    bool useContext = true;  // condition on previous text
    int  maxLen    = 0;      // max chars per segment (0 = whisper default)
    std::string prompt;      // initial prompt (names, jargon, punctuation style)
    int  engine = 0;         // 0 = Whisper, 1 = Parakeet (needs WITH_PARAKEET build)
    std::string pkDir;       // Parakeet model folder (encoder/decoder/joiner/tokens)
    std::string pkVad;       // silero_vad.onnx
};

struct Job {
    std::atomic<bool> running{false};
    std::atomic<bool> done{false};
    std::atomic<bool> abort{false};
    std::atomic<int>  progress{0};
    std::mutex mtx;                 // guards segs, status, lang
    std::vector<Segment> segs;
    std::string status = "Idle";
    std::string lang;
    std::thread th;
};

struct App {
    whisper_context* ctx = nullptr;  // only touched by the worker or after join
    std::string ctxPath;
    Job job;
    Settings s;
    std::string message;             // last UI message (export results etc.)

    // in-app model downloads
    dl::Downloader downloader;
    int  dlKind = 0;                 // 0 = Whisper model, 1 = Parakeet
    int  dlSel = 0;                  // selected row in the download list
    bool dlOpenRequest = false;      // open the popup on the next frame
    std::string dlMsg;               // outcome text shown in the popup
    bool dlMsgOk = false;
    int  dlGen = 0;                  // bumped when a download finishes (forces file re-checks)
    std::string dlTarget;            // folder shown as "Saves to"
};

static const struct { const char* code; const char* name; } LANGS[] = {
    {"auto", "Auto-detect"}, {"en", "English"},  {"es", "Spanish"},   {"fr", "French"},
    {"de", "German"},        {"it", "Italian"},  {"pt", "Portuguese"},{"nl", "Dutch"},
    {"pl", "Polish"},        {"ru", "Russian"},  {"uk", "Ukrainian"}, {"tr", "Turkish"},
    {"sv", "Swedish"},       {"zh", "Chinese"},  {"ja", "Japanese"},  {"ko", "Korean"},
    {"ar", "Arabic"},        {"hi", "Hindi"},    {"he", "Hebrew"},    {"cs", "Czech"},
    {"el", "Greek"},         {"ro", "Romanian"}, {"hu", "Hungarian"}, {"da", "Danish"},
    {"fi", "Finnish"},       {"no", "Norwegian"},{"id", "Indonesian"},{"vi", "Vietnamese"},
};
static const int N_LANGS = (int)(sizeof(LANGS) / sizeof(LANGS[0]));

static App* g_app = nullptr;

// ----------------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------------
static std::string fmt_ts(int64_t cs, char sep) {
    int64_t ms = cs * 10;
    int h = (int)(ms / 3600000); ms %= 3600000;
    int m = (int)(ms / 60000);   ms %= 60000;
    int s = (int)(ms / 1000);    ms %= 1000;
    char b[48];
    snprintf(b, sizeof b, "%02d:%02d:%02d%c%03d", h, m, s, sep, (int)ms);
    return b;
}

static std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

#ifdef _WIN32
static std::wstring utf8_to_wide(const std::string& s) {
    if (s.empty()) return L"";
    int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), nullptr, 0);
    std::wstring w(n, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), &w[0], n);
    return w;
}
#endif

static void set_status(Job& j, const std::string& s) {
    std::lock_guard<std::mutex> lk(j.mtx);
    j.status = s;
}

// ----------------------------------------------------------------------------
// Audio decoding -> 16 kHz mono float PCM (what whisper expects)
// ----------------------------------------------------------------------------
static bool decode_miniaudio(const std::string& path, std::vector<float>& out) {
    ma_decoder_config cfg = ma_decoder_config_init(ma_format_f32, 1, 16000);
    ma_decoder dec;
#ifdef _WIN32
    if (ma_decoder_init_file_w(utf8_to_wide(path).c_str(), &cfg, &dec) != MA_SUCCESS) return false;
#else
    if (ma_decoder_init_file(path.c_str(), &cfg, &dec) != MA_SUCCESS) return false;
#endif
    std::vector<float> buf(16384);
    for (;;) {
        ma_uint64 n = 0;
        ma_result r = ma_decoder_read_pcm_frames(&dec, buf.data(), buf.size(), &n);
        if (n > 0) out.insert(out.end(), buf.begin(), buf.begin() + (size_t)n);
        if (r != MA_SUCCESS || n == 0) break;
    }
    ma_decoder_uninit(&dec);
    return !out.empty();
}

// Fallback for m4a / mp4 / ogg / opus / mkv / aac ... (needs ffmpeg on PATH)
static bool decode_ffmpeg(const std::string& path, std::vector<float>& out) {
    std::string cmd;
#ifdef _WIN32
    std::wstring wcmd = L"ffmpeg -v error -i \"" + utf8_to_wide(path) +
                        L"\" -f f32le -ac 1 -ar 16000 -";
    FILE* p = _wpopen(wcmd.c_str(), L"rb");
#else
    std::string q = "'";
    for (char c : path) { if (c == '\'') q += "'\\''"; else q += c; }
    q += "'";
    cmd = "ffmpeg -v error -i " + q + " -f f32le -ac 1 -ar 16000 -";
    FILE* p = popen(cmd.c_str(), "r");
#endif
    if (!p) return false;
    std::vector<float> buf(16384);
    size_t n;
    while ((n = fread(buf.data(), sizeof(float), buf.size(), p)) > 0)
        out.insert(out.end(), buf.begin(), buf.begin() + n);
#ifdef _WIN32
    _pclose(p);
#else
    pclose(p);
#endif
    return !out.empty();
}

static bool load_audio(const std::string& path, std::vector<float>& pcm) {
    pcm.clear();
    if (decode_miniaudio(path, pcm)) return true;
    pcm.clear();
    return decode_ffmpeg(path, pcm);
}

// ----------------------------------------------------------------------------
// Worker thread: load model -> decode audio -> whisper_full
// ----------------------------------------------------------------------------
static void worker(App* app, Settings s) {
    Job& job = app->job;

    const bool useParakeet = (s.engine == 1);

    // 1. model (Whisper only; Parakeet loads its own model below)
    if (!useParakeet && (!app->ctx || app->ctxPath != s.model)) {
        set_status(job, "Loading model...");
        if (app->ctx) { whisper_free(app->ctx); app->ctx = nullptr; }
        whisper_context_params cp = whisper_context_default_params();
        app->ctx = whisper_init_from_file_with_params(s.model.c_str(), cp);
        if (!app->ctx) {
            set_status(job, "ERROR: could not load model file.");
            job.done = true;
            return;
        }
        app->ctxPath = s.model;
    }

    // 2. audio
    set_status(job, "Decoding audio...");
    std::vector<float> pcm;
    if (!load_audio(s.audio, pcm)) {
        set_status(job, "ERROR: could not decode audio (install ffmpeg for exotic formats).");
        job.done = true;
        return;
    }
    if (job.abort) { set_status(job, "Cancelled."); job.done = true; return; }

#ifdef WITH_PARAKEET
    if (useParakeet) {
        pk::Config pc;
        pc.modelDir = s.pkDir;
        pc.vadModel = s.pkVad;
        pc.threads  = s.threads;
        pc.maxChars = s.maxLen;

        std::string err;
        bool ok = pk::transcribe(pcm, pc,
            [&job](int64_t t0, int64_t t1, const std::string& text) {
                std::lock_guard<std::mutex> lk(job.mtx);
                job.segs.push_back({ t0, t1, text });
            },
            [&job](int p) { job.progress = p; },
            [&job](const std::string& t) { set_status(job, t); },
            job.abort, err);

        if (!ok)            set_status(job, "ERROR: " + err);
        else if (job.abort) set_status(job, "Cancelled (partial result kept).");
        else {
            std::lock_guard<std::mutex> lk(job.mtx);
            job.status = job.segs.empty() ? "Done - no speech detected." : "Done.";
            job.progress = 100;
        }
        job.done = true;
        return;
    }
#endif

    // 3. transcribe (Whisper)
    whisper_full_params wp = whisper_full_default_params(
        s.beam > 1 ? WHISPER_SAMPLING_BEAM_SEARCH : WHISPER_SAMPLING_GREEDY);

    wp.n_threads        = s.threads;
    wp.language         = LANGS[s.langIdx].code;  // "auto" => detect
    wp.translate        = s.translate;
    wp.no_context       = !s.useContext;
    wp.print_progress   = false;
    wp.print_realtime   = false;
    wp.print_timestamps = false;
    wp.print_special    = false;
    // Temperature fallback + quality gates (these are whisper's accuracy safeguards)
    wp.temperature_inc  = 0.2f;
    wp.entropy_thold    = 2.4f;
    wp.logprob_thold    = -1.0f;
    wp.no_speech_thold  = 0.6f;
    wp.greedy.best_of   = 5;
    wp.beam_search.beam_size = std::max(1, s.beam);
    if (!s.prompt.empty()) wp.initial_prompt = s.prompt.c_str();
    if (s.maxLen > 0) {            // finer-grained, subtitle-length segments
        wp.token_timestamps = true;
        wp.max_len          = s.maxLen;
        wp.split_on_word    = true;
    }

    wp.new_segment_callback = [](whisper_context* c, whisper_state*, int n_new, void* ud) {
        Job* j = (Job*)ud;
        int n = whisper_full_n_segments(c);
        std::lock_guard<std::mutex> lk(j->mtx);
        for (int i = n - n_new; i < n; ++i) {
            j->segs.push_back({ whisper_full_get_segment_t0(c, i),
                                whisper_full_get_segment_t1(c, i),
                                trim(whisper_full_get_segment_text(c, i)) });
        }
    };
    wp.new_segment_callback_user_data = &job;

    wp.progress_callback = [](whisper_context*, whisper_state*, int p, void* ud) {
        ((Job*)ud)->progress = p;
    };
    wp.progress_callback_user_data = &job;

    wp.abort_callback = [](void* ud) -> bool { return ((Job*)ud)->abort.load(); };
    wp.abort_callback_user_data = &job;

    char st[96];
    snprintf(st, sizeof st, "Transcribing (%.1f min of audio)...", pcm.size() / 16000.0 / 60.0);
    set_status(job, st);

    int rc = whisper_full(app->ctx, wp, pcm.data(), (int)pcm.size());

    if (job.abort)      set_status(job, "Cancelled (partial result kept).");
    else if (rc != 0)   set_status(job, "ERROR: whisper_full failed.");
    else {
        std::lock_guard<std::mutex> lk(job.mtx);
        job.lang = whisper_lang_str(whisper_full_lang_id(app->ctx));
        job.status = "Done.";
        job.progress = 100;
    }
    job.done = true;
}

// ----------------------------------------------------------------------------
// Export
// ----------------------------------------------------------------------------
enum class Fmt { SRT, VTT, TXT };

static std::string build_export(const std::vector<Segment>& segs, Fmt f) {
    std::string o;
    if (f == Fmt::VTT) o += "WEBVTT\n\n";
    int idx = 1;
    for (auto& s : segs) {
        if (f == Fmt::SRT) {
            o += std::to_string(idx++) + "\n";
            o += fmt_ts(s.t0, ',') + " --> " + fmt_ts(s.t1, ',') + "\n" + s.text + "\n\n";
        } else if (f == Fmt::VTT) {
            o += fmt_ts(s.t0, '.') + " --> " + fmt_ts(s.t1, '.') + "\n" + s.text + "\n\n";
        } else {
            o += "[" + fmt_ts(s.t0, '.') + " --> " + fmt_ts(s.t1, '.') + "]  " + s.text + "\n";
        }
    }
    return o;
}

static std::string open_dialog(const char* name, const char* exts) {
    nfdu8filteritem_t f[1] = {{ name, exts }};
    nfdu8char_t* out = nullptr;
    std::string r;
    if (NFD_OpenDialogU8(&out, f, 1, nullptr) == NFD_OKAY) { r = out; NFD_FreePathU8(out); }
    return r;
}

static std::string pick_folder() {
    nfdu8char_t* out = nullptr;
    std::string r;
    if (NFD_PickFolderU8(&out, nullptr) == NFD_OKAY) { r = out; NFD_FreePathU8(out); }
    return r;
}

static std::string save_dialog(const char* name, const char* ext, const char* defName) {
    nfdu8filteritem_t f[1] = {{ name, ext }};
    nfdu8char_t* out = nullptr;
    std::string r;
    if (NFD_SaveDialogU8(&out, f, 1, nullptr, defName) == NFD_OKAY) { r = out; NFD_FreePathU8(out); }
    return r;
}

static void do_export(App& app, Fmt f) {
    const char* ext  = f == Fmt::SRT ? "srt" : f == Fmt::VTT ? "vtt" : "txt";
    std::string path = save_dialog(ext, ext, (std::string("transcript.") + ext).c_str());
    if (path.empty()) return;
    std::string data;
    {
        std::lock_guard<std::mutex> lk(app.job.mtx);
        data = build_export(app.job.segs, f);
    }
    std::ofstream out(std::filesystem::u8path(path), std::ios::binary);
    if (out) { out << data; app.message = "Saved " + path; }
    else     { app.message = "Could not write " + path; }
}

// ----------------------------------------------------------------------------
// GUI
// ----------------------------------------------------------------------------
// Taskbar / title-bar icon. The icon lives in the exe (app.rc, resource "GLFW_ICON"); asking for
// explicit sizes keeps it sharp on the taskbar, which scales the "big" icon down.
static void apply_window_icon(GLFWwindow* win) {
#ifdef _WIN32
    HWND hwnd = glfwGetWin32Window(win);
    if (!hwnd) return;
    HINSTANCE inst = GetModuleHandleW(nullptr);
    int smallPx = GetSystemMetrics(SM_CXSMICON);
    int bigPx   = GetSystemMetrics(SM_CXICON) * 2;
    HANDLE hs = LoadImageW(inst, L"GLFW_ICON", IMAGE_ICON, smallPx, smallPx, LR_SHARED);
    HANDLE hb = LoadImageW(inst, L"GLFW_ICON", IMAGE_ICON, bigPx, bigPx, LR_SHARED);
    if (hs) SendMessageW(hwnd, WM_SETICON, ICON_SMALL, (LPARAM)hs);
    if (hb) SendMessageW(hwnd, WM_SETICON, ICON_BIG, (LPARAM)hb);
#else
    (void)win;
#endif
}

static void drop_cb(GLFWwindow*, int count, const char** paths) {
    if (g_app && count > 0 && !g_app->job.running) g_app->s.audio = paths[0];
}

static void load_font(ImGuiIO& io) {
    const char* cands[] = {
        "C:/Windows/Fonts/segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    };
    for (const char* p : cands) {
        if (std::ifstream(p).good() &&
            io.Fonts->AddFontFromFileTTF(p, 17.0f, nullptr, io.Fonts->GetGlyphRangesCyrillic()))
            return;
    }
    io.Fonts->AddFontDefault();
}

static void start_job(App& app) {
    Job& j = app.job;
    const bool pkMode = (app.s.engine == 1);
    if (app.s.audio.empty() || (!pkMode && app.s.model.empty()) ||
        (pkMode && (app.s.pkDir.empty() || app.s.pkVad.empty()))) {
        app.message = pkMode ? "Choose the Parakeet model folder, the VAD model and an audio file first."
                             : "Choose a model file and an audio file first.";
        return;
    }
    {
        std::lock_guard<std::mutex> lk(j.mtx);
        j.segs.clear();
        j.lang.clear();
    }
    j.abort = false; j.done = false; j.progress = 0; j.running = true;
    app.message.clear();
    j.th = std::thread(worker, &app, app.s);
}

// ----------------------------------------------------------------------------
// In-app model downloads (curl runs in a background thread, see downloader.cpp)
// ----------------------------------------------------------------------------
static void request_download(App& app, int kind) {
    if (!app.downloader.busy()) {
        app.dlKind = kind;
        app.dlSel = 0;
        app.dlMsg.clear();
    }
    app.dlTarget = dl::models_dir();
    app.dlOpenRequest = true;
}

// Once per frame: apply a finished download to the settings.
static void poll_downloads(App& app) {
    dl::Result r;
    if (!app.downloader.poll(r)) return;
    ++app.dlGen;
    if (r.ok) {
        if (app.dlKind == 0) app.s.model = r.path;
        else { app.s.pkDir = r.dir; app.s.pkVad = r.vad; }
        app.dlMsg = r.message + " The model is selected - you're ready to transcribe.";
        app.dlMsgOk = true;
    } else {
        app.dlMsg = r.cancelled ? "Download cancelled. Partial files are kept, so downloading again resumes."
                                : r.error;
        app.dlMsgOk = false;
    }
    app.dlOpenRequest = true;   // make sure the outcome is seen
}

static void draw_download_popup(App& app) {
    if (app.dlOpenRequest) {
        ImGui::OpenPopup("Download model");
        app.dlOpenRequest = false;
    }
    ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSizeConstraints(ImVec2(660, 0), ImVec2(660, FLT_MAX));
    if (!ImGui::BeginPopupModal("Download model", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) return;

    const bool whisper = (app.dlKind == 0);
    ImGui::TextUnformatted(whisper ? "Download a Whisper model" : "Download the Parakeet model");
    ImGui::TextDisabled("Saves to: %s", app.dlTarget.c_str());
    ImGui::Separator();

    if (app.downloader.busy()) {
        // ---- progress view
        std::string st = app.downloader.status();
        ImGui::TextWrapped("%s", st.c_str());
        ImGui::ProgressBar(app.downloader.percent() / 100.0f, ImVec2(-FLT_MIN, 0));
        if (ImGui::Button("Cancel download", ImVec2(170, 0))) app.downloader.cancel();
        ImGui::SameLine();
        ImGui::TextDisabled("Partial files are kept so you can resume later.");
    } else if (!app.dlMsg.empty()) {
        // ---- outcome view
        ImGui::TextColored(app.dlMsgOk ? ImVec4(0.4f, 0.9f, 0.4f, 1) : ImVec4(1, 0.45f, 0.4f, 1), "%s",
                           app.dlMsgOk ? "Done" : "Problem");
        ImGui::TextWrapped("%s", app.dlMsg.c_str());
        ImGui::Spacing();
        if (ImGui::Button(app.dlMsgOk ? "OK" : "Back", ImVec2(110, 0))) {
            bool ok = app.dlMsgOk;
            app.dlMsg.clear();
            if (ok) ImGui::CloseCurrentPopup();
        }
    } else {
        // ---- choose what to download
        const int count = whisper ? dl::whisper_model_count() : dl::parakeet_variant_count();
        if (ImGui::BeginTable("dlist", whisper ? 3 : 2, ImGuiTableFlags_RowBg | ImGuiTableFlags_Borders)) {
            ImGui::TableSetupColumn(whisper ? "Model" : "Version", ImGuiTableColumnFlags_WidthFixed,
                                    whisper ? 170.0f : 270.0f);
            if (whisper) ImGui::TableSetupColumn("Size", ImGuiTableColumnFlags_WidthFixed, 70.0f);
            ImGui::TableSetupColumn("Notes", ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableHeadersRow();
            for (int i = 0; i < count; ++i) {
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                const char* label = whisper ? dl::whisper_model(i).id : dl::parakeet_variant(i).label;
                if (ImGui::Selectable(label, app.dlSel == i, ImGuiSelectableFlags_SpanAllColumns))
                    app.dlSel = i;
                int col = 1;
                if (whisper) {
                    ImGui::TableSetColumnIndex(col++);
                    ImGui::TextUnformatted(dl::whisper_model(i).size);
                }
                ImGui::TableSetColumnIndex(col);
                ImGui::TextWrapped("%s", whisper ? dl::whisper_model(i).note : dl::parakeet_variant(i).note);
            }
            ImGui::EndTable();
        }
        ImGui::Spacing();
        ImGui::TextDisabled(whisper ? "large-v3-turbo is a good balance if GPU memory is limited."
                                    : "Also downloads the small Silero VAD speech-detection model.");
        ImGui::Spacing();
        if (ImGui::Button("Download", ImVec2(140, 0))) {
            app.dlMsg.clear();
            if (whisper) app.downloader.start_whisper(app.dlSel);
            else         app.downloader.start_parakeet(app.dlSel);
        }
        ImGui::SameLine();
        if (ImGui::Button("Close", ImVec2(110, 0))) ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
}

static void draw_ui(App& app) {
    Job& job = app.job;
    Settings& s = app.s;
    poll_downloads(app);

    // reap finished worker
    if (job.done && job.th.joinable()) { job.th.join(); job.running = false; job.done = false; }
    bool running = job.running;

    const ImGuiViewport* vp = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(vp->WorkPos);
    ImGui::SetNextWindowSize(vp->WorkSize);
    ImGui::Begin("main", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
                 ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoBringToFrontOnFocus);

    ImGui::BeginDisabled(running);

    // --- files
    float btnW = 90.0f;
    float dlW  = 110.0f;
    float sp   = ImGui::GetStyle().ItemSpacing.x;
    // one common input width so the Browse buttons line up on every row
    float inW = ImGui::GetContentRegionAvail().x - btnW - dlW - 3 * sp
              - ImGui::CalcTextSize("Parakeet folder").x - 12;
    bool usePk = false;
#ifdef WITH_PARAKEET
    {
        static const char* ENGINES[] = {
            "Whisper  (most accurate, 99 languages)",
            "Parakeet  (very fast, English + 24 European languages)" };
        ImGui::SetNextItemWidth(420);
        if (ImGui::BeginCombo("Engine", ENGINES[s.engine])) {
            for (int i = 0; i < 2; ++i)
                if (ImGui::Selectable(ENGINES[i], i == s.engine)) s.engine = i;
            ImGui::EndCombo();
        }
        usePk = (s.engine == 1);
    }

    // cached validity checks for the Parakeet files (re-run only when a path changes)
    static std::string pkSeenDir, pkSeenVad, pkMsg;
    static bool pkOk = false, pkVadOk = false;
    static int pkSeenGen = -1;
    const bool pkRecheck = (pkSeenGen != app.dlGen);   // a download just finished
    pkSeenGen = app.dlGen;
    if (pkRecheck || s.pkDir != pkSeenDir) { pkSeenDir = s.pkDir; pkOk = pk::check_model_dir(s.pkDir, pkMsg); }
    if (pkRecheck || s.pkVad != pkSeenVad) {
        pkSeenVad = s.pkVad;
        std::error_code ec;
        pkVadOk = !s.pkVad.empty() && std::filesystem::exists(std::filesystem::u8path(s.pkVad), ec);
    }
#endif

    if (!usePk) {
        ImGui::SetNextItemWidth(inW);
        ImGui::InputText("##model", &s.model, 0); ImGui::SameLine();
        if (ImGui::Button("Browse##m", ImVec2(btnW, 0))) {
            std::string p = open_dialog("Whisper model (ggml)", "bin,gguf");
            if (!p.empty()) s.model = p;
        }
        ImGui::SameLine();
        if (ImGui::Button("Download...##w", ImVec2(dlW, 0))) request_download(app, 0);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Download a Whisper model into the models folder");
        ImGui::SameLine(); ImGui::TextUnformatted("Model");
    }
#ifdef WITH_PARAKEET
    else {
        ImGui::SetNextItemWidth(inW);
        ImGui::InputText("##pkdir", &s.pkDir, 0); ImGui::SameLine();
        if (ImGui::Button("Browse##pkd", ImVec2(btnW, 0))) {
            std::string p = pick_folder();
            if (!p.empty()) s.pkDir = p;
        }
        ImGui::SameLine();
        if (ImGui::Button("Download...##pk", ImVec2(dlW, 0))) request_download(app, 1);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Download the Parakeet model + Silero VAD into the models folder");
        ImGui::SameLine(); ImGui::TextUnformatted("Parakeet folder");
        if (s.pkDir.empty())
            ImGui::TextDisabled("Click Download... to fetch the model, or browse to an existing model folder.");
        else if (pkOk)
            ImGui::TextColored(ImVec4(0.4f, 0.9f, 0.4f, 1), "Parakeet model found.");
        else
            ImGui::TextColored(ImVec4(1, 0.5f, 0.4f, 1), "%s", pkMsg.c_str());

        ImGui::SetNextItemWidth(inW);
        ImGui::InputText("##pkvad", &s.pkVad, 0); ImGui::SameLine();
        if (ImGui::Button("Browse##pkv", ImVec2(btnW, 0))) {
            std::string p = open_dialog("Silero VAD model", "onnx");
            if (!p.empty()) s.pkVad = p;
        }
        ImGui::SameLine(); ImGui::TextUnformatted("VAD model");
        if (!s.pkVad.empty() && !pkVadOk)
            ImGui::TextColored(ImVec4(1, 0.5f, 0.4f, 1), "silero_vad.onnx not found at that path.");
    }
#endif

    ImGui::SetNextItemWidth(inW);
    ImGui::InputText("##audio", &s.audio, 0); ImGui::SameLine();
    if (ImGui::Button("Browse##a", ImVec2(btnW, 0))) {
        std::string p = open_dialog("Audio / video",
            "wav,mp3,flac,m4a,ogg,opus,aac,wma,mp4,mkv,webm,mov");
        if (!p.empty()) s.audio = p;
    }
    ImGui::SameLine(); ImGui::TextUnformatted("Audio file");
    ImGui::TextDisabled("Tip: you can also drag & drop an audio file onto this window.");

    // --- settings
    ImGui::Separator();
    ImGui::BeginDisabled(usePk);
    ImGui::SetNextItemWidth(160);
    if (ImGui::BeginCombo("Language", usePk ? "Auto (Parakeet)" : LANGS[s.langIdx].name)) {
        for (int i = 0; i < N_LANGS; ++i)
            if (ImGui::Selectable(LANGS[i].name, i == s.langIdx)) s.langIdx = i;
        ImGui::EndCombo();
    }
    ImGui::EndDisabled();
    ImGui::SameLine(0, 24);
    ImGui::SetNextItemWidth(110);
    ImGui::SliderInt("Threads", &s.threads, 1, std::max(1u, std::thread::hardware_concurrency()));
    ImGui::SameLine(0, 24);
    ImGui::BeginDisabled(usePk);
    ImGui::SetNextItemWidth(110);
    ImGui::SliderInt("Beam size", &s.beam, 1, 8);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("1 = greedy (fastest). 5 = recommended for accuracy.");
    ImGui::EndDisabled();

    ImGui::SetNextItemWidth(110);
    ImGui::InputInt("Max chars/segment", &s.maxLen, 0, 0);
    s.maxLen = std::max(0, s.maxLen);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("0 = natural segments (sentences). ~42 gives subtitle-sized lines.");
    ImGui::BeginDisabled(usePk);
    ImGui::SameLine(0, 24);
    ImGui::Checkbox("Translate to English", &s.translate);
    ImGui::SameLine(0, 24);
    ImGui::Checkbox("Use previous text as context", &s.useContext);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Usually more coherent. Turn OFF if the output gets stuck repeating itself.");

    ImGui::SetNextItemWidth(-FLT_MIN);
    ImGui::InputTextWithHint("##prompt", "Optional prompt: names, jargon, e.g. \"Dr. Nguyen, Kubernetes, GPU.\"",
                             &s.prompt);
    ImGui::EndDisabled();
    if (usePk)
        ImGui::TextDisabled("Parakeet: language is detected automatically; prompt, beam search and translation are Whisper-only.");
    ImGui::EndDisabled();

    // --- run controls
    ImGui::Separator();
    if (!running) {
        if (ImGui::Button("Transcribe", ImVec2(130, 0))) start_job(app);
    } else {
        if (ImGui::Button("Cancel", ImVec2(130, 0))) job.abort = true;
    }
    ImGui::SameLine();
    ImGui::ProgressBar(job.progress / 100.0f, ImVec2(220, 0));
    ImGui::SameLine();
    {
        std::lock_guard<std::mutex> lk(job.mtx);
        ImGui::TextUnformatted(job.status.c_str());
        if (!job.lang.empty()) { ImGui::SameLine(); ImGui::TextDisabled("(detected: %s)", job.lang.c_str()); }
    }

    // --- results table
    float footer = ImGui::GetFrameHeightWithSpacing() * 2.2f;
    if (ImGui::BeginTable("segs", 3,
            ImGuiTableFlags_ScrollY | ImGuiTableFlags_RowBg | ImGuiTableFlags_Borders |
            ImGuiTableFlags_Resizable, ImVec2(0, -footer))) {
        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableSetupColumn("Start", ImGuiTableColumnFlags_WidthFixed, 110);
        ImGui::TableSetupColumn("End",   ImGuiTableColumnFlags_WidthFixed, 110);
        ImGui::TableSetupColumn("Text",  ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableHeadersRow();
        {
            std::lock_guard<std::mutex> lk(job.mtx);
            for (auto& seg : job.segs) {
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0); ImGui::TextUnformatted(fmt_ts(seg.t0, '.').c_str());
                ImGui::TableSetColumnIndex(1); ImGui::TextUnformatted(fmt_ts(seg.t1, '.').c_str());
                ImGui::TableSetColumnIndex(2); ImGui::TextWrapped("%s", seg.text.c_str());
            }
        }
        if (running && ImGui::GetScrollY() >= ImGui::GetScrollMaxY() - 40.0f)
            ImGui::SetScrollHereY(1.0f);
        ImGui::EndTable();
    }

    // --- export
    bool haveSegs;
    { std::lock_guard<std::mutex> lk(job.mtx); haveSegs = !job.segs.empty(); }
    ImGui::BeginDisabled(!haveSegs || running);
    if (ImGui::Button("Export SRT")) do_export(app, Fmt::SRT);
    ImGui::SameLine(); if (ImGui::Button("Export VTT")) do_export(app, Fmt::VTT);
    ImGui::SameLine(); if (ImGui::Button("Export TXT")) do_export(app, Fmt::TXT);
    ImGui::SameLine();
    if (ImGui::Button("Copy all")) {
        std::lock_guard<std::mutex> lk(job.mtx);
        ImGui::SetClipboardText(build_export(job.segs, Fmt::TXT).c_str());
        app.message = "Copied to clipboard.";
    }
    ImGui::EndDisabled();
    ImGui::SameLine(0, 20);
    ImGui::TextUnformatted(app.message.c_str());

    draw_download_popup(app);

    ImGui::End();
}

int main() {
    if (!glfwInit()) return 1;
#ifdef __APPLE__
    const char* glsl = "#version 150";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#else
    const char* glsl = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
#endif
    GLFWwindow* win = glfwCreateWindow(1100, 760, "WhisperScribe", nullptr, nullptr);
    if (!win) { glfwTerminate(); return 1; }
    glfwMakeContextCurrent(win);
    glfwSwapInterval(1);
    apply_window_icon(win);

    NFD_Init();

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = nullptr;
    load_font(io);
    ImGui::StyleColorsDark();
    ImGui::GetStyle().FrameRounding = 4.0f;
    ImGui_ImplGlfw_InitForOpenGL(win, true);
    ImGui_ImplOpenGL3_Init(glsl);

    App app;
    g_app = &app;
#ifdef WITH_PARAKEET
    pk::autodetect(app.s.pkDir, app.s.pkVad);   // fills paths from a nearby "models" folder
#endif
    app.s.threads = (int)std::min(8u, std::max(1u, std::thread::hardware_concurrency()));
    glfwSetDropCallback(win, drop_cb);

    while (!glfwWindowShouldClose(win)) {
        glfwPollEvents();
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        draw_ui(app);

        ImGui::Render();
        int w, h;
        glfwGetFramebufferSize(win, &w, &h);
        glViewport(0, 0, w, h);
        glClearColor(0.1f, 0.1f, 0.12f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(win);
    }

    // shutdown: stop worker cleanly, then free the model
    app.downloader.cancel();
    if (app.job.th.joinable()) { app.job.abort = true; app.job.th.join(); }
    if (app.ctx) whisper_free(app.ctx);
#ifdef WITH_PARAKEET
    pk::shutdown();
#endif

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    NFD_Quit();
    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}
