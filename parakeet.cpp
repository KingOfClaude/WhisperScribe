#include "parakeet.h"

#ifdef _WIN32
  #ifndef NOMINMAX
    #define NOMINMAX
  #endif
  #ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
  #endif
  #include <windows.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>

#include "sherpa-onnx/c-api/c-api.h"

namespace fs = std::filesystem;

namespace pk {
namespace {

// ----------------------------------------------------------------------------
// Recognizer cache (loading the 0.6B model takes a second or two)
// ----------------------------------------------------------------------------
struct Cache {
    const SherpaOnnxOfflineRecognizer* rec = nullptr;
    std::string key;
};
Cache g_cache;

// ----------------------------------------------------------------------------
// Model folder handling
// ----------------------------------------------------------------------------
std::string find_onnx(const fs::path& dir, const std::string& prefix) {
    std::error_code ec;
    std::string plain, int8;
    for (fs::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec)) {
        if (!it->is_regular_file(ec)) continue;
        const fs::path& p = it->path();
        std::string name = p.filename().u8string();
        if (name.rfind(prefix, 0) != 0 || p.extension() != ".onnx") continue;
        if (name.find("int8") != std::string::npos) int8 = p.u8string();
        else plain = p.u8string();
    }
    return !int8.empty() ? int8 : plain;   // prefer the int8 files when both exist
}

struct ModelFiles { std::string enc, dec, join, tokens; };

bool locate(const std::string& dir, ModelFiles& m, std::string& msg) {
    std::error_code ec;
    fs::path d = fs::u8path(dir);
    if (dir.empty() || !fs::is_directory(d, ec)) { msg = "Folder not found."; return false; }
    m.enc  = find_onnx(d, "encoder");
    m.dec  = find_onnx(d, "decoder");
    m.join = find_onnx(d, "joiner");
    fs::path t = d / "tokens.txt";
    if (fs::exists(t, ec)) m.tokens = t.u8string();

    std::string missing;
    if (m.enc.empty())    missing += " encoder*.onnx";
    if (m.dec.empty())    missing += " decoder*.onnx";
    if (m.join.empty())   missing += " joiner*.onnx";
    if (m.tokens.empty()) missing += " tokens.txt";
    if (!missing.empty()) { msg = "Missing:" + missing; return false; }
    msg = "OK";
    return true;
}

// ----------------------------------------------------------------------------
// Tokens -> words -> sentence-sized segments
// ----------------------------------------------------------------------------
struct Word {
    std::string text;
    double t0 = 0, t1 = 0;   // seconds, relative to the chunk that was decoded
};

const char kMarker[] = "\xE2\x96\x81";   // U+2581, SentencePiece "word start"

std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    return s.substr(a, s.find_last_not_of(" \t\r\n") - a + 1);
}

std::vector<Word> build_words(const SherpaOnnxOfflineRecognizerResult* r, double chunkDur) {
    std::vector<Word> words;
    const int n = (int)r->count;
    for (int i = 0; i < n; ++i) {
        const char* tok = r->tokens_arr[i];
        if (!tok) continue;
        double t0 = r->timestamps[i];
        double t1 = r->durations ? t0 + r->durations[i] : t0;
        bool marker = std::strncmp(tok, kMarker, 3) == 0;
        const char* body = marker ? tok + 3 : tok;
        if (words.empty() || marker) {
            Word w;
            w.text = body;
            w.t0 = t0;
            w.t1 = t1;
            words.push_back(w);
        } else {
            words.back().text += body;
            words.back().t1 = std::max(words.back().t1, t1);
        }
    }
    words.erase(std::remove_if(words.begin(), words.end(),
                               [](const Word& w) { return w.text.empty(); }),
                words.end());

    // make end times sane: never past the next word's start, never before our own start
    for (size_t i = 0; i < words.size(); ++i) {
        double next = (i + 1 < words.size()) ? words[i + 1].t0 : chunkDur;
        if (!r->durations || words[i].t1 <= words[i].t0) words[i].t1 = next;
        words[i].t1 = std::min(words[i].t1, std::max(next, words[i].t0));
        words[i].t1 = std::max(words[i].t1, words[i].t0);
    }
    return words;
}

bool ends_sentence(const std::string& w) {
    if (w.empty()) return false;
    static const char* abbrev[] = { "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Sr.", "Jr.", "St.", "vs." };
    for (const char* a : abbrev) if (w == a) return false;
    char c = w.back();
    if (c == '.' || c == '?' || c == '!') return true;
    if (w.size() >= 3 && w.compare(w.size() - 3, 3, "\xE2\x80\xA6") == 0) return true;  // ellipsis
    return false;
}

void emit_segments(const std::vector<Word>& words, double offset, int maxChars,
                   const SegmentCb& cb) {
    std::string text;
    double t0 = 0, t1 = 0;
    size_t count = 0;

    auto flush = [&]() {
        if (text.empty()) { count = 0; return; }
        int64_t a = (int64_t)std::llround((offset + t0) * 100.0);
        int64_t b = (int64_t)std::llround((offset + t1) * 100.0);
        if (b < a) b = a;
        cb(a, b, text);
        text.clear();
        count = 0;
    };

    const size_t limit = maxChars > 0 ? (size_t)maxChars : 220;
    for (const Word& w : words) {
        if (count > 0) {
            if (text.size() + 1 + w.text.size() > limit) flush();   // too long
            else if (w.t0 - t1 > 1.0) flush();                       // long pause
        }
        if (count == 0) t0 = w.t0; else text += ' ';
        text += w.text;
        t1 = std::max(w.t1, w.t0);
        ++count;
        if (ends_sentence(w.text)) flush();
    }
    flush();
}

fs::path exe_dir() {
#ifdef _WIN32
    wchar_t buf[MAX_PATH * 2];
    DWORD cap = (DWORD)(sizeof(buf) / sizeof(buf[0]));
    DWORD n = GetModuleFileNameW(nullptr, buf, cap);
    if (n > 0 && n < cap) return fs::path(buf).parent_path();
#endif
    std::error_code ec;
    return fs::current_path(ec);
}

}  // namespace

// ----------------------------------------------------------------------------
// Public API
// ----------------------------------------------------------------------------
void shutdown() {
    if (g_cache.rec) {
        SherpaOnnxDestroyOfflineRecognizer(g_cache.rec);
        g_cache.rec = nullptr;
    }
    g_cache.key.clear();
}

bool check_model_dir(const std::string& dir, std::string& msg) {
    ModelFiles m;
    return locate(dir, m, msg);
}

void autodetect(std::string& modelDir, std::string& vadModel) {
    std::error_code ec;
    std::vector<fs::path> starts = { exe_dir(), fs::current_path(ec) };
    for (fs::path base : starts) {
        for (int up = 0; up < 6 && !base.empty(); ++up, base = base.parent_path()) {
            fs::path models = base / "models";
            if (!fs::is_directory(models, ec)) continue;

            std::string best;
            for (fs::directory_iterator it(models, ec), end; !ec && it != end; it.increment(ec)) {
                if (!it->is_directory(ec)) continue;
                std::string name = it->path().filename().u8string();
                if (name.rfind("sherpa-onnx-nemo-parakeet", 0) != 0) continue;
                ModelFiles mf;
                std::string m;
                if (!locate(it->path().u8string(), mf, m)) continue;
                if (best.empty() || name.find("v3") != std::string::npos)
                    best = it->path().u8string();
            }
            if (!best.empty() && modelDir.empty()) modelDir = best;

            fs::path v = models / "silero_vad.onnx";
            if (vadModel.empty() && fs::exists(v, ec)) vadModel = v.u8string();

            if (!modelDir.empty() && !vadModel.empty()) return;
        }
    }
}

bool transcribe(const std::vector<float>& pcm, const Config& cfg,
                const SegmentCb& onSegment, const ProgressCb& onProgress,
                const StatusCb& onStatus, const std::atomic<bool>& abort,
                std::string& err) {
    err.clear();
    const int SR = 16000;

    // ---- validate inputs
    ModelFiles mf;
    std::string msg;
    if (!locate(cfg.modelDir, mf, msg)) { err = "Parakeet model folder: " + msg; return false; }
    std::error_code ec;
    if (cfg.vadModel.empty() || !fs::exists(fs::u8path(cfg.vadModel), ec)) {
        err = "silero_vad.onnx not found. Run download-parakeet.ps1 or browse to the file.";
        return false;
    }
    if (pcm.empty()) { err = "No audio samples."; return false; }

    // ---- recognizer (cached)
    const std::string key = cfg.modelDir + "|" + std::to_string(cfg.threads);
    if (!g_cache.rec || g_cache.key != key) {
        onStatus("Loading Parakeet model...");
        shutdown();
        SherpaOnnxOfflineRecognizerConfig rc;
        std::memset(&rc, 0, sizeof rc);
        rc.model_config.transducer.encoder = mf.enc.c_str();
        rc.model_config.transducer.decoder = mf.dec.c_str();
        rc.model_config.transducer.joiner  = mf.join.c_str();
        rc.model_config.tokens             = mf.tokens.c_str();
        rc.model_config.model_type         = "nemo_transducer";
        rc.model_config.num_threads        = std::max(1, cfg.threads);
        rc.model_config.provider           = "cpu";
        rc.model_config.debug              = 0;
        rc.decoding_method                 = "greedy_search";
        g_cache.rec = SherpaOnnxCreateOfflineRecognizer(&rc);
        if (!g_cache.rec) {
            err = "Could not load the Parakeet model. Check the model folder and that the "
                  "files are intact (avoid non-ASCII characters in the path).";
            return false;
        }
        g_cache.key = key;
    }
    const SherpaOnnxOfflineRecognizer* rec = g_cache.rec;

    // ---- 1) VAD: find speech chunks
    onStatus("Finding speech...");
    SherpaOnnxVadModelConfig vc;
    std::memset(&vc, 0, sizeof vc);
    const std::string vadPath = cfg.vadModel;
    vc.silero_vad.model                = vadPath.c_str();
    vc.silero_vad.threshold            = 0.40f;   // a bit sensitive: missing speech is worse than extra noise
    vc.silero_vad.min_silence_duration = 0.50f;
    vc.silero_vad.min_speech_duration  = 0.25f;
    vc.silero_vad.window_size          = 512;
    vc.silero_vad.max_speech_duration  = 30.0f;   // forces a split on long monologues
    vc.sample_rate                     = SR;
    vc.num_threads                     = 1;
    vc.provider                        = "cpu";
    vc.debug                           = 0;

    const SherpaOnnxVoiceActivityDetector* vad = SherpaOnnxCreateVoiceActivityDetector(&vc, 120.0f);
    if (!vad) { err = "Could not load the VAD model (silero_vad.onnx)."; return false; }

    struct Range { int64_t start; int64_t n; };
    std::vector<Range> ranges;
    auto drain = [&]() {
        while (!SherpaOnnxVoiceActivityDetectorEmpty(vad)) {
            const SherpaOnnxSpeechSegment* s = SherpaOnnxVoiceActivityDetectorFront(vad);
            if (s) ranges.push_back({ (int64_t)s->start, (int64_t)s->n });
            if (s) SherpaOnnxDestroySpeechSegment(s);
            SherpaOnnxVoiceActivityDetectorPop(vad);
        }
    };

    const size_t W = 512, total = pcm.size();
    for (size_t i = 0; i < total && !abort; i += W) {
        size_t n = std::min(W, total - i);
        SherpaOnnxVoiceActivityDetectorAcceptWaveform(vad, pcm.data() + i, (int32_t)n);
        drain();
        if ((i / W) % 4096 == 0) onProgress((int)(10.0 * (double)i / (double)total));
    }
    if (!abort) {
        SherpaOnnxVoiceActivityDetectorFlush(vad);
        drain();
    }
    SherpaOnnxDestroyVoiceActivityDetector(vad);

    if (abort) return true;
    if (ranges.empty()) { onStatus("No speech detected."); onProgress(100); return true; }

    // ---- 2) pad each chunk a little so word edges aren't clipped (without overlapping neighbours)
    const int64_t PAD = (int64_t)(0.12 * SR);
    const int64_t T = (int64_t)total;
    struct Span { int64_t a, b; };
    std::vector<Span> spans(ranges.size());
    for (size_t i = 0; i < ranges.size(); ++i) {
        int64_t s = std::clamp<int64_t>(ranges[i].start, 0, T);
        int64_t e = std::clamp<int64_t>(ranges[i].start + ranges[i].n, 0, T);
        int64_t prevEnd   = (i == 0) ? 0 : std::min<int64_t>(ranges[i - 1].start + ranges[i - 1].n, T);
        int64_t nextStart = (i + 1 == ranges.size()) ? T : std::min<int64_t>(ranges[i + 1].start, T);
        int64_t padBefore = (i == 0) ? std::min(PAD, s) : std::min(PAD, std::max<int64_t>(0, s - prevEnd) / 2);
        int64_t padAfter  = (i + 1 == ranges.size()) ? std::min(PAD, T - e)
                                                     : std::min(PAD, std::max<int64_t>(0, nextStart - e) / 2);
        spans[i] = { s - padBefore, e + padAfter };
    }

    int64_t totalToDecode = 0;
    for (auto& sp : spans) totalToDecode += (sp.b - sp.a);
    if (totalToDecode <= 0) totalToDecode = 1;

    // ---- 3) decode each chunk
    onStatus("Transcribing with Parakeet...");
    int64_t done = 0;
    for (size_t i = 0; i < spans.size(); ++i) {
        if (abort) break;
        const int64_t a = spans[i].a, b = spans[i].b;
        if (b - a < SR / 20) { done += (b - a); continue; }   // < 50 ms: nothing to decode

        const SherpaOnnxOfflineStream* st = SherpaOnnxCreateOfflineStream(rec);
        if (!st) { err = "Could not create a recognition stream."; return false; }
        SherpaOnnxAcceptWaveformOffline(st, SR, pcm.data() + a, (int32_t)(b - a));
        SherpaOnnxDecodeOfflineStream(rec, st);
        const SherpaOnnxOfflineRecognizerResult* r = SherpaOnnxGetOfflineStreamResult(st);

        if (r) {
            const double offset = (double)a / SR;
            const double chunkDur = (double)(b - a) / SR;
            bool emitted = false;
            if (r->count > 0 && r->tokens_arr && r->timestamps) {
                std::vector<Word> words = build_words(r, chunkDur);
                if (!words.empty()) {
                    emit_segments(words, offset, cfg.maxChars, onSegment);
                    emitted = true;
                }
            }
            if (!emitted) {   // no token timing available: one segment for the whole chunk
                std::string text = trim(r->text ? r->text : "");
                if (!text.empty())
                    onSegment((int64_t)std::llround(offset * 100.0),
                              (int64_t)std::llround((offset + chunkDur) * 100.0), text);
            }
            SherpaOnnxDestroyOfflineRecognizerResult(r);
        }
        SherpaOnnxDestroyOfflineStream(st);

        done += (b - a);
        onProgress(10 + (int)(90.0 * (double)done / (double)totalToDecode));
    }
    if (!abort) onProgress(100);
    return true;
}

}  // namespace pk
