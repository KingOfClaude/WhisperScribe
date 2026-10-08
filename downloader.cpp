#include "downloader.h"

#ifdef _WIN32
  #ifndef NOMINMAX
    #define NOMINMAX
  #endif
  #ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
  #endif
  #include <windows.h>
#else
  #include <errno.h>
  #include <fcntl.h>
  #include <poll.h>
  #include <signal.h>
  #include <sys/wait.h>
  #include <unistd.h>
#endif

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <vector>

namespace fs = std::filesystem;

namespace dl {
namespace {

// ----------------------------------------------------------------------------
// Catalog
// ----------------------------------------------------------------------------
const WhisperModel kWhisper[] = {
    { "large-v3",            "ggml-large-v3.bin",            "~3.1 GB", "Most accurate. Needs about 4-5 GB of GPU memory." },
    { "large-v3-turbo",      "ggml-large-v3-turbo.bin",      "~1.6 GB", "Nearly as accurate, much faster." },
    { "large-v3-q5_0",       "ggml-large-v3-q5_0.bin",       "~1.1 GB", "large-v3 compressed: smaller, tiny accuracy loss." },
    { "large-v3-turbo-q5_0", "ggml-large-v3-turbo-q5_0.bin", "~0.6 GB", "Smallest of the large family." },
    { "medium",              "ggml-medium.bin",              "~1.5 GB", "Good middle ground." },
    { "medium.en",           "ggml-medium.en.bin",           "~1.5 GB", "English only; slightly better than medium for English." },
    { "small",               "ggml-small.bin",               "~0.5 GB", "Fast, noticeably less accurate." },
    { "small.en",            "ggml-small.en.bin",            "~0.5 GB", "English only." },
    { "base",                "ggml-base.bin",                "~140 MB", "Quick tests only." },
    { "tiny",                "ggml-tiny.bin",                "~75 MB",  "Quick tests only." },
};

const ParakeetVariant kParakeet[] = {
    { "v3", "v3  (English + 24 European languages)", "Detects the language automatically. Download is roughly 0.7 GB." },
    { "v2", "v2  (English only)",                    "English only. Similar size." },
};

// Non-const so tests can point them at local files.
std::string   g_hfBase    = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/";
std::string   g_sherpaBase = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/";
std::uintmax_t g_minWhisperBytes = 50ull * 1024 * 1024;   // even "tiny" is ~75 MB
std::uintmax_t g_minVadBytes     = 100ull * 1024;         // silero_vad.onnx is ~2 MB

// ----------------------------------------------------------------------------
// Paths
// ----------------------------------------------------------------------------
#ifdef _WIN32
std::wstring utf8_to_wide(const std::string& s) {
    if (s.empty()) return L"";
    int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), nullptr, 0);
    std::wstring w(n, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), &w[0], n);
    return w;
}
std::string wide_to_utf8(const std::wstring& w) {
    if (w.empty()) return "";
    int n = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), (int)w.size(), nullptr, 0, nullptr, nullptr);
    std::string s(n, '\0');
    WideCharToMultiByte(CP_UTF8, 0, w.c_str(), (int)w.size(), &s[0], n, nullptr, nullptr);
    return s;
}
#endif

fs::path exe_dir() {
#ifdef _WIN32
    wchar_t buf[MAX_PATH * 2];
    DWORD cap = (DWORD)(sizeof(buf) / sizeof(buf[0]));
    DWORD n = GetModuleFileNameW(nullptr, buf, cap);
    if (n > 0 && n < cap) return fs::path(buf).parent_path();
#else
    std::error_code ec;
    fs::path p = fs::read_symlink("/proc/self/exe", ec);
    if (!ec) return p.parent_path();
#endif
    std::error_code ec2;
    return fs::current_path(ec2);
}

// Finds a command-line tool. On Windows prefer the copy that ships with the OS
// (System32\curl.exe, System32\tar.exe), because other "tar.exe"s on PATH can choke on "C:\".
std::string find_tool(const std::string& name) {
#ifdef _WIN32
    wchar_t sys[MAX_PATH];
    UINT n = GetSystemDirectoryW(sys, MAX_PATH);
    if (n > 0 && n < MAX_PATH) {
        std::error_code ec;
        fs::path p = fs::path(sys) / (name + ".exe");
        if (fs::exists(p, ec)) return p.u8string();
    }
    wchar_t buf[MAX_PATH * 2];
    DWORD cap = (DWORD)(sizeof(buf) / sizeof(buf[0]));
    DWORD n2 = SearchPathW(nullptr, utf8_to_wide(name + ".exe").c_str(), nullptr, cap, buf, nullptr);
    if (n2 > 0 && n2 < cap) return wide_to_utf8(buf);
    return "";
#else
    return name;   // resolved through PATH by execvp
#endif
}

// ----------------------------------------------------------------------------
// Child process (hidden window), merged stdout+stderr delivered through onData
// ----------------------------------------------------------------------------
using DataCb = std::function<void(const char*, size_t)>;

#ifdef _WIN32
std::string quote_arg(const std::string& a) {
    if (!a.empty() && a.find_first_of(" \t\"") == std::string::npos) return a;
    std::string r = "\"";
    for (char c : a) { if (c == '"') r += "\\\""; else r += c; }
    r += "\"";
    return r;
}

bool run_process(const std::vector<std::string>& args, const std::string& cwd,
                 const DataCb& onData, const std::atomic<bool>& abort, int& exitCode) {
    std::string cmd;
    for (size_t i = 0; i < args.size(); ++i) { if (i) cmd += ' '; cmd += quote_arg(args[i]); }

    SECURITY_ATTRIBUTES sa{};
    sa.nLength = sizeof sa;
    sa.bInheritHandle = TRUE;
    HANDLE rd = nullptr, wr = nullptr;
    if (!CreatePipe(&rd, &wr, &sa, 0)) return false;
    SetHandleInformation(rd, HANDLE_FLAG_INHERIT, 0);
    HANDLE nul = CreateFileW(L"NUL", GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, &sa,
                             OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);

    STARTUPINFOW si{};
    si.cb = sizeof si;
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdInput = nul;
    si.hStdOutput = wr;
    si.hStdError = wr;
    PROCESS_INFORMATION pi{};

    std::wstring w = utf8_to_wide(cmd);
    std::vector<wchar_t> buf(w.begin(), w.end());
    buf.push_back(L'\0');
    std::wstring wcwd = utf8_to_wide(cwd);
    BOOL ok = CreateProcessW(nullptr, buf.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW,
                             nullptr, wcwd.empty() ? nullptr : wcwd.c_str(), &si, &pi);
    CloseHandle(wr);
    if (nul != INVALID_HANDLE_VALUE) CloseHandle(nul);
    if (!ok) { CloseHandle(rd); return false; }

    char chunk[4096];
    for (;;) {
        DWORD avail = 0;
        BOOL pk = PeekNamedPipe(rd, nullptr, 0, nullptr, &avail, nullptr);
        if (pk && avail > 0) {
            DWORD n = 0;
            if (ReadFile(rd, chunk, std::min<DWORD>(avail, (DWORD)sizeof chunk), &n, nullptr) && n > 0)
                onData(chunk, (size_t)n);
            continue;
        }
        if (!pk) break;   // pipe closed: finished and fully read
        if (WaitForSingleObject(pi.hProcess, 50) == WAIT_OBJECT_0) {
            avail = 0;
            if (PeekNamedPipe(rd, nullptr, 0, nullptr, &avail, nullptr) && avail > 0) continue;
            break;
        }
        if (abort.load()) TerminateProcess(pi.hProcess, 1);
    }

    WaitForSingleObject(pi.hProcess, 3000);
    DWORD code = 1;
    GetExitCodeProcess(pi.hProcess, &code);
    exitCode = (int)code;
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    CloseHandle(rd);
    return true;
}

#else  // POSIX

bool run_process(const std::vector<std::string>& args, const std::string& cwd,
                 const DataCb& onData, const std::atomic<bool>& abort, int& exitCode) {
    int pfd[2];
    if (pipe(pfd) != 0) return false;
    pid_t pid = fork();
    if (pid < 0) { close(pfd[0]); close(pfd[1]); return false; }
    if (pid == 0) {                                   // child
        dup2(pfd[1], 1);
        dup2(pfd[1], 2);
        close(pfd[0]);
        close(pfd[1]);
        int nul = open("/dev/null", O_RDONLY);
        if (nul >= 0) { dup2(nul, 0); close(nul); }
        if (!cwd.empty() && chdir(cwd.c_str()) != 0) _exit(126);
        std::vector<char*> argv;
        for (auto& a : args) argv.push_back(const_cast<char*>(a.c_str()));
        argv.push_back(nullptr);
        execvp(argv[0], argv.data());
        _exit(127);
    }
    close(pfd[1]);

    bool killed = false;
    bool gotData = false;
    char buf[4096];
    for (;;) {
        struct pollfd p{ pfd[0], POLLIN, 0 };
        int pr = poll(&p, 1, 100);
        if (abort.load() && !killed) { kill(pid, SIGTERM); killed = true; }
        if (pr > 0) {
            ssize_t n = read(pfd[0], buf, sizeof buf);
            if (n > 0) { gotData = true; onData(buf, (size_t)n); }
            else if (n == 0) break;
            else if (errno != EINTR) break;
        } else if (pr < 0 && errno != EINTR) {
            break;
        }
    }
    close(pfd[0]);
    int status = 0;
    waitpid(pid, &status, 0);
    exitCode = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    if (exitCode == 127 && !gotData) return false;    // exec failed: tool not found
    return true;
}
#endif

// ----------------------------------------------------------------------------
// curl progress / errors
// ----------------------------------------------------------------------------
// curl's --progress-bar output looks like "\r####   45.3%". Returns the last percentage
// found in s, or -1.
double last_percent(const std::string& s) {
    size_t pos = s.rfind('%');
    while (pos != std::string::npos) {
        size_t b = pos;
        while (b > 0 && (std::isdigit((unsigned char)s[b - 1]) || s[b - 1] == '.')) --b;
        if (b < pos) return atof(s.substr(b, pos - b).c_str());
        if (pos == 0) break;
        pos = s.rfind('%', pos - 1);
    }
    return -1;
}

std::string last_curl_error(const std::string& s) {
    size_t k = s.rfind("curl:");
    if (k == std::string::npos) return "";
    size_t e = s.find_first_of("\r\n", k);
    std::string line = s.substr(k, e == std::string::npos ? std::string::npos : e - k);
    return line;
}

std::string describe_curl_failure(int code, const std::string& curlLine) {
    std::string msg = "Download failed";
    if (!curlLine.empty()) msg += " (" + curlLine + ")";
    else msg += " (curl exit code " + std::to_string(code) + ")";
    if (code == 6 || code == 7 || code == 28 || code == 35 || code == 56)
        msg += ". Check your internet connection and try again.";
    else if (code == 23)
        msg += ". Could not write the file - is the disk full or the folder read-only?";
    return msg;
}

// Downloads url to dest (via dest + ".part", resuming if a partial file exists).
// Returns false on failure/cancel with a message in err.
bool fetch_file(const std::string& url, const fs::path& dest, const std::atomic<bool>& abort,
                const std::function<void(double)>& onPercent, std::string& err) {
    std::string curl = find_tool("curl");
    if (curl.empty()) {
        err = "curl.exe was not found. It ships with Windows 10 and newer.";
        return false;
    }
    std::error_code ec;
    fs::create_directories(dest.parent_path(), ec);
    fs::path part = dest;
    part += ".part";
    const bool hadPart = fs::exists(part, ec);

    for (int attempt = 0; attempt < 2; ++attempt) {
        std::vector<std::string> args = { curl, "-L", "--fail", "--connect-timeout", "30",
                                          "--retry", "3", "--retry-delay", "2", "--progress-bar" };
        if (attempt == 0) { args.push_back("-C"); args.push_back("-"); }   // resume
        args.push_back("-o"); args.push_back(part.u8string());
        args.push_back(url);

        std::string tail;
        auto onData = [&](const char* d, size_t n) {
            tail.append(d, n);
            if (tail.size() > 600) tail.erase(0, tail.size() - 600);
            double p = last_percent(tail);
            if (p >= 0) onPercent(std::min(100.0, p));
        };

        int code = -1;
        bool started = run_process(args, "", onData, abort, code);
        if (abort.load()) { err = "Cancelled."; return false; }
        if (!started) { err = "Could not start curl."; return false; }

        if (code == 0) {
            std::error_code ec2;
            fs::remove(dest, ec2);
            fs::rename(part, dest, ec2);
            if (ec2) { err = "Could not move the finished download into place: " + ec2.message(); return false; }
            onPercent(100.0);
            return true;
        }

        err = describe_curl_failure(code, last_curl_error(tail));
        if (attempt == 0 && hadPart) {      // a stale partial file may be the problem: start clean once
            fs::remove(part, ec);
            continue;
        }
        break;
    }
    return false;
}

bool dir_has_prefix(const fs::path& dir, const std::string& prefix) {
    std::error_code ec;
    for (fs::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec))
        if (it->path().filename().u8string().rfind(prefix, 0) == 0) return true;
    return false;
}

bool parakeet_complete(const fs::path& dir) {
    std::error_code ec;
    return fs::exists(dir / "tokens.txt", ec) && dir_has_prefix(dir, "encoder") &&
           dir_has_prefix(dir, "decoder") && dir_has_prefix(dir, "joiner");
}

}  // namespace

// ----------------------------------------------------------------------------
// Public helpers
// ----------------------------------------------------------------------------
int whisper_model_count() { return (int)(sizeof(kWhisper) / sizeof(kWhisper[0])); }
const WhisperModel& whisper_model(int i) { return kWhisper[std::clamp(i, 0, whisper_model_count() - 1)]; }
int parakeet_variant_count() { return (int)(sizeof(kParakeet) / sizeof(kParakeet[0])); }
const ParakeetVariant& parakeet_variant(int i) { return kParakeet[std::clamp(i, 0, parakeet_variant_count() - 1)]; }

std::string models_dir() {
    std::error_code ec;
    std::vector<fs::path> starts = { exe_dir(), fs::current_path(ec) };
    for (fs::path base : starts) {
        for (int up = 0; up < 6 && !base.empty(); ++up, base = base.parent_path()) {
            fs::path m = base / "models";
            if (fs::is_directory(m, ec)) return m.u8string();
        }
    }
    return (exe_dir() / "models").u8string();
}

// ----------------------------------------------------------------------------
// Downloader
// ----------------------------------------------------------------------------
Downloader::~Downloader() {
    abort_ = true;
    if (th_.joinable()) th_.join();
}

std::string Downloader::status() const {
    std::lock_guard<std::mutex> lk(mtx_);
    return status_;
}

void Downloader::set_status(const std::string& s) {
    std::lock_guard<std::mutex> lk(mtx_);
    status_ = s;
}

void Downloader::finish(const Result& r) {
    {
        std::lock_guard<std::mutex> lk(mtx_);
        result_ = r;
    }
    finished_ = true;   // set last: poll() reads result_ only after seeing this
}

bool Downloader::poll(Result& out) {
    if (!finished_) return false;
    if (th_.joinable()) th_.join();
    running_ = false;
    finished_ = false;
    std::lock_guard<std::mutex> lk(mtx_);
    out = result_;
    return true;
}

void Downloader::start_job(int kind, int index) {
    if (running_) return;
    if (th_.joinable()) th_.join();
    abort_ = false;
    percent_ = 0;
    finished_ = false;
    running_ = true;
    set_status("Starting...");
    if (kind == 0) th_ = std::thread([this, index] { run_whisper(index); });
    else           th_ = std::thread([this, index] { run_parakeet(index); });
}

void Downloader::start_whisper(int modelIndex)    { start_job(0, modelIndex); }
void Downloader::start_parakeet(int variantIndex) { start_job(1, variantIndex); }

void Downloader::run_whisper(int index) {
    Result res;
    const WhisperModel& m = whisper_model(index);
    std::error_code ec;
    fs::path dir = fs::u8path(models_dir());
    fs::create_directories(dir, ec);
    fs::path dest = dir / m.file;
    res.path = dest.u8string();

    auto sz = fs::exists(dest, ec) ? fs::file_size(dest, ec) : 0;
    if (!ec && sz > g_minWhisperBytes) {
        percent_ = 100;
        res.ok = true;
        res.message = "Already downloaded.";
        finish(res);
        return;
    }

    set_status(std::string("Downloading ") + m.file + " (" + m.size + ")...");
    std::string err;
    bool ok = fetch_file(g_hfBase + m.file, dest, abort_,
                         [this](double p) { percent_ = (int)p; }, err);
    if (ok) {
        auto got = fs::file_size(dest, ec);
        if (ec || got < g_minWhisperBytes) {
            fs::remove(dest, ec);
            ok = false;
            err = "The downloaded file is too small - probably an error page instead of the model.";
        }
    }
    res.ok = ok;
    res.cancelled = abort_.load();
    res.error = err;
    if (ok) res.message = "Downloaded.";
    finish(res);
}

void Downloader::run_parakeet(int index) {
    Result res;
    const ParakeetVariant& v = parakeet_variant(index);
    std::error_code ec;
    fs::path dir = fs::u8path(models_dir());
    fs::create_directories(dir, ec);

    const std::string name = std::string("sherpa-onnx-nemo-parakeet-tdt-0.6b-") + v.id + "-int8";
    fs::path modelDir = dir / name;
    fs::path vad = dir / "silero_vad.onnx";
    fs::path archive = dir / (name + ".tar.bz2");
    res.dir = modelDir.u8string();
    res.vad = vad.u8string();

    auto fail = [&](const std::string& err) {
        res.ok = false;
        res.cancelled = abort_.load();
        res.error = err;
        finish(res);
    };
    auto scaled = [this](int lo, int hi) {
        return [this, lo, hi](double p) { percent_ = lo + (int)((hi - lo) * p / 100.0); };
    };

    // 1) Silero VAD (small)
    auto vadSize = fs::exists(vad, ec) ? fs::file_size(vad, ec) : 0;
    if (ec || vadSize < g_minVadBytes) {
        set_status("Downloading Silero VAD (small)...");
        std::string err;
        if (!fetch_file(g_sherpaBase + "silero_vad.onnx", vad, abort_, scaled(0, 5), err)) return fail(err);
    }
    percent_ = 5;

    // 2) the model
    bool already = parakeet_complete(modelDir);
    if (!already) {
        if (!fs::exists(archive, ec)) {
            set_status(std::string("Downloading Parakeet ") + v.id + " (roughly 0.7 GB)...");
            std::string err;
            if (!fetch_file(g_sherpaBase + name + ".tar.bz2", archive, abort_, scaled(5, 92), err))
                return fail(err);
        }

        set_status("Extracting...");
        percent_ = 95;
        std::string tar = find_tool("tar");
        if (tar.empty()) return fail("tar.exe was not found. It ships with Windows 10 and newer.");

        std::string out;
        int code = -1;
        bool started = run_process({ tar, "-xjf", archive.filename().u8string() }, dir.u8string(),
                                   [&](const char* d, size_t n) {
                                       out.append(d, n);
                                       if (out.size() > 400) out.erase(0, out.size() - 400);
                                   },
                                   abort_, code);
        if (abort_.load() || !started || code != 0 || !parakeet_complete(modelDir)) {
            fs::remove_all(modelDir, ec);                  // never leave a half-extracted model behind
            if (abort_.load()) return fail("Cancelled.");
            fs::remove(archive, ec);                       // likely a corrupt archive: re-download next time
            return fail(started ? "Extracting the model archive failed. " + out
                                : "Could not start tar to extract the archive.");
        }
        fs::remove(archive, ec);
    }

    percent_ = 100;
    res.ok = true;
    res.message = already ? "Already downloaded." : "Downloaded.";
    finish(res);
}

}  // namespace dl
