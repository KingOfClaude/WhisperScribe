// In-app model downloader for WhisperScribe.
// Runs the system's own curl (and tar for the Parakeet archive) in a background thread, so
// the GUI stays responsive, shows progress and can cancel. Windows 10+ ships both tools.
#pragma once

#include <atomic>
#include <mutex>
#include <string>
#include <thread>

namespace dl {

// ---- what can be downloaded -------------------------------------------------------
struct WhisperModel {
    const char* id;     // "large-v3"
    const char* file;   // "ggml-large-v3.bin"
    const char* size;   // "~3.1 GB"
    const char* note;
};
int                 whisper_model_count();
const WhisperModel& whisper_model(int index);

struct ParakeetVariant {
    const char* id;     // "v3"
    const char* label;
    const char* note;
};
int                    parakeet_variant_count();
const ParakeetVariant& parakeet_variant(int index);

// ---- result of a finished job -------------------------------------------------------
struct Result {
    bool ok = false;
    bool cancelled = false;
    std::string error;      // when !ok && !cancelled
    std::string message;    // when ok, e.g. "Downloaded." / "Already downloaded."
    std::string path;       // Whisper: the model file
    std::string dir;        // Parakeet: model folder
    std::string vad;        // Parakeet: silero_vad.onnx
};

// Folder downloads are saved to: an existing "models" folder next to the exe (or a few
// levels above it, same search as the Parakeet auto-detect), else <exe folder>\models.
std::string models_dir();

class Downloader {
public:
    Downloader() = default;
    ~Downloader();
    Downloader(const Downloader&) = delete;
    Downloader& operator=(const Downloader&) = delete;

    bool busy() const { return running_; }          // true from start until poll() reports the result
    void start_whisper(int modelIndex);
    void start_parakeet(int variantIndex);
    void cancel() { abort_ = true; }

    // Call from the UI thread every frame. Returns true exactly once per finished job.
    bool poll(Result& out);

    int         percent() const { return percent_; }   // 0..100
    std::string status() const;

private:
    void run_whisper(int index);
    void run_parakeet(int index);
    void start_job(int kind, int index);
    void finish(const Result& r);
    void set_status(const std::string& s);

    std::atomic<bool> running_{false};
    std::atomic<bool> finished_{false};
    std::atomic<bool> abort_{false};
    std::atomic<int>  percent_{0};
    mutable std::mutex mtx_;
    std::string status_;
    Result result_;
    std::thread th_;
};

}  // namespace dl
