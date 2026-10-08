// Parakeet (NVIDIA NeMo TDT) backend for WhisperScribe, running through sherpa-onnx.
// Pipeline: Silero VAD splits the audio into speech chunks -> Parakeet transcribes each
// chunk -> token timestamps are regrouped into sentence-sized segments.
#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace pk {

struct Config {
    std::string modelDir;       // folder holding encoder/decoder/joiner .onnx + tokens.txt
    std::string vadModel;       // path to silero_vad.onnx
    int threads  = 4;
    int maxChars = 0;           // 0 = split on sentence ends; >0 = also cap segment length
};

// Times are in centiseconds (1 = 10 ms), the same unit the Whisper path uses.
using SegmentCb  = std::function<void(int64_t t0, int64_t t1, const std::string& text)>;
using ProgressCb = std::function<void(int percent)>;
using StatusCb   = std::function<void(const std::string& text)>;

// Transcribe 16 kHz mono float PCM. Returns false on error (message in err).
// On cancel it returns true with whatever was produced so far.
bool transcribe(const std::vector<float>& pcm16k, const Config& cfg,
                const SegmentCb& onSegment, const ProgressCb& onProgress,
                const StatusCb& onStatus, const std::atomic<bool>& abort,
                std::string& err);

// Frees the cached recognizer. Call once at program exit (after the worker is joined).
void shutdown();

// UI helpers ------------------------------------------------------------------
// true if dir contains encoder/decoder/joiner .onnx and tokens.txt; msg explains if not.
bool check_model_dir(const std::string& dir, std::string& msg);

// Looks for a "models" folder next to the exe (or up to a few levels above it) and fills in
// the Parakeet model folder and silero_vad.onnx if found. Leaves outputs untouched otherwise.
void autodetect(std::string& modelDir, std::string& vadModel);

}  // namespace pk
