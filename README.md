# WhisperScribe

Repetition loops: if long audio gets stuck repeating a phrase, untick “Use previous text as context”.

A desktop app for accurate, timestamped speech-to-text. Pick an audio or video file, choose an engine, and get a transcript with start/end times that you can export as **SRT, VTT or TXT**.

Two engines are built in:

| | **Whisper** (via [whisper.cpp](https://github.com/ggml-org/whisper.cpp)) | **Parakeet** (via [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)) |
|---|---|---|
| Languages | ~99 | English (v2) or English + 24 European languages (v3) |
| Speed | GPU recommended (CUDA / Vulkan) | Very fast, even on CPU |
| Prompt / translate-to-English | Yes | No |
| Language detection | Automatic or manual | Automatic |
| Best for | Maximum accuracy, rare languages, jargon (via prompt) | Fast turnaround on English / European speech, lower accuracy |

Built with C++17, [Dear ImGui](https://github.com/ocornut/imgui) and GLFW. Windows is the primary platform.

![WhisperScribe using Whisper model](https://i.postimg.cc/bvjy1xJt/ex.png)
![WhisperScribe using Parakeet model](https://i.postimg.cc/pLKxVmZh/ex2.png)

## Features

- **Timestamped segments** that appear live as they are decoded
- **Export** to SRT, VTT or TXT (with timestamps), or copy everything to the clipboard
- **Drag & drop** an audio or video file onto the window
- **In-app model downloads**: a **Download...** button next to the model *Browse* button fetches Whisper or Parakeet models with a progress bar and cancel, and selects the model when done. Interrupted downloads resume.
- **Wide format support**: WAV, MP3 and FLAC decode directly; M4A, MP4, MKV, OGG, Opus and more via `ffmpeg` (optional)
- **Cancel anytime**, keeping what has been transcribed so far
- **Whisper options**: language, beam size, thread count, translate to English, previous-text context, initial prompt for names and jargon
- **Segment length control**: set max characters per segment for subtitle-sized lines
- **GPU acceleration** for Whisper: NVIDIA CUDA or Vulkan (AMD / Intel / NVIDIA)

## Quick start (Windows)

### 1. Install the tools (once)

- **Git**, **CMake**, and **Visual Studio** with the *Desktop development with C++* workload (the free Build Tools are enough)
- For an NVIDIA GPU build: the **CUDA Toolkit**. For an AMD / Intel GPU build: the **Vulkan SDK**
- Optional: **ffmpeg** for M4A / MP4 / OGG and similar (`winget install ffmpeg`)

```powershell
winget install Kitware.CMake
winget install Git.Git
```

### 2. Download the models

Models are not included in the repository. The easiest way is inside the app: click **Download...** next to the model's *Browse* button (see step 4). You can also use these scripts, which put the files in `models\`:

```powershell
# Whisper (large-v3 is the default, ~3 GB). Run with -List to see all choices.
powershell -ExecutionPolicy Bypass -File .\download-whisper.ps1
powershell -ExecutionPolicy Bypass -File .\download-whisper.ps1 -Model large-v3-turbo

# Parakeet + Silero VAD (v3 by default; use -Model v2 for English only)
powershell -ExecutionPolicy Bypass -File .\download-parakeet.ps1
```

### 3. Build

```powershell
# NVIDIA GPU (detects your GPU's compute capability automatically)
powershell -ExecutionPolicy Bypass -File .\build-cuda.ps1

# AMD / Intel / NVIDIA via Vulkan
powershell -ExecutionPolicy Bypass -File .\build-vulkan.ps1
```

The first build downloads all dependencies and compiles them; CUDA builds in particular take 10-20 minutes. Add `-Clean` to wipe the build folder and start over.

The executable ends up in `build-cuda\Release\whisper_scribe.exe` (or `build-vulkan\Release\`).

### 4. Run

Start `whisper_scribe.exe`, then:

1. Choose the **Engine**.
2. **Whisper**: click **Download...** next to **Model** to fetch one (large-v3-turbo is a good balance), or *Browse* to a `ggml-*.bin` file you already have.
   **Parakeet**: click **Download...** next to **Parakeet folder**; it fetches the model and the small VAD file together. If a `models\` folder with both is already next to the build folder they are filled in automatically; otherwise browse to them.
3. Choose an audio or video file, or drop one onto the window.
4. Click **Transcribe**, then export from the buttons at the bottom.

## Building by hand

```powershell
cmake -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

| Option | Effect |
|---|---|
| `-DGGML_CUDA=ON` | NVIDIA GPU support for Whisper |
| `-DCMAKE_CUDA_ARCHITECTURES=86` | Your GPU's compute capability (`75` RTX 20 / GTX 16, `86` RTX 30, `89` RTX 40, `120` RTX 50). Or `native`. |
| `-DGGML_VULKAN=ON` | Vulkan GPU support for Whisper |
| `-DWITH_PARAKEET=OFF` | Build without the Parakeet engine (no sherpa-onnx download) |
| `-DSHERPA_ONNX_VERSION=x.y.z` | Use a different sherpa-onnx release (default `1.13.1`) |
| `-DSHERPA_ONNX_ROOT=<folder>` | Use an already-extracted sherpa-onnx *shared* package instead of downloading |

Find your GPU's compute capability with:

```powershell
nvidia-smi --query-gpu=name,compute_cap --format=csv
```

## How it works

**Whisper path:** audio is decoded to 16 kHz mono float samples (miniaudio, or `ffmpeg` as a fallback) and passed to `whisper_full`. Beam search, temperature fallback and the log-probability / no-speech gates are enabled for accuracy. Segment callbacks feed the live table.

**Parakeet path:** Parakeet works best on short chunks, so the audio is first split into speech chunks (up to 30 s) with the Silero voice-activity detector. Each chunk is padded slightly so word edges are not clipped, then transcribed by the Parakeet TDT transducer. Token timestamps are merged into words and regrouped into sentence-sized segments, splitting on sentence ends, long pauses, or the max-characters limit.

## Accuracy tips

- Use **large-v3** (or **large-v3-turbo** for speed). Avoid `tiny` and `base` if accuracy matters.
- Set the **language** explicitly if you know it instead of using auto-detect.
- Use the **prompt** box for names and jargon (Whisper only), e.g. `Dr. Nguyen, Kubernetes`.
- If Whisper gets stuck repeating a phrase on long audio, untick **Use previous text as context**.
- For subtitles, set **Max chars/segment** to about 42.
- Try both engines on a sample of your own audio. Which is better depends on the recording.

## Troubleshooting

| Problem | Fix |
|---|---|
| *"File ... is not digitally signed"* when running a `.ps1` | Run it as `powershell -ExecutionPolicy Bypass -File .\script.ps1` |
| `cmake` is not recognized | `winget install Kitware.CMake`, then open a **new** PowerShell window |
| *"could not find any instance of Visual Studio"* | Install the *Desktop development with C++* workload |
| **CUDA error: no kernel image is available for execution on the device** (Whisper loads, then the app closes) | The GPU code was built for a different GPU generation. Rebuild with `build-cuda.ps1 -Clean`, or force it: `-Arch 86` (use your card's number) |
| CUDA build fails with a `/Zc:preprocessor` or `C1189` error | CUDA 13 needs MSVC's standard preprocessor. The provided `CMakeLists.txt` already passes it; make sure you use the current file |
| *"nvcc fatal: A single input file is required"* | A stray global compiler flag reached nvcc. Use the current `CMakeLists.txt`, which applies `/utf-8` to the app target only |
| Older GPU (GTX 9xx / 10xx / Titan V) | CUDA 13 no longer supports it. Use `build-vulkan.ps1` or install CUDA 12.x |
| Model fails to load | Keep model paths free of accented or non-Latin characters |
| M4A / MP4 / OGG won't open | Install ffmpeg (`winget install ffmpeg`) and restart the app |
| Parakeet says the model folder is missing | Click **Download...** next to *Parakeet folder* (or run `download-parakeet.ps1`), or browse to the folder containing `encoder*.onnx`, `decoder*.onnx`, `joiner*.onnx` and `tokens.txt` |
| **Download...** fails | The app uses the system's `curl` and `tar` (included with Windows 10+; on Linux install `curl` and `bzip2`). Check your internet connection or proxy, then try again. Partial downloads resume. The `download-*.ps1` scripts are an alternative |
| Non-Latin text shows as `?` in the window | The built-in font covers Latin and Cyrillic only. Exported files are still correct |

## Project layout

```
main.cpp               GUI, Whisper engine, audio decoding, export
parakeet.h / .cpp      Parakeet engine (sherpa-onnx): VAD, decoding, timestamp grouping
downloader.h / .cpp    In-app model downloads (runs curl/tar in a background thread)
app.rc                 Windows resources: exe/taskbar icon and version info
assets/                icon.ico (+ icon.png) and make_icon.py to regenerate it
CMakeLists.txt         Fetches all dependencies automatically
build-cuda.ps1         Windows build, NVIDIA GPU (auto-detects architecture)
build-vulkan.ps1       Windows build, Vulkan GPU
common.ps1             Shared helpers for the build scripts
download-whisper.ps1   Downloads a Whisper ggml model into models\
download-parakeet.ps1  Downloads the Parakeet model and Silero VAD into models\
```

## Platform notes

Windows is the tested platform. The code and CMake are written to be portable and the Whisper engine should build on Linux and macOS, but those platforms have not been tested. Parakeet downloads a prebuilt sherpa-onnx package per platform; on unsupported platforms build with `-DWITH_PARAKEET=OFF` or point `SHERPA_ONNX_ROOT` at your own package. On Linux you also need the GTK3 and X11 development packages for the file dialogs and window.

## Credits and licenses

This project builds on excellent open-source work. Each component is under its own license:

- [whisper.cpp](https://github.com/ggml-org/whisper.cpp) and the OpenAI Whisper models
- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) and [Silero VAD](https://github.com/snakers4/silero-vad)
- NVIDIA [Parakeet](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) models (see the model card for license terms)
- [Dear ImGui](https://github.com/ocornut/imgui), [GLFW](https://www.glfw.org/), [Native File Dialog Extended](https://github.com/btzy/nativefiledialog-extended), [miniaudio](https://github.com/mackron/miniaudio)

License: CC0-1.0 license
