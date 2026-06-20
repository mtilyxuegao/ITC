# Injected via PYTHONPATH=/app: bypass torchaudio->torchcodec (torch 2.9) for WAV loading.
# torchaudio 2.9 routes load() to torchcodec which needs a matching FFmpeg/torchcodec build.
# We read WAV with soundfile (already a dependency) instead.
try:
    import torch
    import torchaudio
    import soundfile as sf

    def _sf_load(filepath, *args, **kwargs):
        data, sr = sf.read(str(filepath), dtype="float32", always_2d=True)  # (T, C)
        wav = torch.from_numpy(data.T).contiguous()  # (C, T)
        return wav, sr

    torchaudio.load = _sf_load
except Exception as _e:  # never block startup
    import sys
    print(f"[sitecustomize] torchaudio.load patch skipped: {_e}", file=sys.stderr)
