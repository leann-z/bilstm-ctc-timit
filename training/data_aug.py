import json
from pathlib import Path

import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi

TRAIN_WAV_JSON   = "data/json/train.json"
TRAIN_FBANK_JSON = "data/json/train_fbank.json"
OUT_JSON         = "data/json/train_fbank_sp.json"
OUT_ROOT         = Path("fbanks_sp")

RATES = [0.9, 1.1]
TARGET_SR = 16000

FBANK_KWARGS = dict(
    num_mel_bins=23,         
    sample_frequency=TARGET_SR,
    use_energy=False
)

def speed_perturb(wav: torch.Tensor, sr: int, rate: float) -> torch.Tensor:
    wav_sp = torchaudio.functional.resample(wav, sr, int(sr * rate))
    wav_sp = torchaudio.functional.resample(wav_sp, int(sr * rate), sr)
    return wav_sp

def compute_fbank(wav: torch.Tensor) -> torch.Tensor:
    if wav.dim() == 2:
        wav1 = wav[:1]              
    else:
        wav1 = wav.unsqueeze(0)
    return kaldi.fbank(wav1, **FBANK_KWARGS)

def main():
    with open(TRAIN_WAV_JSON, "r") as f:
        train_wav = json.load(f)

    with open(TRAIN_FBANK_JSON, "r") as f:
        train_fbank = json.load(f)

    out = dict(train_fbank)

    missing_in_baseline = 0
    created = 0

    for utt_id, meta in train_wav.items():
        if utt_id not in train_fbank:
            missing_in_baseline += 1
            continue

        wav_path = meta["wav"]
        spk_id = meta["spk_id"]
        phn = meta["phn"]
        duration = float(meta["duration"])

        wav, sr = torchaudio.load(wav_path)
        if sr != TARGET_SR:
            wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
            sr = TARGET_SR

        (OUT_ROOT / spk_id).mkdir(parents=True, exist_ok=True)

        for rate in RATES:
            new_id = f"{utt_id}_sp{rate}" 
            out_path = OUT_ROOT / spk_id / f"{new_id}.pt"

            if out_path.exists():
                out[new_id] = {
                    "fbank": str(out_path),
                    "duration": duration / rate,
                    "spk_id": spk_id,
                    "phn": phn
                }
                continue

            wav_sp = speed_perturb(wav, sr, rate)
            feats = compute_fbank(wav_sp)
            torch.save(feats, out_path)

            out[new_id] = {
                "fbank": str(out_path),
                "duration": duration / rate,
                "spk_id": spk_id,
                "phn": phn
            }
            created += 1

    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Wrote {OUT_JSON}")
    print(f"Created {created} augmented fbank files under {OUT_ROOT}/")

if __name__ == "__main__":
    main()

