import json
import os
import torch
import torchaudio

from tqdm import tqdm

def process_split(input_json, output_json, split_name):
    with open(input_json, "r") as f:
        data = json.load(f)

    new_data = {}
    print(f"Processing {split_name}...")

    for uid, item in tqdm(data.items()):
        wav_path = item["wav"]
        spk = item["spk_id"]
        phn = item["phn"]

        waveform, sr = torchaudio.load(wav_path)
        assert sr == 16000, f"Expected 16kHz but got {sr}"

        # Compute FBANK
        fbank = torchaudio.compliance.kaldi.fbank(
            waveform,
            sample_frequency=16000,
            num_mel_bins=23
        )

       
        save_dir = f"fbanks/{spk}"
        os.makedirs(save_dir, exist_ok=True)
        save_path = f"{save_dir}/{uid}.pt"

        torch.save(fbank, save_path)

        # new JSON entry
        new_data[uid] = {
            "fbank": save_path,
            "spk_id": spk,
            "duration": item["duration"],
            "phn": phn,
        }

   
    with open(output_json, "w") as f:
        json.dump(new_data, f, indent=4)


if __name__ == "__main__":

    os.makedirs("fbanks", exist_ok=True)

    process_split(
        "json/train.json",
        "json/train_fbank.json",
        "train"
    )

    process_split(
        "json/dev.json",
        "json/dev_fbank.json",
        "dev"
    )

    process_split(
        "json/test.json",
        "json/test_fbank.json",
        "test"
    )