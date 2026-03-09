# BiLSTM-CTC Phoneme Recognition on TIMIT

End-to-end CTC-based phoneme recognition built on top of a baseline BiLSTM architecture, trained and evaluated on the TIMIT corpus. This is an exploration of regularization, optimization, architecture scaling, data augmentation, and decoding strategies to push Phone Error Rate (PER) from a 30.12% baseline down to **24.70%**, which is competitive with published BLSTM-CTC results on TIMIT.

---

## Results at a Glance

| Model | Params | Test PER |
|---|---|---|
| Baseline (1-layer BiLSTM, 128-dim) | 167K | 30.12% |
| + Dropout (p=0.1) + Gradient Clipping | 167K | 29.03% |
| + SGD with LR Scheduler | 167K | 28.20% |
| + 2-Layer BiLSTM | 562K | 26.81% |
| + Speed Perturbation Augmentation | 562K | **24.70%**|
| 2-Layer Wide (512-dim) + Augmentation | 8.5M | 24.08% |

> **Best efficiency/accuracy trade-off**: 2-layer 128-dim BiLSTM + speed perturbation achieves near-SOTA PER with only 562K parameters.

---

# Repository Structure

```
.
├── README.md
├── Report_CTC.pdf           # Full experimental report
│
├── data/
│   └── vocab_39.txt         # 39-phoneme vocabulary (+ silence)
│
├── preprocessing/
│   ├── create_fbanks.py     # Log Mel filterbank feature extraction (23-dim)
│   └── make_vocab.py        # Phoneme vocabulary construction
│
├── training/
│   ├── dataloader.py        # TIMIT dataset loader + speed perturbation augmentation
│   ├── data_aug.py          # Speed perturbation (0.9x, 1.0x, 1.1x)
│   ├── models.py            # BiLSTM and UniLSTM model definitions
│   ├── trainer.py           # Training loop, LR scheduler, gradient clipping
│   └── utils.py             # Metrics, logging, helper functions
│
├── decoding/
│   ├── decoder.py           # CTC greedy + blank penalty decoding
│   └── phone_map           # Phone mapping for 39-phoneme set
│
└── run.py                   # Main entry point
```

---

## Experiments

### Features
TIMIT utterances are converted to 23-dimensional log Mel filterbank (FBank) features. The 61-phoneme set is reduced to a standard 39-phoneme set plus a silence token.

### Regularization
- **Dropout** (p=0.1 on BiLSTM outputs) improved generalization — higher values hurt CTC alignment stability by zeroing features mid-sequence.
- **Gradient clipping** (max-norm=1) reduced loss spikes and improved test PER by ~0.6% over unclipped training.

### Optimisation
- **Adam** is highly sensitive to learning rate in CTC training; performs well only at lr=0.001.
- **SGD + LR Scheduler** (halve LR on validation loss plateau) gives the best generalisation. The scheduler allows large early updates then fine-grained alignment sharpening in later epochs.

### Architecture
- **Depth**: A 2-layer BiLSTM outperforms 1-layer by 1.4% PER — the first layer captures acoustic patterns, the second models phonetic transitions.
- **Width**: 512-dim layers improve PER further (24.08%) but at 15× the parameter count of the narrow model.
- **UniLSTM** vs **BiLSTM**: A 6–7% PER gap confirms that future phonetic context is critical for English phoneme disambiguation.

### Data Augmentation
Speed perturbation at 0.9× and 1.1× triples training data and reduces PER by 2.11% on the 128-dim model. The smaller architecture benefits more — width was its bottleneck in capturing speaker variation.

### Decoding
- **Blank penalty**: Subtracting a small penalty (0.5) from blank posteriors reduces deletion errors and improves PER to 24.30% on the best model.
- **Confusion analysis**: The model most often confuses acoustically similar vowels ("ah"/"ih", "eh"/"ih") and voiced/unvoiced fricative pairs ("s"/"z").

---

# Setup & Usage

```bash
# Install dependencies
pip install torch torchaudio numpy matplotlib

# Extract FBank features
python preprocessing/create_fbanks.py --data_dir /path/to/TIMIT

# Train
python run.py \
  --num_layers 2 \
  --model_dims 128 \
  --dropout 0.1 \
  --lr 0.0001 \
  --lr_factor 0.5 \
  --num_epochs 20

# Evaluate with blank penalty
python run.py --eval --blank_penalty 0.5
```

---

## Report

Full experimental details, ablation tables, and figures are in [`Report_CTC.pdf`](./Report_CTC.pdf).

---

## Acknowledgements

The baseline BiLSTM-CTC skeleton was provided by the University of Cambridge
Department of Engineering as part of a graduate-level speech recognition course.
Original framework conceived and developed by Tony Zheng, Eric Li, Brian Sun,
Xiaoyu Yang, and Phil Woodland. All experimental extensions, architectural
modifications, and analysis are my own work.

