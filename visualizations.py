import json
import math
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

import models
from dataloader import get_dataloader
from utils import concat_inputs

# ARGS
parser = argparse.ArgumentParser(description="CTC visualisation: spike plot, entropy, trellis, Viterbi-FBank overlay")
parser.add_argument('--model_path',  type=str, default="checkpoints/model_20",  help="path to saved model checkpoint")
parser.add_argument('--json_path',   type=str, default="data/json/dev_fbank.json", help="path to dev/test JSON")
parser.add_argument('--vocab',       type=str, default="data/vocab_39.txt",           help="vocabulary file path")
parser.add_argument('--num_layers',  type=int, default=2,   help="number of BiLSTM layers")
parser.add_argument('--fbank_dims',  type=int, default=23,  help="filterbank feature dimension")
parser.add_argument('--model_dims',  type=int, default=128, help="model hidden size (use 512 for wide model)")
parser.add_argument('--concat',      type=int, default=1,   help="frame concatenation factor")
parser.add_argument('--topk',        type=int, default=8,   help="number of non-blank phones to show in spike plot")
parser.add_argument('--out_dir',     type=str, default=".",  help="output directory for saved plots")
args = parser.parse_args()

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
EPS = 1e-12

# Helpers
def load_vocab(path):
    vocab = {}
    with open(path) as f:
        for i, line in enumerate(f):
            vocab[line.strip()] = i
    return vocab

def logsumexp(a, b):
    if a == -np.inf: return b
    if b == -np.inf: return a
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))

def logsumexp3(a, b, c):
    return logsumexp(logsumexp(a, b), c)

def make_ctc_extended_targets(target_ids, blank_id):
    ext = [blank_id]
    for t in target_ids:
        ext.append(t)
        ext.append(blank_id)
    return ext

def ctc_forward_trellis(log_probs, ext, blank_id):
    T, V = log_probs.shape
    S = len(ext)
    alpha = np.full((T, S), -np.inf, dtype=np.float64)
    alpha[0, 0] = log_probs[0, blank_id]
    if S > 1:
        alpha[0, 1] = log_probs[0, ext[1]]
    for t in range(1, T):
        alpha[t, 0] = alpha[t-1, 0] + log_probs[t, ext[0]]
        for s in range(1, S):
            stay = alpha[t-1, s]
            prev = alpha[t-1, s-1]
            skip = -np.inf
            if s >= 2 and ext[s] != blank_id and ext[s] != ext[s-2]:
                skip = alpha[t-1, s-2]
            alpha[t, s] = log_probs[t, ext[s]] + logsumexp3(stay, prev, skip)
    return alpha

def ctc_viterbi_path(log_probs, ext, blank_id):
    T, V = log_probs.shape
    S = len(ext)
    dp = np.full((T, S), -np.inf, dtype=np.float64)
    bp = np.full((T, S), -1, dtype=np.int32)

    dp[0, 0] = log_probs[0, blank_id]
    if S > 1:
        dp[0, 1] = log_probs[0, ext[1]]
        bp[0, 1] = 1

    for t in range(1, T):
        dp[t, 0] = dp[t-1, 0] + log_probs[t, ext[0]]
        bp[t, 0] = 0
        for s in range(1, S):
            stay = dp[t-1, s]
            prev = dp[t-1, s-1]
            skip = -np.inf
            can_skip = (s >= 2 and ext[s] != blank_id and ext[s] != ext[s-2])
            if can_skip:
                skip = dp[t-1, s-2]
            best_val, best_move = stay, 0
            if prev > best_val: best_val, best_move = prev, 1
            if can_skip and skip > best_val: best_val, best_move = skip, 2
            dp[t, s] = log_probs[t, ext[s]] + best_val
            bp[t, s] = best_move

    end_candidates = [(dp[T-1, S-1], S-1)]
    if S >= 2:
        end_candidates.append((dp[T-1, S-2], S-2))
    _, best_s = max(end_candidates, key=lambda x: x[0])

    path = []
    t, s = T - 1, best_s
    while t >= 0 and s >= 0:
        path.append((t, s))
        move = bp[t, s]
        if t == 0: break
        if move == 0: t -= 1
        elif move == 1: t -= 1; s -= 1
        else: t -= 1; s -= 2
    path.reverse()
    return path

def collapse_ctc_labels(ext_labels, blank_id, idx2phn):
    phones, prev = [], None
    for lab in ext_labels:
        if lab == blank_id:
            prev = lab
            continue
        if prev is None or lab != prev:
            phones.append(idx2phn[lab])
        prev = lab
    return phones

# Main
def main():
    vocab = load_vocab(args.vocab)
    idx2phn = {v: k for k, v in vocab.items()}
    blank_id = vocab["_"]

    # Load model
    model = models.BiLSTM(args.num_layers, args.fbank_dims * args.concat, args.model_dims, len(vocab))
    state = torch.load(args.model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    print(f"Loaded model: {args.model_path}")

    # Load first utterance from JSON
    with open(args.json_path) as f:
        j = json.load(f)
    utt_id = next(iter(j.keys()))

    loader = get_dataloader(args.json_path, 1, False)
    inputs, in_lens, trans, _ = next(iter(loader))
    ref_tokens = trans[0].split()

    true_len = int(in_lens[0])
    inputs = inputs[:true_len]
    in_lens = torch.tensor([true_len], device=in_lens.device)
    inputs = inputs.to(DEVICE)
    in_lens = in_lens.to(DEVICE)
    inputs, in_lens = concat_inputs(inputs, in_lens, factor=args.concat)

    print(f"Utterance : {utt_id}")
    print(f"Reference : {' '.join(ref_tokens[:20])}")

    with torch.no_grad():
        logits = model(inputs)
        log_probs_t = torch.nn.functional.log_softmax(logits, dim=-1)

    log_probs = log_probs_t[:, 0, :].detach().cpu().numpy()
    probs = np.exp(log_probs)
    T, V = probs.shape
    x = np.arange(T)

    import os
    os.makedirs(args.out_dir, exist_ok=True)


    # (1) Spike plot
    peak = probs.max(axis=0)
    peak[blank_id] = -1.0
    top_ids = np.argsort(peak)[::-1][:args.topk]
    top_labels = [idx2phn[i] for i in top_ids]

    plt.figure(figsize=(12, 4))
    for i, lab in zip(top_ids, top_labels):
        plt.plot(x, probs[:, i], label=lab, linewidth=1.0)
    plt.yscale("log")
    plt.ylim(1e-8, 1.0)
    plt.xlabel("Frame index")
    plt.ylabel("Posterior prob (log scale)")
    plt.title(f"CTC Spike Plot (top-{args.topk} phones) — {utt_id}")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=6, fontsize=8)
    plt.tight_layout()
    out = os.path.join(args.out_dir, "viz_spike.png")
    plt.savefig(out, dpi=200); plt.close()
    print(f"Saved: {out}")


    # (2) Entropy plot
    ent = -(probs * np.log(probs + EPS)).sum(axis=1)
    ent_norm = ent / np.log(V)

    plt.figure(figsize=(12, 3.2))
    plt.plot(x, ent_norm, linewidth=1.25)
    plt.xlabel("Frame index")
    plt.ylabel("Entropy (normalized)")
    plt.title(f"CTC Output Entropy Over Time — {utt_id}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(args.out_dir, "viz_entropy.png")
    plt.savefig(out, dpi=200); plt.close()
    print(f"Saved: {out}")


    # (3) Alignment trellis
    target_ids = [vocab[p] for p in ref_tokens]
    ext = make_ctc_extended_targets(target_ids, blank_id)
    alpha = ctc_forward_trellis(log_probs, ext, blank_id)
    alpha_vis = np.exp(alpha - alpha.max(axis=1, keepdims=True))

    plt.figure(figsize=(12, 6))
    plt.imshow(alpha_vis.T, aspect="auto", origin="lower", cmap="magma")
    plt.xlabel("Frame index (t)")
    plt.ylabel("CTC state (2N+1)")
    plt.title(f"CTC Alignment Trellis (forward mass) — {utt_id}")
    plt.colorbar(label="Relative probability mass")
    plt.tight_layout()
    out = os.path.join(args.out_dir, "viz_trellis.png")
    plt.savefig(out, dpi=200); plt.close()
    print(f"Saved: {out}")

    # (4) Viterbi path on FBank spectrogram
    v_path = ctc_viterbi_path(log_probs, ext, blank_id)
    decoded_phones = collapse_ctc_labels([ext[s] for (t, s) in v_path], blank_id, idx2phn)

    boundaries, prev_lab = [], None
    for (t, s) in v_path:
        lab = ext[s]
        if lab != prev_lab:
            if lab != blank_id:
                boundaries.append((t, idx2phn[lab]))
        prev_lab = lab

    fbank = inputs[:, 0, :].detach().cpu().numpy()

    plt.figure(figsize=(12, 5))
    plt.imshow(fbank.T, aspect="auto", origin="lower", cmap="magma")
    plt.xlabel("Frame index")
    plt.ylabel("FBank bin")
    plt.title(f"FBank spectrogram with Viterbi CTC alignment boundaries — {utt_id}")

    y_top = fbank.shape[1] - 1
    for i, (t, ph) in enumerate(boundaries):
        plt.axvline(t, color="white", alpha=0.25, linewidth=1.0)
        y = y_top if (i % 2 == 0) else y_top - 2
        plt.text(t, y, ph, rotation=90, color="white", fontsize=11, fontweight="bold",
                 va="top", ha="center",
                 bbox=dict(facecolor="black", alpha=0.35, edgecolor="none", pad=1.2),
                 clip_on=True)

    plt.tight_layout()
    out = os.path.join(args.out_dir, "viz_viterbi_fbank.png")
    plt.savefig(out, dpi=200); plt.close()
    print(f"Saved: {out}")

    print(f"\nDecoded phones (first 30): {' '.join(decoded_phones[:30])}")

if __name__ == "__main__":
    main()