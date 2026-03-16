"""
baseline.py
Baseline HuggingFace beam search benchmark across:
  - WMT14 En-De       → facebook/wmt19-en-de      → BLEU
  - WMT17 En-De       → facebook/wmt19-en-de      → BLEU
  - CNN/DailyMail     → facebook/bart-large-cnn   → ROUGE
  - WikiText-103      → gpt2-medium               → Throughput only
"""

import time, csv, gc, os
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    AutoModelForCausalLM, pipeline
)
from datasets import load_dataset
from sacrebleu.metrics import BLEU
from rouge_score import rouge_scorer

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
BEAM_WIDTHS  = [2, 4, 8, 16]
N_SAMPLES    = 100       # number of examples per dataset per run
MAX_NEW_TOKENS = 128
WARMUP_STEPS = 5        # warm up GPU before timing
RESULTS_CSV  = "baseline_results.csv"

@dataclass
class BenchResult:
    task:           str
    dataset:        str
    beam_width:     int
    mean_latency_ms: float          # ms per output token
    p95_latency_ms: float
    throughput_tps: float           # output tokens / sec
    peak_gpu_mb:    float
    quality_metric: str             # e.g. "BLEU" or "ROUGE-1"
    quality_value:  float

results: List[BenchResult] = []


def reset_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

def peak_gpu_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024**2
    return 0.0

def batch(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]


def timed_generate(model, tokenizer, texts, num_beams,
                   max_new_tokens=MAX_NEW_TOKENS,
                   batch_size=8, is_causal=False):
    """
    Returns (all_decoded_texts, per_sample_latency_ms, tokens_per_sec)
    Latency is measured end-to-end per sample (ms per output token).
    """
    all_decoded = []
    per_token_latencies = []   # ms/token per sample
    total_output_tokens = 0
    total_time_s = 0.0

    # Warm-up pass (not recorded)
    warmup_texts = texts[:min(WARMUP_STEPS, len(texts))]
    enc = tokenizer(warmup_texts, return_tensors="pt",
                    padding=True, truncation=True,
                    max_length=512).to(DEVICE)
    with torch.no_grad():
        model.generate(**enc, num_beams=num_beams,
                       max_new_tokens=max_new_tokens,
                       early_stopping=True)
    reset_gpu_memory()

    for btexts in batch(texts, batch_size):
        enc = tokenizer(btexts, return_tensors="pt",
                        padding=True, truncation=True,
                        max_length=512).to(DEVICE)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            out = model.generate(
                **enc,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
                early_stopping=True,
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        elapsed_s = t1 - t0
        total_time_s += elapsed_s

        # Count only newly generated tokens (excluding prompt for causal)
        if is_causal:
            input_len = enc["input_ids"].shape[1]
            gen_tokens = out[:, input_len:]
        else:
            gen_tokens = out

        n_out = gen_tokens.numel()          # total tokens in batch
        total_output_tokens += n_out
        n_samples_in_batch = len(btexts)

        # per-sample average token latency (ms)
        tokens_per_sample = n_out / n_samples_in_batch
        ms_per_token = (elapsed_s * 1000) / max(tokens_per_sample, 1)
        per_token_latencies.extend([ms_per_token] * n_samples_in_batch)

        decoded = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
        all_decoded.extend(decoded)

    throughput = total_output_tokens / total_time_s  # tokens/sec
    return all_decoded, per_token_latencies, throughput

## ── 1. WMT Translation (En→De) ──────────────
def run_translation(dataset_name: str, wmt_year: str):
    print(f"\n{'='*55}")
    print(f"  Task: Translation | Dataset: {dataset_name}")
    print(f"{'='*55}")

    MODEL_ID = "facebook/wmt19-en-de"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16 if DEVICE=="cuda" else torch.float32
    ).to(DEVICE).eval()

    ds = load_dataset(f"wmt/{wmt_year}", "de-en",
                      split="test", trust_remote_code=True)
    sources  = [ex["translation"]["en"] for ex in ds.select(range(N_SAMPLES))]
    references = [ex["translation"]["de"] for ex in ds.select(range(N_SAMPLES))]

    bleu = BLEU()

    for k in BEAM_WIDTHS:
        print(f"  Beam width K={k} ...", end=" ", flush=True)
        reset_gpu_memory()

        decoded, latencies, tps = timed_generate(
            model, tokenizer, sources,
            num_beams=k, batch_size=8
        )
        p_gpu = peak_gpu_mb()

        score = bleu.corpus_score(decoded, [references]).score
        mean_lat = float(np.mean(latencies))
        p95_lat  = float(np.percentile(latencies, 95))
        print(f"BLEU={score:.2f}  {tps:.1f} tok/s  "
              f"lat={mean_lat:.1f}ms  GPU={p_gpu:.0f}MB")

        results.append(BenchResult(
            task="translation", dataset=dataset_name,
            beam_width=k, mean_latency_ms=mean_lat, p95_latency_ms=p95_lat,
            throughput_tps=tps, peak_gpu_mb=p_gpu,
            quality_metric="BLEU", quality_value=score
        ))

    del model; reset_gpu_memory()

## ── 2. CNN/DailyMail Summarization ──────────
def run_summarization():
    print(f"\n{'='*55}")
    print(f"  Task: Summarization | Dataset: CNN/DailyMail")
    print(f"{'='*55}")

    MODEL_ID = "facebook/bart-large-cnn"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16 if DEVICE=="cuda" else torch.float32
    ).to(DEVICE).eval()

    ds = load_dataset("cnn_dailymail", "3.0.0", split="test")
    articles  = [ex["article"]   for ex in ds.select(range(N_SAMPLES))]
    highlights = [ex["highlights"] for ex in ds.select(range(N_SAMPLES))]

    scorer = rouge_scorer.RougeScorer(["rouge1","rouge2","rougeL"],
                                       use_stemmer=True)

    for k in BEAM_WIDTHS:
        print(f"  Beam width K={k} ...", end=" ", flush=True)
        reset_gpu_memory()

        # BART uses slightly different generate kwargs
        decoded, latencies, tps = timed_generate(
            model, tokenizer, articles,
            num_beams=k, max_new_tokens=142, batch_size=4
        )
        p_gpu = peak_gpu_mb()

        # Aggregate ROUGE-1 F1
        r1_scores = [
            scorer.score(ref, hyp)["rouge1"].fmeasure
            for ref, hyp in zip(highlights, decoded)
        ]
        r1 = float(np.mean(r1_scores)) * 100
        mean_lat = float(np.mean(latencies))
        p95_lat  = float(np.percentile(latencies, 95))
        print(f"ROUGE-1={r1:.2f}  {tps:.1f} tok/s  "
              f"lat={mean_lat:.1f}ms  GPU={p_gpu:.0f}MB")

        results.append(BenchResult(
            task="summarization", dataset="cnn_dailymail",
            beam_width=k, mean_latency_ms=mean_lat, p95_latency_ms=p95_lat,
            throughput_tps=tps, peak_gpu_mb=p_gpu,
            quality_metric="ROUGE-1", quality_value=r1
        ))

    del model; reset_gpu_memory()


## ── 3. WikiText-103 Generation Throughput ───
def run_lm_generation():
    print(f"\n{'='*55}")
    print(f"  Task: LM Generation | Dataset: WikiText-103")
    print(f"{'='*55}")

    MODEL_ID = "gpt2-medium"   # or "gpt2-xl" if memory allows
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16 if DEVICE=="cuda" else torch.float32
    ).to(DEVICE).eval()

    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    # Use non-empty lines as prompts; truncate to ~50 tokens each
    raw_texts = [t.strip() for t in ds["text"] if len(t.strip()) > 50]
    prompts   = raw_texts[:N_SAMPLES]

    for k in BEAM_WIDTHS:
        print(f"  Beam width K={k} ...", end=" ", flush=True)
        reset_gpu_memory()

        decoded, latencies, tps = timed_generate(
            model, tokenizer, prompts,
            num_beams=k, max_new_tokens=MAX_NEW_TOKENS,
            batch_size=4, is_causal=True
        )
        p_gpu = peak_gpu_mb()
        mean_lat = float(np.mean(latencies))
        p95_lat  = float(np.percentile(latencies, 95))
        print(f"{tps:.1f} tok/s  lat={mean_lat:.1f}ms  GPU={p_gpu:.0f}MB")

        results.append(BenchResult(
            task="lm_generation", dataset="wikitext-103",
            beam_width=k, mean_latency_ms=mean_lat, p95_latency_ms=p95_lat,
            throughput_tps=tps, peak_gpu_mb=p_gpu,
            quality_metric="N/A", quality_value=0.0
        ))

    del model; reset_gpu_memory()

if __name__ == "__main__":
    run_translation("wmt14",  "wmt14")
    run_translation("wmt17",  "wmt17")
    run_summarization()
    run_lm_generation()

    # Save CSV
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "task","dataset","beam_width",
            "mean_latency_ms","p95_latency_ms","throughput_tps",
            "peak_gpu_mb","quality_metric","quality_value"
        ])
        for r in results:
            writer.writerow([
                r.task, r.dataset, r.beam_width,
                f"{r.mean_latency_ms:.3f}", f"{r.p95_latency_ms:.3f}",
                f"{r.throughput_tps:.2f}", f"{r.peak_gpu_mb:.1f}",
                r.quality_metric, f"{r.quality_value:.4f}"
            ])
    print(f"\nResults saved to {RESULTS_CSV}")
