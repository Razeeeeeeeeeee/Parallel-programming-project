"""
variant_a_fusion.py
Variant A: Fused log-softmax + beam score update in one Triton kernel.
Runs all 4 benchmarks from baseline.py using a custom beam search loop
that calls fused_score() at every decode step.
"""

import time, csv, gc
import torch
import triton
import triton.language as tl
import numpy as np
from typing import List
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from transformers.modeling_outputs import BaseModelOutput
from datasets import load_dataset
from sacrebleu.metrics import BLEU
from rouge_score import rouge_scorer

DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
BEAM_WIDTHS    = [2, 4, 8, 16]
N_SAMPLES      = 100
MAX_NEW_TOKENS = 128
WARMUP_STEPS   = 3
RESULTS_CSV    = "variant_a_results.csv"

@dataclass
class BenchResult:
    task:            str
    dataset:         str
    beam_width:      int
    mean_latency_ms: float
    p95_latency_ms:  float
    throughput_tps:  float
    peak_gpu_mb:     float
    quality_metric:  str
    quality_value:   float

results: List[BenchResult] = []


# ─────────────────────────────────────────────────────────────────────────────
# Triton fused kernel: log_softmax + beam score addition in one pass
# V is a plain int (NOT constexpr) → one compiled variant handles all vocab sizes
# BLOCK_V is constexpr → required by tl.arange
# ─────────────────────────────────────────────────────────────────────────────
@triton.jit
def fused_lsm_score_kernel(
    logits_ptr,          # [K, V] float32 input
    scores_ptr,          # [K]    float32 beam cumulative log-probs
    output_ptr,          # [K, V] float32 output
    V,                   # plain int: actual vocab size (NOT tl.constexpr)
    BLOCK_V: tl.constexpr,  # tile width — must be constexpr for tl.arange
):
    beam_id = tl.program_id(0)     # one CTA per beam row
    base    = beam_id * V
    score_k = tl.load(scores_ptr + beam_id)

    # Pass 1: row max for numerical stability
    row_max = tl.full([1], -float("inf"), dtype=tl.float32)
    for s in range(0, V, BLOCK_V):
        offs = s + tl.arange(0, BLOCK_V)
        mask = offs < V
        x    = tl.load(logits_ptr + base + offs, mask=mask, other=-float("inf"))
        row_max = tl.maximum(row_max, tl.max(x, axis=0, keep_dims=True))

    # Pass 2: sum(exp(x - max)) for log-sum-exp denominator
    sum_exp = tl.zeros([1], dtype=tl.float32)
    for s in range(0, V, BLOCK_V):
        offs = s + tl.arange(0, BLOCK_V)
        mask = offs < V
        x    = tl.load(logits_ptr + base + offs, mask=mask, other=-float("inf"))
        sum_exp += tl.sum(tl.where(mask, tl.exp(x - row_max), 0.0),
                          axis=0, keep_dims=True)
    log_denom = tl.log(sum_exp) + row_max

    # Pass 3: write log_softmax(logits[k]) + beam_score[k]
    for s in range(0, V, BLOCK_V):
        offs = s + tl.arange(0, BLOCK_V)
        mask = offs < V
        x    = tl.load(logits_ptr + base + offs, mask=mask, other=-float("inf"))
        tl.store(output_ptr + base + offs,
                 tl.where(mask, x - log_denom + score_k, -float("inf")),
                 mask=mask)


def fused_score(logits: torch.Tensor, beam_scores: torch.Tensor) -> torch.Tensor:
    """
    Replaces: F.log_softmax(logits, dim=-1) + beam_scores[:, None]
    Input:  logits [K, V] float32,  beam_scores [K] float32
    Output: [K, V] float32
    One Triton kernel reads logits once and writes result once — no intermediate tensor.
    """
    K, V    = logits.shape
    out     = torch.empty_like(logits)
    BLOCK_V = triton.next_power_of_2(min(V, 4096))
    fused_lsm_score_kernel[(K,)](
        logits, beam_scores, out,
        V, BLOCK_V,        # V=plain int, BLOCK_V=constexpr
        num_warps=8,
    )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def reset_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

def peak_mb() -> float:
    return torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0

def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def reorder_cache(model, past_kv, idx: torch.Tensor):
    if past_kv is None:
        return None

    # New-style Cache object (transformers >= 4.38)
    if hasattr(past_kv, 'reorder_cache'):
        past_kv.reorder_cache(idx)
        return past_kv

    # Standard models that implement _reorder_cache
    if hasattr(model, '_reorder_cache'):
        return model._reorder_cache(past_kv, idx)

    # FSMT (facebook/wmt19-en-de) and other legacy models:
    # past_kv is tuple of 4-tuples (self_k, self_v, cross_k, cross_v)
    return tuple(
        tuple(
            t.index_select(0, idx.to(t.device))
            if isinstance(t, torch.Tensor) else t
            for t in layer
        )
        for layer in past_kv
    )


def batch_iter(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]


# ─────────────────────────────────────────────────────────────────────────────
# Seq2Seq beam search (wmt19-en-de, bart-large-cnn)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def fused_beam_seq2seq(model, tokenizer,
                       input_ids: torch.Tensor,
                       attention_mask: torch.Tensor,
                       num_beams: int,
                       max_new_tokens: int,
                       ) -> List[str]:
    device   = input_ids.device
    B, K     = input_ids.shape[0], num_beams
    V        = model.config.vocab_size
    eos_id   = model.config.eos_token_id
    start_id = getattr(model.config, 'decoder_start_token_id', eos_id)

    # ── Encode once, expand K times ───────────────────────────
    enc_out = model.get_encoder()(
        input_ids=input_ids, attention_mask=attention_mask, return_dict=True
    )
    # [B, S, D] → [B*K, S, D]
    enc_hidden = (enc_out.last_hidden_state
                  .unsqueeze(1).expand(B, K, -1, -1)
                  .contiguous().view(B*K, -1, enc_out.last_hidden_state.shape[-1]))
    enc_attn   = (attention_mask
                  .unsqueeze(1).expand(B, K, -1)
                  .contiguous().view(B*K, -1))

    # ── Initialise beams ──────────────────────────────────────
    sequences   = torch.full((B*K, 1), start_id, dtype=torch.long, device=device)
    beam_scores = torch.full((B*K,), float('-inf'), device=device)
    for b in range(B):
        beam_scores[b*K] = 0.0          # only first beam of each group is live

    best_score  = [float('-inf')] * B
    best_seq    = [None] * B
    past_kv     = None
    batch_off   = torch.arange(B, device=device).unsqueeze(1) * K  # [B, 1]

    for _  in range(max_new_tokens):
        # ── Decoder step (uses KV cache after first step) ─────
        out = model(
            decoder_input_ids=sequences[:, -1:],
            encoder_outputs=BaseModelOutput(last_hidden_state=enc_hidden),
            attention_mask=enc_attn,
            past_key_values=past_kv,
            use_cache=True, return_dict=True,
        )
        logits  = out.logits[:, -1, :].float()   # [B*K, V]
        
        
        past_kv = out.past_key_values


        # ── Fused scoring: one kernel call per beam row ────────
        # Process all B batch items; kernel launch is [K,] CTAs per item
        scored = torch.empty_like(logits)
        for b in range(B):
            sl = slice(b*K, (b+1)*K)
            scored[sl] = fused_score(logits[sl], beam_scores[sl])

        

        # ── Top-2K selection over [K*V] per batch item ────────
        # [B*K, V] → [B, K*V] → topk along dim=1
        top_scores, top_flat = torch.topk(scored.view(B, K*V), 2*K, dim=1)
        src_beams  = top_flat // V   # [B, 2K]  local beam index
        src_tokens = top_flat % V    # [B, 2K]  token index

        # Take top K; convert local beam → global [B*K] index
        global_par  = (src_beams[:, :K] + batch_off).contiguous().view(-1)   # [B*K]

        # ── Update state ──────────────────────────────────────
        past_kv     = reorder_cache(model, past_kv, global_par)
        sequences   = torch.cat([sequences[global_par],
                                  src_tokens[:, :K].contiguous().view(-1, 1)], dim=1)
        beam_scores = top_scores[:, :K].contiguous().view(-1)

        # ── EOS bookkeeping ───────────────────────────────────
        last_tok = src_tokens[:, :K]    # [B, K]
        for b in range(B):
            for ki in range(K):
                if last_tok[b, ki].item() == eos_id:
                    sc = beam_scores[b*K + ki].item()
                    if sc > best_score[b]:
                        best_score[b] = sc
                        best_seq[b]   = sequences[b*K + ki, 1:].tolist()  # drop start_id

    # ── Return best sequence per batch item ───────────────────
    final_ids = []
    for b in range(B):
        if best_seq[b]:
            final_ids.append(best_seq[b])
        else:
            ki = beam_scores[b*K:(b+1)*K].argmax().item()
            final_ids.append(sequences[b*K + ki, 1:].tolist())
    return tokenizer.batch_decode(final_ids, skip_special_tokens=True)


# ─────────────────────────────────────────────────────────────────────────────
# Causal LM beam search (gpt2-medium)
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def fused_beam_causal(model, tokenizer,
                      input_ids: torch.Tensor,
                      attention_mask: torch.Tensor,
                      num_beams: int,
                      max_new_tokens: int) -> List[str]:
    device    = input_ids.device
    B, plen   = input_ids.shape
    K         = num_beams
    V         = model.config.vocab_size
    eos_id    = tokenizer.eos_token_id
    batch_off = torch.arange(B, device=device).unsqueeze(1) * K  # [B, 1]

    # ── Expand prompt [B, S] → [B*K, S] ──────────────────────
    inp_exp  = input_ids.unsqueeze(1).expand(B, K, -1).contiguous().view(B*K, -1)
    attn_exp = attention_mask.unsqueeze(1).expand(B, K, -1).contiguous().view(B*K, -1)

    # ── Prefill: full prompt in one forward ───────────────────
    out     = model(input_ids=inp_exp, attention_mask=attn_exp,
                    use_cache=True, return_dict=True)
    logits  = out.logits[:, -1, :].float()   # [B*K, V]
    past_kv = out.past_key_values

    # ── Init beam scores (only first beam per group is live) ──
    beam_scores = torch.full((B*K,), float('-inf'), device=device)
    for b in range(B):
        beam_scores[b*K] = 0.0

    # ── First token step using prefill logits ─────────────────
    scored = torch.empty_like(logits)
    for b in range(B):
        sl = slice(b*K, (b+1)*K)
        scored[sl] = fused_score(logits[sl], beam_scores[sl])

    top_s, top_f = torch.topk(scored.view(B, K*V), 2*K, dim=1)
    global_par   = (top_f[:, :K] // V + batch_off).contiguous().view(-1)
    past_kv      = reorder_cache(model, past_kv, global_par)
    beam_scores  = top_s[:, :K].contiguous().view(-1)
    sequences    = (top_f[:, :K] % V).contiguous().view(-1, 1)   # [B*K, 1]

    best_score = [float('-inf')] * B
    best_seq   = [None] * B

    # ── Auto-regressive decode steps ──────────────────────────
    for step in range(1, max_new_tokens):
        cur_attn = torch.ones(B*K, plen + step, dtype=torch.long, device=device)
        out = model(input_ids=sequences[:, -1:], attention_mask=cur_attn,
                    past_key_values=past_kv, use_cache=True, return_dict=True)
        logits  = out.logits[:, -1, :].float()
        past_kv = out.past_key_values

        scored = torch.empty_like(logits)
        for b in range(B):
            sl = slice(b*K, (b+1)*K)
            scored[sl] = fused_score(logits[sl], beam_scores[sl])

        top_s, top_f = torch.topk(scored.view(B, K*V), 2*K, dim=1)
        src_tokens   = top_f[:, :K] % V
        global_par   = (top_f[:, :K] // V + batch_off).contiguous().view(-1)
        past_kv      = reorder_cache(model, past_kv, global_par)
        sequences    = torch.cat([sequences[global_par],
                                   src_tokens.contiguous().view(-1, 1)], dim=1)
        beam_scores  = top_s[:, :K].contiguous().view(-1)

        for b in range(B):
            for ki in range(K):
                if src_tokens[b, ki].item() == eos_id:
                    sc = beam_scores[b*K + ki].item()
                    if sc > best_score[b]:
                        best_score[b] = sc
                        best_seq[b]   = sequences[b*K + ki].tolist()

    final_ids = []
    for b in range(B):
        if best_seq[b]:
            final_ids.append(best_seq[b])
        else:
            ki = beam_scores[b*K:(b+1)*K].argmax().item()
            final_ids.append(sequences[b*K + ki].tolist())
    return tokenizer.batch_decode(final_ids, skip_special_tokens=True)


# ─────────────────────────────────────────────────────────────────────────────
# Timed wrappers (mirror baseline.py structure exactly)
# ─────────────────────────────────────────────────────────────────────────────
def timed_generate_seq2seq(model, tokenizer, texts, num_beams,
                            max_new_tokens, batch_size=4):
    all_decoded, lats, total_tok, total_t = [], [], 0, 0.0

    # Warmup (not timed)
    enc = tokenizer(texts[:WARMUP_STEPS], return_tensors="pt",
                    padding=True, truncation=True, max_length=512).to(DEVICE)
    fused_beam_seq2seq(model, tokenizer, enc.input_ids, enc.attention_mask,
                       num_beams, max_new_tokens)
    reset_gpu()

    for btexts in batch_iter(texts, batch_size):
        enc = tokenizer(btexts, return_tensors="pt",
                        padding=True, truncation=True, max_length=512).to(DEVICE)
        sync(); t0 = time.perf_counter()
        decoded = fused_beam_seq2seq(model, tokenizer, enc.input_ids,
                                     enc.attention_mask, num_beams, max_new_tokens)
        sync(); elapsed = time.perf_counter() - t0

        n_out = sum(len(tokenizer.encode(d)) for d in decoded)
        total_tok += n_out;  total_t += elapsed
        ms_per_tok = (elapsed * 1000) / max(n_out / len(btexts), 1)
        lats.extend([ms_per_tok] * len(btexts))
        all_decoded.extend(decoded)

    return all_decoded, lats, total_tok / total_t


def timed_generate_causal(model, tokenizer, texts, num_beams,
                           max_new_tokens, batch_size=4):
    all_decoded, lats, total_tok, total_t = [], [], 0, 0.0

    enc = tokenizer(texts[:WARMUP_STEPS], return_tensors="pt",
                    padding=True, truncation=True, max_length=512).to(DEVICE)
    fused_beam_causal(model, tokenizer, enc.input_ids, enc.attention_mask,
                      num_beams, max_new_tokens)
    reset_gpu()

    for btexts in batch_iter(texts, batch_size):
        enc = tokenizer(btexts, return_tensors="pt",
                        padding=True, truncation=True, max_length=512).to(DEVICE)
        sync(); t0 = time.perf_counter()
        decoded = fused_beam_causal(model, tokenizer, enc.input_ids,
                                    enc.attention_mask, num_beams, max_new_tokens)
        sync(); elapsed = time.perf_counter() - t0

        n_out = sum(len(tokenizer.encode(d, add_special_tokens=False)) for d in decoded)
        total_tok += n_out;  total_t += elapsed
        ms_per_tok = (elapsed * 1000) / max(n_out / len(btexts), 1)
        lats.extend([ms_per_tok] * len(btexts))
        all_decoded.extend(decoded)

    return all_decoded, lats, total_tok / total_t


# ─────────────────────────────────────────────────────────────────────────────
# Task runners — identical structure to baseline.py
# ─────────────────────────────────────────────────────────────────────────────
def run_translation(dataset_name: str, wmt_year: str):
    print(f"\n{'='*55}\n  Variant A | Translation | {dataset_name}\n{'='*55}")
    MODEL_ID  = "facebook/wmt19-en-de"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model     = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16 if DEVICE=="cuda" else torch.float32
    ).to(DEVICE).eval()

    ds  = load_dataset(f"wmt/{wmt_year}", "de-en", split="test", trust_remote_code=True)
    src = [ex["translation"]["en"] for ex in ds.select(range(N_SAMPLES))]
    ref = [ex["translation"]["de"] for ex in ds.select(range(N_SAMPLES))]
    bleu = BLEU()

    for k in BEAM_WIDTHS:
        print(f"  K={k:2d} ...", end=" ", flush=True)
        reset_gpu()
        decoded, lats, tps = timed_generate_seq2seq(
            model, tokenizer, src, num_beams=k, max_new_tokens=MAX_NEW_TOKENS)
        gpu  = peak_mb()
        sc   = bleu.corpus_score(decoded, [ref]).score
        mlat = float(np.mean(lats));  p95 = float(np.percentile(lats, 95))
        print(f"BLEU={sc:.2f}  {tps:.1f} tok/s  lat={mlat:.1f}ms  "
              f"p95={p95:.1f}ms  GPU={gpu:.0f}MB")
        results.append(BenchResult("translation", dataset_name, k,
                                    mlat, p95, tps, gpu, "BLEU", sc))
    del model;  reset_gpu()


def run_summarization():
    print(f"\n{'='*55}\n  Variant A | Summarization | CNN/DailyMail\n{'='*55}")
    MODEL_ID  = "facebook/bart-large-cnn"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model     = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16 if DEVICE=="cuda" else torch.float32
    ).to(DEVICE).eval()

    ds   = load_dataset("cnn_dailymail", "3.0.0", split="test")
    arts = [ex["article"]    for ex in ds.select(range(N_SAMPLES))]
    refs = [ex["highlights"] for ex in ds.select(range(N_SAMPLES))]
    rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    for k in BEAM_WIDTHS:
        print(f"  K={k:2d} ...", end=" ", flush=True)
        reset_gpu()
        decoded, lats, tps = timed_generate_seq2seq(
            model, tokenizer, arts, num_beams=k,
            max_new_tokens=142, batch_size=2,
            )   # batch_size=2: bart is large
        gpu  = peak_mb()
        r1   = float(np.mean([rouge.score(r, h)["rouge1"].fmeasure
                               for r, h in zip(refs, decoded)])) * 100
        mlat = float(np.mean(lats));  p95 = float(np.percentile(lats, 95))
        print(f"ROUGE-1={r1:.2f}  {tps:.1f} tok/s  lat={mlat:.1f}ms  "
              f"p95={p95:.1f}ms  GPU={gpu:.0f}MB")
        results.append(BenchResult("summarization", "cnn_dailymail", k,
                                    mlat, p95, tps, gpu, "ROUGE-1", r1))
    del model;  reset_gpu()


def run_lm_generation():
    print(f"\n{'='*55}\n  Variant A | LM Generation | WikiText-103\n{'='*55}")
    MODEL_ID  = "gpt2-medium"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model     = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16 if DEVICE=="cuda" else torch.float32
    ).to(DEVICE).eval()

    ds      = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    prompts = [t.strip() for t in ds["text"] if len(t.strip()) > 50][:N_SAMPLES]

    for k in BEAM_WIDTHS:
        print(f"  K={k:2d} ...", end=" ", flush=True)
        reset_gpu()
        decoded, lats, tps = timed_generate_causal(
            model, tokenizer, prompts, num_beams=k, max_new_tokens=MAX_NEW_TOKENS)
        gpu  = peak_mb()
        mlat = float(np.mean(lats));  p95 = float(np.percentile(lats, 95))
        print(f"{tps:.1f} tok/s  lat={mlat:.1f}ms  p95={p95:.1f}ms  GPU={gpu:.0f}MB")
        results.append(BenchResult("lm_generation", "wikitext-103", k,
                                    mlat, p95, tps, gpu, "N/A", 0.0))
    del model;  reset_gpu()


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # ── Correctness gate: fused_score must match PyTorch reference ────────────
    print("=== Correctness check: fused_score vs F.log_softmax + add ===")
    torch.manual_seed(0)
    for Kc, Vc in [(2, 32128), (4, 50265), (8, 32000), (16, 50257)]:
        logits   = torch.randn(Kc, Vc, device=DEVICE)
        bscores  = torch.linspace(-0.1, -1.5, Kc, device=DEVICE)
        ref_out  = torch.nn.functional.log_softmax(logits.float(), dim=-1) + bscores[:, None]
        fused_out = fused_score(logits.float(), bscores)
        err = (ref_out - fused_out).abs().max().item()
        status = "OK ✓" if err < 1e-3 else "FAIL ✗"
        print(f"  K={Kc:2d}, V={Vc}: max_abs_err={err:.2e}  {status}")
    print()

    run_translation("wmt14", "wmt14")
    run_translation("wmt17", "wmt17")
    run_summarization()
    run_lm_generation()

    with open(RESULTS_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task","dataset","beam_width","mean_latency_ms","p95_latency_ms",
                    "throughput_tps","peak_gpu_mb","quality_metric","quality_value"])
        for r in results:
            w.writerow([r.task, r.dataset, r.beam_width,
                        f"{r.mean_latency_ms:.3f}", f"{r.p95_latency_ms:.3f}",
                        f"{r.throughput_tps:.2f}", f"{r.peak_gpu_mb:.1f}",
                        r.quality_metric, f"{r.quality_value:.4f}"])
    print(f"\nAll results saved → {RESULTS_CSV}")

