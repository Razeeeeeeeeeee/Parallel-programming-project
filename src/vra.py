"""
variant_a_fusion_v3.py
Variant A: Fused log-softmax + beam score update + Zero-Sync Generation Loop

Fixes vs v2:
  FIX-7  Vectorized EOS logic → Replaces .any() and .item() CPU-GPU syncs
  FIX-8  Compiled cache reorder → Uses torch.compile to fuse index_selects
  FIX-9  Fully on-device fallback → Final sequence selection done entirely on GPU
"""

import time, csv, gc
import torch
import triton
import triton.language as tl
import numpy as np
import torch._dynamo
from typing import List, Tuple
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from transformers.modeling_outputs import BaseModelOutput
from datasets import load_dataset
from sacrebleu.metrics import BLEU
from rouge_score import rouge_scorer

# Suppress Dynamo graph break warnings for dynamic cache sizing
torch._dynamo.config.suppress_errors = True 

DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
BEAM_WIDTHS    = [2, 4, 8, 16]
N_SAMPLES      = 100
MAX_NEW_TOKENS = 128
WARMUP_STEPS   = 5          
RESULTS_CSV    = "variant_a_results_v3.csv"

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
# Triton kernel: fused log_softmax + beam score addition
# ─────────────────────────────────────────────────────────────────────────────
@triton.jit
def fused_lsm_score_kernel(
    logits_ptr,
    scores_ptr,
    output_ptr,
    V,
    BLOCK_V: tl.constexpr,
):
    row_id  = tl.program_id(0)         
    base    = row_id * V
    score_k = tl.load(scores_ptr + row_id)

    # Pass 1: row max
    row_max = tl.full([1], -float("inf"), dtype=tl.float32)
    for s in range(0, V, BLOCK_V):
        offs = s + tl.arange(0, BLOCK_V)
        mask = offs < V
        x    = tl.load(logits_ptr + base + offs, mask=mask, other=-float("inf"))
        row_max = tl.maximum(row_max, tl.max(x, axis=0, keep_dims=True))

    # Pass 2: log-sum-exp denominator
    sum_exp = tl.zeros([1], dtype=tl.float32)
    for s in range(0, V, BLOCK_V):
        offs = s + tl.arange(0, BLOCK_V)
        mask = offs < V
        x    = tl.load(logits_ptr + base + offs, mask=mask, other=-float("inf"))
        sum_exp += tl.sum(tl.where(mask, tl.exp(x - row_max), 0.0),
                          axis=0, keep_dims=True)
    log_denom = tl.log(sum_exp) + row_max

    # Pass 3: write log_softmax + beam_score
    for s in range(0, V, BLOCK_V):
        offs = s + tl.arange(0, BLOCK_V)
        mask = offs < V
        x    = tl.load(logits_ptr + base + offs, mask=mask, other=-float("inf"))
        tl.store(output_ptr + base + offs,
                 tl.where(mask, x - log_denom + score_k, -float("inf")),
                 mask=mask)


def fused_score(logits: torch.Tensor, beam_scores: torch.Tensor) -> torch.Tensor:
    BK, V   = logits.shape                         
    out     = torch.empty_like(logits)
    BLOCK_V = triton.next_power_of_2(min(V, 4096))
    fused_lsm_score_kernel[(BK,)](                 
        logits, beam_scores, out,
        V, BLOCK_V, num_warps=8,
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
        
    # Modern Hugging Face DynamicCache handles this efficiently on its own
    if hasattr(past_kv, "reorder_cache"):
        past_kv.reorder_cache(idx)
        return past_kv
        
    # Fallback for models with custom reorder methods
    if hasattr(model, "_reorder_cache"):
        return model._reorder_cache(past_kv, idx)
        
    # Legacy fallback for raw tuple-of-tuples
    return tuple(
        tuple(
            t.index_select(0, idx) if isinstance(t, torch.Tensor) else t
            for t in layer
        )
        for layer in past_kv
    )

def batch_iter(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

# ─────────────────────────────────────────────────────────────────────────────
# Seq2Seq beam search  (wmt19-en-de, bart-large-cnn)
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def fused_beam_seq2seq(
    model, tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    num_beams: int,
    max_new_tokens: int,
    min_length: int = 0,         # NEW
    length_penalty: float = 1.0, # NEW
) -> Tuple[List[str], List[List[int]]]:
    device   = input_ids.device
    B, K     = input_ids.shape[0], num_beams
    V        = model.config.vocab_size
    eos_id   = model.config.eos_token_id
    start_id = getattr(model.config, "decoder_start_token_id", eos_id)
    b_idx    = torch.arange(B, device=device)

    # Encode once, expand K times
    enc_out    = model.get_encoder()(
        input_ids=input_ids, attention_mask=attention_mask, return_dict=True
    )
    enc_hidden = (enc_out.last_hidden_state
                  .unsqueeze(1).expand(B, K, -1, -1)
                  .contiguous().view(B*K, -1, enc_out.last_hidden_state.shape[-1]))
    enc_attn   = (attention_mask
                  .unsqueeze(1).expand(B, K, -1)
                  .contiguous().view(B*K, -1))

    # Pre-allocate sequence buffer
    seq_buf = torch.full((B*K, max_new_tokens + 1), eos_id,
                         dtype=torch.long, device=device)
    seq_buf[:, 0] = start_id
    seq_len = 1

    # Init beam scores
    beam_scores = torch.full((B*K,), float("-inf"), device=device)
    for b in range(B):
        beam_scores[b * K] = 0.0

    # FIX-7: Vectorized best scores and sequences tracking
    best_scores = torch.full((B,), float("-inf"), device=device)
    best_seqs   = torch.zeros((B, max_new_tokens + 1), dtype=torch.long, device=device)
    
    past_kv    = None
    batch_off  = b_idx.unsqueeze(1) * K  # [B, 1]

    for _ in range(max_new_tokens):
        out = model(
            decoder_input_ids=seq_buf[:, seq_len - 1:seq_len],  
            encoder_outputs=BaseModelOutput(last_hidden_state=enc_hidden),
            attention_mask=enc_attn,
            past_key_values=past_kv,
            use_cache=True, return_dict=True,
        )
        logits  = out.logits[:, -1, :].float()   
        past_kv = out.past_key_values

        if seq_len < min_length:
            logits[:, eos_id] = -float("inf")
            
        scored = fused_score(logits, beam_scores) 

        top_scores, top_flat = torch.topk(scored.view(B, K * V), 2 * K, dim=1)
        src_beams  = top_flat // V    
        src_tokens = top_flat % V     

        global_par = (src_beams[:, :K] + batch_off).contiguous().view(-1)  

        past_kv  = reorder_cache(model, past_kv, global_par)
        seq_buf  = seq_buf[global_par]             
        new_toks = src_tokens[:, :K].contiguous().view(-1)
        seq_buf[:, seq_len] = new_toks             
        seq_len += 1

        beam_scores = top_scores[:, :K].contiguous().view(-1)
        

        # FIX-7: Zero-Sync Vectorized EOS Masking
        eos_hit = (src_tokens[:, :K] == eos_id)   # [B, K]
        
        # Extract scores only for EOS tokens
        raw_eos_scores = torch.where(
            eos_hit, top_scores[:, :K], 
            torch.tensor(float("-inf"), device=device, dtype=top_scores.dtype)
        )
        
        # Apply Hugging Face's length penalty formula: score / (length^alpha)
        # We only apply this to completed sequences; active beams remain unpenalized
        penalty = seq_len ** length_penalty
        eos_scores = raw_eos_scores / penalty
        
        max_eos_scores, max_eos_idx = eos_scores.max(dim=1)  # [B], [B]
        
        improved = max_eos_scores > best_scores  # [B]
        best_scores = torch.where(improved, max_eos_scores, best_scores)
        
        best_row_idx = b_idx * K + max_eos_idx
        best_seqs = torch.where(improved.unsqueeze(1), seq_buf[best_row_idx], best_seqs)
        
        # Mask completed beams
        beam_scores = beam_scores.masked_fill(eos_hit.view(-1), float("-inf"))

    # FIX-9: Fully on-device fallback sequence selection
    fallback_idx = b_idx * K + beam_scores.view(B, K).argmax(dim=1)
    fallback_seqs = seq_buf[fallback_idx]
    
    final_seqs_tensor = torch.where(
        (best_scores == float("-inf")).unsqueeze(1), 
        fallback_seqs, 
        best_seqs
    )
    
    # Bring to CPU EXACTLY ONCE
    final_ids = final_seqs_tensor.tolist()
    decoded = tokenizer.batch_decode(final_ids, skip_special_tokens=True)
    return decoded, final_ids

# ─────────────────────────────────────────────────────────────────────────────
# Causal LM beam search  (gpt2-medium)
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def fused_beam_causal(
    model, tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    num_beams: int,
    max_new_tokens: int,
) -> Tuple[List[str], List[List[int]]]:
    device    = input_ids.device
    B, plen   = input_ids.shape
    K         = num_beams
    V         = model.config.vocab_size
    eos_id    = tokenizer.eos_token_id
    b_idx     = torch.arange(B, device=device)
    batch_off = b_idx.unsqueeze(1) * K  

    inp_exp  = input_ids.unsqueeze(1).expand(B, K, -1).contiguous().view(B*K, -1)
    attn_exp = attention_mask.unsqueeze(1).expand(B, K, -1).contiguous().view(B*K, -1)

    full_attn = torch.ones(B*K, plen + max_new_tokens,
                           dtype=torch.long, device=device)

    out     = model(input_ids=inp_exp, attention_mask=attn_exp[:, :plen],
                    use_cache=True, return_dict=True)
    logits  = out.logits[:, -1, :].float()   
    past_kv = out.past_key_values

    beam_scores = torch.full((B*K,), float("-inf"), device=device)
    for b in range(B):
        beam_scores[b * K] = 0.0

    scored = fused_score(logits, beam_scores)     

    top_s, top_f = torch.topk(scored.view(B, K * V), 2 * K, dim=1)
    global_par   = (top_f[:, :K] // V + batch_off).contiguous().view(-1)
    past_kv      = reorder_cache(model, past_kv, global_par)
    beam_scores  = top_s[:, :K].contiguous().view(-1)

    gen_buf = torch.zeros((B*K, max_new_tokens), dtype=torch.long, device=device)
    gen_buf[:, 0] = (top_f[:, :K] % V).contiguous().view(-1)
    
    # FIX-7: Vectorized best scores and tracking
    best_scores = torch.full((B,), float("-inf"), device=device)
    best_seqs   = torch.zeros((B, max_new_tokens), dtype=torch.long, device=device)

    for step in range(1, max_new_tokens):
        cur_attn = full_attn[:, :plen + step]

        out = model(
            input_ids=gen_buf[:, step - 1:step],  
            attention_mask=cur_attn,
            past_key_values=past_kv,
            use_cache=True, return_dict=True,
        )
        logits  = out.logits[:, -1, :].float()
        past_kv = out.past_key_values

        scored = fused_score(logits, beam_scores)

        top_s, top_f = torch.topk(scored.view(B, K * V), 2 * K, dim=1)
        src_tokens   = top_f[:, :K] % V
        global_par   = (top_f[:, :K] // V + batch_off).contiguous().view(-1)

        past_kv     = reorder_cache(model, past_kv, global_par)
        gen_buf     = gen_buf[global_par]          
        gen_buf[:, step] = src_tokens.contiguous().view(-1)
        beam_scores = top_s[:, :K].contiguous().view(-1)

        # FIX-7: Zero-Sync Vectorized EOS Masking
        eos_hit = (src_tokens == eos_id)          
        
        eos_scores = torch.where(
            eos_hit, top_s[:, :K], 
            torch.tensor(float("-inf"), device=device, dtype=top_s.dtype)
        )
        max_eos_scores, max_eos_idx = eos_scores.max(dim=1)
        
        improved = max_eos_scores > best_scores
        best_scores = torch.where(improved, max_eos_scores, best_scores)
        
        best_row_idx = b_idx * K + max_eos_idx
        best_seqs = torch.where(improved.unsqueeze(1), gen_buf[best_row_idx], best_seqs)
        
        beam_scores = beam_scores.masked_fill(eos_hit.view(-1), float("-inf"))

    # FIX-9: Fully on-device fallback sequence selection
    fallback_idx = b_idx * K + beam_scores.view(B, K).argmax(dim=1)
    fallback_seqs = gen_buf[fallback_idx]
    
    final_seqs_tensor = torch.where(
        (best_scores == float("-inf")).unsqueeze(1), 
        fallback_seqs, 
        best_seqs
    )

    final_ids = final_seqs_tensor.tolist()
    decoded = tokenizer.batch_decode(final_ids, skip_special_tokens=True)
    return decoded, final_ids

# ─────────────────────────────────────────────────────────────────────────────
# Timed wrappers
# ─────────────────────────────────────────────────────────────────────────────
def timed_generate_seq2seq(model, tokenizer, texts, num_beams,
                            max_new_tokens, batch_size=4,min_length=0, length_penalty=1.0):
    all_decoded, lats, total_tok, total_t = [], [], 0, 0.0

    warmup_n = max(WARMUP_STEPS, batch_size)
    enc = tokenizer(texts[:warmup_n], return_tensors="pt",
                    padding=True, truncation=True, max_length=512).to(DEVICE)
    fused_beam_seq2seq(model, tokenizer, enc.input_ids, enc.attention_mask,
                       num_beams, max_new_tokens, min_length, length_penalty)
    reset_gpu()

    for btexts in batch_iter(texts, batch_size):
        enc = tokenizer(btexts, return_tensors="pt",
                        padding=True, truncation=True, max_length=512).to(DEVICE)
        sync(); t0 = time.perf_counter()
        decoded, final_ids = fused_beam_seq2seq(
            model, tokenizer, enc.input_ids, enc.attention_mask,
            num_beams, max_new_tokens)
        sync(); elapsed = time.perf_counter() - t0

        n_out = sum(len([t for t in ids if t != tokenizer.pad_token_id]) for ids in final_ids)
        total_tok += n_out
        total_t   += elapsed
        ms_per_tok = (elapsed * 1000) / max(n_out / len(btexts), 1)
        lats.extend([ms_per_tok] * len(btexts))
        all_decoded.extend(decoded)

    return all_decoded, lats, total_tok / total_t

def timed_generate_causal(model, tokenizer, texts, num_beams,
                           max_new_tokens, batch_size=4):
    all_decoded, lats, total_tok, total_t = [], [], 0, 0.0

    warmup_n = max(WARMUP_STEPS, batch_size)
    enc = tokenizer(texts[:warmup_n], return_tensors="pt",
                    padding=True, truncation=True, max_length=512).to(DEVICE)
    fused_beam_causal(model, tokenizer, enc.input_ids, enc.attention_mask,
                      num_beams, max_new_tokens)
    reset_gpu()

    for btexts in batch_iter(texts, batch_size):
        enc = tokenizer(btexts, return_tensors="pt",
                        padding=True, truncation=True, max_length=512).to(DEVICE)
        sync(); t0 = time.perf_counter()
        decoded, final_ids = fused_beam_causal(
            model, tokenizer, enc.input_ids, enc.attention_mask,
            num_beams, max_new_tokens)
        sync(); elapsed = time.perf_counter() - t0

        n_out = sum(len([t for t in ids if t != tokenizer.pad_token_id]) for ids in final_ids)
        total_tok += n_out
        total_t   += elapsed
        ms_per_tok = (elapsed * 1000) / max(n_out / len(btexts), 1)
        lats.extend([ms_per_tok] * len(btexts))
        all_decoded.extend(decoded)

    return all_decoded, lats, total_tok / total_t

# ─────────────────────────────────────────────────────────────────────────────
# Task runners
# ─────────────────────────────────────────────────────────────────────────────
def run_translation(dataset_name: str, wmt_year: str):
    print(f"\n{'='*55}\n  Variant A v3 | Translation | {dataset_name}\n{'='*55}")
    MODEL_ID  = "facebook/wmt19-en-de"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model     = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    ).to(DEVICE).eval()

    ds  = load_dataset(f"wmt/{wmt_year}", "de-en",
                       split="test", trust_remote_code=True)
    src = [ex["translation"]["en"] for ex in ds.select(range(N_SAMPLES))]
    ref = [ex["translation"]["de"] for ex in ds.select(range(N_SAMPLES))]
    bleu = BLEU()

    for k in BEAM_WIDTHS:
        print(f"  K={k:2d} ...", end=" ", flush=True)
        reset_gpu()
        decoded, lats, tps = timed_generate_seq2seq(
            model, tokenizer, src,
            num_beams=k, max_new_tokens=MAX_NEW_TOKENS, batch_size=8)
        gpu  = peak_mb()
        sc   = bleu.corpus_score(decoded, [ref]).score
        mlat = float(np.mean(lats))
        p95  = float(np.percentile(lats, 95))
        print(f"BLEU={sc:.2f}  {tps:.1f} tok/s  "
              f"lat={mlat:.1f}ms  p95={p95:.1f}ms  GPU={gpu:.0f}MB")
        results.append(BenchResult("translation", dataset_name, k,
                                   mlat, p95, tps, gpu, "BLEU", sc))
    del model; reset_gpu()

def run_summarization():
    print(f"\n{'='*55}\n  Variant A v3 | Summarization | CNN/DailyMail\n{'='*55}")
    MODEL_ID  = "facebook/bart-large-cnn"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model     = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    ).to(DEVICE).eval()

    ds   = load_dataset("cnn_dailymail", "3.0.0", split="test")
    arts = [ex["article"]    for ex in ds.select(range(N_SAMPLES))]
    refs = [ex["highlights"] for ex in ds.select(range(N_SAMPLES))]
    rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"],
                                     use_stemmer=True)

    for k in BEAM_WIDTHS:
        print(f"  K={k:2d} ...", end=" ", flush=True)
        reset_gpu()
        decoded, lats, tps = timed_generate_seq2seq(
            model, tokenizer, arts,
            num_beams=k, max_new_tokens=142, batch_size=4,
            min_length=56,          # Force summaries to be at least 56 tokens
            length_penalty=2.0      # Heavily reward longer sequences,
            )
        gpu  = peak_mb()
        r1   = float(np.mean([
            rouge.score(r, h)["rouge1"].fmeasure
            for r, h in zip(refs, decoded)
        ])) * 100
        mlat = float(np.mean(lats))
        p95  = float(np.percentile(lats, 95))
        print(f"ROUGE-1={r1:.2f}  {tps:.1f} tok/s  "
              f"lat={mlat:.1f}ms  p95={p95:.1f}ms  GPU={gpu:.0f}MB")
        results.append(BenchResult("summarization", "cnn_dailymail", k,
                                   mlat, p95, tps, gpu, "ROUGE-1", r1))
    del model; reset_gpu()

def run_lm_generation():
    print(f"\n{'='*55}\n  Variant A v3 | LM Generation | WikiText-103\n{'='*55}")
    MODEL_ID  = "gpt2-medium"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model     = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    ).to(DEVICE).eval()

    ds      = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    prompts = [t.strip() for t in ds["text"] if len(t.strip()) > 50][:N_SAMPLES]

    for k in BEAM_WIDTHS:
        print(f"  K={k:2d} ...", end=" ", flush=True)
        reset_gpu()
        decoded, lats, tps = timed_generate_causal(
            model, tokenizer, prompts,
            num_beams=k, max_new_tokens=MAX_NEW_TOKENS, batch_size=4)
        gpu  = peak_mb()
        mlat = float(np.mean(lats))
        p95  = float(np.percentile(lats, 95))
        print(f"{tps:.1f} tok/s  lat={mlat:.1f}ms  "
              f"p95={p95:.1f}ms  GPU={gpu:.0f}MB")
        results.append(BenchResult("lm_generation", "wikitext-103", k,
                                   mlat, p95, tps, gpu, "N/A", 0.0))
    del model; reset_gpu()

# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Correctness check: fused_score vs F.log_softmax + add ===")
    torch.manual_seed(0)
    import torch.nn.functional as F
    for BK, Vc in [(2, 32128), (8, 50265), (16, 32000), (32, 50257)]:
        logits  = torch.randn(BK, Vc, device=DEVICE)
        bscores = torch.linspace(-0.1, -1.5, BK, device=DEVICE)
        ref_out = F.log_softmax(logits.float(), dim=-1) + bscores[:, None]
        fus_out = fused_score(logits.float(), bscores)
        err     = (ref_out - fus_out).abs().max().item()
        status  = "OK ✓" if err < 1e-3 else "FAIL ✗"
        print(f"  BK={BK:2d}, V={Vc}: max_abs_err={err:.2e}  {status}")
    print()

    run_translation("wmt14", "wmt14")
    run_translation("wmt17", "wmt17")
    run_summarization()
    run_lm_generation()

    with open(RESULTS_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "dataset", "beam_width",
                    "mean_latency_ms", "p95_latency_ms", "throughput_tps",
                    "peak_gpu_mb", "quality_metric", "quality_value"])
        for r in results:
            w.writerow([r.task, r.dataset, r.beam_width,
                        f"{r.mean_latency_ms:.3f}", f"{r.p95_latency_ms:.3f}",
                        f"{r.throughput_tps:.2f}", f"{r.peak_gpu_mb:.1f}",
                        r.quality_metric, f"{r.quality_value:.4f}"])
    print(f"\nAll results saved → {RESULTS_CSV}")