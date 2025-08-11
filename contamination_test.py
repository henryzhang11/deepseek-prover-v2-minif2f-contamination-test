# use grpo_env for this script
import argparse, json, pathlib, re, csv, time
from collections import defaultdict
from typing      import List, Dict, Tuple
import torch
from tqdm        import tqdm
import string
from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------- helpers ---------------------------------------------------------

def make_prompt(q_tokens: List[int], ratio: float):
    cut = int(len(q_tokens) * ratio)
    return q_tokens[:cut], q_tokens[cut:]

def load_minif2f(path: str):
    records = [json.loads(l) for l in open(path)]
    out = []
    for ex in records:
        q = ex["formal_statement"]
        out.append({"question": q})
    return out

# ---------- timing utilities ------------------------------------------------

class Timer:
    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start = time.perf_counter()
        return self
    def __exit__(self, *exc):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.end = time.perf_counter()
        self.elapsed = self.end - self.start

# ---------- backend: vLLM ---------------------------------------------------

def init_vllm(model_name: str, max_tokens: int, dtype: str):
    torch_dtype = getattr(torch, dtype)
    llm = LLM(model=model_name, gpu_memory_utilization=0.95, dtype=getattr(torch, dtype), max_model_len=max_tokens, max_num_batched_tokens=8192,)
    tok = llm.get_tokenizer()
    sp  = SamplingParams(temperature=0.0,
                         top_p=1.0,
                         top_k=-1,
                         max_tokens=max_tokens)
    return llm, tok, sp

def run_vllm(llm, tok, base_sp, prompt: List[str], max_new: int) -> Tuple[List[int], int]:
    """Return (text, gen_tokens)."""
    sp = SamplingParams(
        temperature=base_sp.temperature,
        top_p=base_sp.top_p,
        top_k=base_sp.top_k,
        max_tokens=max_new
    )
    out = llm.generate(None, sp, [prompt])[0].outputs[0].token_ids
    out_len = len(out)
    return out, out_len

# ---------- backend: torch --------------------------------------------------

def init_torch(model_name: str, dtype: str):
    torch_dtype = getattr(torch, dtype)
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="cuda:0",      # will spread across GPUs if big
    )
    model.eval()
    return model, tok

@torch.inference_mode()
def run_torch(model, tok, prompt_ids: List[int], max_new: int):
    input_ids = torch.tensor([prompt_ids], device=model.device)
    attn_mask = torch.ones_like(input_ids)
    with torch.cuda.amp.autocast(enabled=(model.dtype in {torch.float16, torch.bfloat16})):
        gen_out = model.generate(
            input_ids=input_ids,
            attention_mask=attn_mask,
            do_sample=False,
            max_new_tokens=max_new,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
        )
    new_tok = gen_out[:, input_ids.shape[1]:]
    gen_ids = new_tok.squeeze(0).tolist()
    return gen_ids, len(gen_ids)

# ---------- main ------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Any HF model path (works for both vLLM and transformers)")
    p.add_argument("--data",  required=True)
    p.add_argument("--ratios", default=[0.8, 0.6, 0.4], nargs="+", type=float)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--output-dir", default="contamination_out")
    p.add_argument("--backend", choices=["vllm", "torch", "both"],
                   default="both",
                   help="Which inference backend(s) to run and time.")
    p.add_argument("--dtype", default="float16",
                   choices=["float16", "bfloat16", "float32"],
                   help="Computation dtype for both backends.")
    p.add_argument("--warmup", type=int, default=2,
                   help="Number of examples to warm up each backend.")
    args = p.parse_args()

    pathlib.Path(args.output_dir).mkdir(exist_ok=True)

    # 0. load benchmark
    data = load_minif2f(args.data)

    # 1. init metrics
    results = defaultdict(lambda: defaultdict(list))
    total_elapsed = defaultdict(float)

    # 2. init backends
    backends = []
    if args.backend in ("vllm", "both"):
        llm, tok_v, sp = init_vllm(args.model, args.max_tokens, args.dtype)
        backends.append(("vllm", llm, tok_v, sp))
    if args.backend in ("torch", "both"):
        model_t, tok_t = init_torch(args.model, args.dtype)
        backends.append(("torch", model_t, tok_t, None))

    # 3. warmup (optional but helps stable timing)
    for name, obj, tok, sp_or_none in backends:
        for i in range(min(args.warmup, len(data))):
            q_tokens = tok.encode(data[i]["question"])
            prompt, answer = make_prompt(q_tokens, args.ratios[0])
            if name == "vllm":
                _ = run_vllm(obj, tok, sp_or_none, prompt, len(answer))
            else:
                _ = run_torch(obj, tok, prompt, len(answer))

    # 4. main loop
    for ex in tqdm(data, desc="Problems"):
        for name, obj, tok, sp_or_none in backends:
            q_tokens = tok.encode(ex["question"])
            for r in args.ratios:
                prompt, answer = make_prompt(q_tokens, r)
                prompt_tok = len(prompt)  # or same formula as above
                with Timer() as t:
                    if name == "vllm":
                        pred, pred_len = run_vllm(obj, tok, sp_or_none, prompt, len(answer)) # changed run_vllm
                    else:
                        pred, pred_len = run_torch(obj, tok, prompt, len(answer))
                elapsed = t.elapsed
                total_elapsed[name] += elapsed
                print("\nPROMPT:\n", tok.decode(prompt), "\nMODEL:\n", tok.decode(pred), "\nANSWER:\n", tok.decode(answer))
                # exact match
                WRAPPERS = {tok.bos_token_id, tok.eos_token_id, tok.pad_token_id}
                def strip_wrappers(seq):
                    return [t for t in seq if t not in WRAPPERS]
                gold  = strip_wrappers(answer)   # the reference proof
                guess = strip_wrappers(pred)     # what the model produced
                em = float(len(guess) == len(gold) and guess == gold)
                # throughput
                tok_per_s  = pred_len / elapsed if elapsed > 0 else 0.0
                prefix = f"{name}_{int(r*100)}"
                results[prefix]['em'].append(em)
                results[prefix]['latency_s'].append(elapsed)
                results[prefix]['gen_tok'].append(pred_len)
                results[prefix]['tok_per_s'].append(tok_per_s)
                results[prefix]['prompt_tok'].append(len(prompt))

    # 5. aggregate and save
    out_csv = f"{args.output_dir}/contamination_metrics.csv"
    with open(out_csv, "a", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow([
            "backend", "prompt_ratio", "EM",
            "latency_s", "gen_tok", "tok_per_s",
            "prompt_tok"
        ])

        for prefix in sorted(results):
            parts = prefix.split("_")
            backend = parts[0]
            ratio   = float(parts[1])/100.0
            r_ = results[prefix]
            n = len(r_['em'])
            em_avg    = sum(r_['em'])    / n
            lat_avg   = sum(r_['latency_s']) / n
            gen_sum   = sum(r_['gen_tok'])
            tokps_avg = sum(r_['tok_per_s']) / n
            ptok_avg  = sum(r_['prompt_tok']) / n

            writer.writerow([
                backend, ratio, f"{em_avg:.4f}",
                f"{lat_avg:.4f}", gen_sum, f"{tokps_avg:.2f}",
                f"{ptok_avg:.1f}"
            ])

            print(
                f"- Backend: {backend.upper()} | - Hint ratio: {int(ratio*100):>3}% "
                f"- EM: {em_avg:.4f} | Lat: {lat_avg:.2f}s | Tok/s: {tokps_avg:.1f}"
            )

    print(f"\nSaved metrics to {out_csv}")
    for b, secs in total_elapsed.items():
        print(f"Total generation time for backend '{b}': {secs:.2f} s ({secs/60:.2f} min)")

if __name__ == "__main__":
    main()
