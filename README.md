After feeding deepseek-prover-v2 original miniF2F prompts and asking it to complete the prompt at temperature 0, it could complete the trailing 40% tokens of 1% of the prompts and the trailing 20% tokens of 12% of the prompts, showing almost certain signs of test-set contamination.

## 0.  System check

```bash
nvidia-smi          # driver ≥ 550, CUDA run‑time ≥ 12.2
nvcc --version      # optional; prints CUDA Toolkit if installed
uname -r            # make sure you are on a stock 22.04 kernel
```

---

## 1.  OS packages

```bash
sudo apt update
sudo apt install -y git build-essential cmake ninja-build python3 python3-venv python3-dev
```

---

## 2.  Create an isolated Python env

```bash
mkdir ~/contamination_test && cd contamination_test
python3 -m venv ~/contamination_test/llm-env
source ~/contamination_test/llm-env/bin/activate
python -m pip install -U pip
```

---

## 3.  Install vLLM + helpers

```bash
# pulls the CUDA‑12.8 build of vLLM and matching PyTorch   :contentReference[oaicite:1]{index=1}
pip install "vllm>=0.9.2" --extra-index-url https://download.pytorch.org/whl/cu128

# evaluation extras
pip install datasets tqdm

# pulls in modules used for regular pytorch/cuda kernel based inference
pip install transformers accelerate bitsandbytes
```

---

## 4.  Grab the MiniF²F test split
```bash
python - <<'PY'
from datasets import load_dataset
ds = load_dataset("Tonic/MiniF2F", split="test")  # all 488 problems live here
ds.to_json("miniF2F-test.jsonl", orient="records", lines=True)
print("Wrote", len(ds), "examples to miniF2F-test.jsonl")
PY
```

---

## 5. Download model weights
```bash
export HF_HUB_ENABLE_XET=0
export HF_HUB_DISABLE_TELEMETRY=1
pip install -U modelscope
modelscope download --model deepseek-ai/DeepSeek-Prover-V2-7B --local_dir deepseek_prover_v2_7b
```

---

## 6.  Run the experiment

```bash
# Each Python process cleans up allocator pool and CUDA context after finishing.
# vLLM pass
python3 contamination_test.py --backend vllm  --model /root/deepseek_prover_v2_7b --data miniF2F-test.jsonl --ratios 0.6 --max-tokens 4096 --dtype float16
# Total generation time for backend 'vllm': 326.89 s
# Torch pass
python3 contamination_test.py --backend torch --model /root/deepseek_prover_v2_7b --data miniF2F-test.jsonl --ratios 0.6 --max-tokens 4096 --dtype float16
# Total generation time for backend 'torch': 708.56 s
```

The script writes result statistics to `contamination_out/contamination_metrics.csv`.
