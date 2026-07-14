# Distributed language model training in `blaxbird`

Implements multiple LLM architectures to highlight their different design choices, and to demonstrate distributed training using `blaxbird`.

| Design axis | Gemma4 | DeepSeek4 | Qwen3Next |
|---|---|---|---|
| Attention mechanism | GQA | GQA + hybrid CSA/HCA block-compressed attention | Gated DeltaNet (linear, 75% of layers) + GQA (25%) |
| Attention pattern | Interleaved local/global, sliding window | Full causal, block-compressed (CSA: top-k selective; HCA: dense over heavier-compressed blocks) | Full causal (GQA layers only -- DeltaNet layers have no explicit attention pattern, it's a recurrence) |
| Positional encoding | Dual-frequency p-RoPE (theta=1M/rotary 25% on global layers, theta=10k/full rotation on local layers) | RoPE | Partial RoPE (first 25% of head, GQA layers only; DeltaNet layers carry no explicit position embedding) |
| KV-cache trick | Key/value reuse on global layers (no separate value projection) | -- | -- |
| Attention sinks | No | Yes -- learnable per-head, per-branch sink logit lets softmax mass sum to <1 | No |
| Normalization | RMSNorm, pre-norm | RMSNorm, pre-norm | RMSNorm, pre-norm |
| FFN activation | GeGLU | GeGLU | SwiGLU (via SparseMoEFFN) |
| Expert routing | -- (dense FFN) | -- (dense FFN) | Top-2-of-8, capacity-based dispatch, Switch-style aux loss (every layer -- both DeltaNet and GQA layers route through it) |
| Embedding tying | Untied | Untied | Untied |
| Sharding | 2D (FSDP+TP) | 2D (FSDP+TP) | 3D (FSDP+TP+expert) |

Data is byte-level `tiny_shakespeare` (TFDS), vocab_size=256, no
subword tokenizer -- raw bytes in, raw bytes out.

Run

```shell
XLA_FLAGS="--xla_force_host_platform_device_count=8" uv run python main.py --model {gemma4,deepseek4,qwen3next}
```

to train a tiny LM on `tiny_shakespeare`, or omit `--model` to run all
three in turn. `--n-steps` (default 100) controls step count. Meshes:
- Gemma4/DeepSeek4 `(2,4)` fsdp+tp mesh,
- Qwen3Next `(2,2,2)` fsdp+tp+expert mesh.
