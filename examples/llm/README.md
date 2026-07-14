# Distributed language model training in `blaxbird`

Implements multiple LLM architectures to highlight their different design choices, and to demonstrate distributed training using `blaxbird`.

|  | Gemma4 | Qwen3Next |
|---|---|---|
| Attention mechanism | GQA | Gated DeltaNet + GQA |
| Attention pattern | Interleaved local/global, sliding window | Full causal for GQA layers |
| Positional encoding | Dual-frequency p-RoPE | Partial RoPE |
| KV-cache trick | Key/value reuse on global layers | -- |
| Normalization | RMSNorm, pre-norm, QK-norm | RMSNorm, pre-norm, QK-norm |
| FFN activation | GeGLU | SwiGLU |
| Expert routing | -- | Top-2-of-8 + 1 shared |
| Embedding tying | Untied | Untied |
| Sharding | 2D (FSDP+TP) | 3D (FSDP+TP+expert) |

Run

```shell
XLA_FLAGS="--xla_force_host_platform_device_count=8" uv run python main.py --model {gemma4,qwen3next}
```

to train a tiny LM on `tiny_shakespeare` with sharding:
- Gemma4 `(2,4)` fsdp+tp mesh,
- Qwen3Next `(2,2,2)` fsdp+tp+expert mesh.
