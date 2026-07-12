# LLM reference suite

Three decoder-only transformer reference implementations, each
demonstrating a genuinely different attention/FFN mechanism and each
real-sharded (not just API-compatible) via `blaxbird.train_fn`'s mesh
API:

- **`GemmaDense`** (`gemma.py`) — Gemma-4-style: GQA + RoPE +
  interleaved local/global attention + dense GeGLU FFN. FSDP+TP (2D
  mesh).
- **`DeepSeekMLA`** (`deepseek.py`) — DeepSeek-V2-style: Multi-head
  Latent Attention (low-rank KV compression + decoupled RoPE) + dense
  GeGLU FFN, full causal attention only. FSDP+TP (2D mesh).
- **`MixtralSMoE`** (`mixtral.py`) — Mixtral-style: GQA + RoPE + full
  causal attention + real sparse top-2-of-8 expert routing via
  capacity-based dispatch/combine (not dense-compute-then-select).
  FSDP+TP+Expert (3D mesh) — the dispatch/combine einsum formulation
  lets JAX's SPMD partitioner insert the cross-device communication
  automatically when the expert axis is sharded, no hand-written
  `jax.lax.all_to_all`.

This is reference code, not a trained model: `main.py` runs a handful of
training steps on random token ids to prove each architecture, the
shared `causal_lm` training objective, and the shared (full-prefix-
recompute) generation loop all compose correctly under
`blaxbird.train_fn` -- none of them learn anything meaningful, since
there's no tokenizer or real text dataset wired up.

Not included (out of scope for this reference): a tokenizer, a real text
dataset/dataloader, loading real Gemma-4/DeepSeek-V2/Mixtral weights,
and a production KV-cache for any of the three (including DeepSeekMLA,
where a real cache would normally be the point of the architecture --
the generation loop recomputes the full prefix every step for all three
model families).

Run: `uv run --active python main.py` (single device, degenerate
sharding) or
`XLA_FLAGS="--xla_force_host_platform_device_count=8" uv run --active python main.py`
(real sharding: Gemma/DeepSeek on a simulated `(2,4)` fsdp+tp mesh,
Mixtral on the full simulated `(2,2,2)` fsdp+tp+expert mesh). Loss is
logged via `absl.logging` at INFO level (not raised by default here);
the printed generated-token-id line is the visible completion signal.
