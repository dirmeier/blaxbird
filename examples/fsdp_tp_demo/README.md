# FSDP + TP sharding demo

Demonstrates real 2D-mesh FSDP+tensor-parallel sharding via
`flax.nnx.with_partitioning` + `nnx.get_named_sharding`, wired through
`blaxbird.train_fn`'s `mesh=`/`data_partition_spec=` parameters.

`ShardedMLP` (`model.py`) annotates its up-projection kernel with
`("fsdp", "tp")` and its down-projection kernel with `("tp", "fsdp")` --
standard Megatron-style column-parallel-then-row-parallel sharding,
combined with FSDP on the same two axes.

No real multi-GPU/TPU hardware needed to see actual multi-device
sharding: run with
`XLA_FLAGS="--xla_force_host_platform_device_count=4" python main.py` to
simulate 4 CPU devices arranged as a (2, 2) fsdp x tp mesh. Without that
env var this still runs (single device, degenerate/unsharded).
