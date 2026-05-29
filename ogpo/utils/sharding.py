import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


def create_mesh(gpu_mode="single"):
    """Create a device mesh based on gpu_mode.

    Args:
        gpu_mode: "single" (device 0 only), "multi" (all devices),
                  or "reserve:N" (all devices except GPU N).
    Returns:
        Mesh over selected devices, or None if single-GPU.
    """
    if gpu_mode == "single":
        return None

    devices = jax.devices()
    if len(devices) <= 1:
        return None

    if gpu_mode == "multi":
        mesh_devices = devices
    elif gpu_mode.startswith("reserve:"):
        exclude = int(gpu_mode.split(":")[1])
        mesh_devices = [d for d in devices if d.id != exclude]
        if not mesh_devices:
            raise ValueError(f"No devices left after reserving GPU {exclude}")
    else:
        raise ValueError(
            f"Unknown gpu_mode: {gpu_mode!r}. Use 'single', 'multi', or 'reserve:N'"
        )

    return Mesh(mesh_devices, axis_names=('data',))


def shard_agent(agent, mesh):
    """Replicate all agent params/opt_state across mesh devices."""
    return jax.device_put(agent, NamedSharding(mesh, P()))


def shard_batch(batch, mesh):
    """Shard UTD-reshaped batch (utd_ratio, batch_size, ...) on axis 1."""
    s = NamedSharding(mesh, P(None, 'data'))
    return jax.tree.map(lambda x: jax.device_put(x, s), batch)


def shard_batch_flat(batch, mesh):
    """Shard flat batch (batch_size, ...) on axis 0. For BC/CalQL phases."""
    s = NamedSharding(mesh, P('data'))
    return jax.tree.map(lambda x: jax.device_put(x, s), batch)
