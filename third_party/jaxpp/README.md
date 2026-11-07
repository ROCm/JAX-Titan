# JaxPP

JaxPP is a JAX library enabling Multiple-Program Multiple-Data (MPMD)
pipeline parallelism through simple user annotations `pipeline_enter_stage(layer)`
and decorators `@mpmd_jit_with_loop`.

JaxPP automatically splits JAX computations into multiple SPMD modules that
are independently jitted and dispatched to different devices.


# Status
JaxPP is under active development, and its APIs are currently unstable and subject to change.

## Changelog

* [Aug 19, 2025] Users must now add a final `pipeline_enter_stage` to mark the last
  stage as well.

# Contacts

As project development is ongoing, we are not accepting Pull Requests to the GitHub repository.
Please contact the maintainers for any questions or concerns.

Issues and feature requests are welcome.




# Installation on ROCm-7.x

JaxPP dependencies are listed in [`pyproject.toml`](https://github.com/ROCm/jaxpp/-/blob/main/pyproject.toml), and we current support JAX-0.7.1 and 0.8.0.

## Installation of JAX-0.7.1

```
https://github.com/ROCm/rocm-jax/tree/rocm-jaxlib-v0.7.1
```

## Installation of cupy
``` bash
git clone --recursive https://github.com/cupy/cupy.git
cd cupy
export HCC_AMDGPU_TARGET=gfx942  # This value should be changed based on your GPU
export __HIP_PLATFORM_HCC__
export CUPY_INSTALL_USE_HIP=1
export ROCM_HOME=/opt/rocm
export HIP_PATH=/opt/rocm
export HIPCC=/opt/rocm/bin/hipcc
export PATH=/opt/rocm/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:${LD_LIBRARY_PATH}

pip install -v .
```
## Installation of JaxPP
```bash
git clone git@github.com:ROCm/jaxpp.git
cd jaxpp
pip install --no-deps .
```

# Verification
You can verify the setup with [`examples/tiny_gpt2_jaxpp_vs_spmd_dp.py`](examples/tiny_gpt2_jaxpp_vs_spmd_dp.py) on a single-node with 8 GPUs.

## running with SPMD
```bash
python examples/tiny_gpt2_jaxpp_vs_spmd_dp.py   --system=spmd_dp   --global-batch=32
```

```
=== SPMD data-parallel (pmap) on 8 GPUs ===
[spmd_dp warmup] loss_sum=10.875779
[spmd_dp 0005/0040] loss_sum=9.484621
[spmd_dp 0010/0040] loss_sum=8.665697
2026-01-11 17:40:07.490737: start profile (spmd_dp)
2026-01-11 17:40:07.491327: E external/xla/xla/python/profiler/internal/python_hooks.cc:416] Can't import tensorflow.python.profiler.trace
[spmd_dp 0015/0040] loss_sum=7.860461
[spmd_dp 0020/0040] loss_sum=7.160092
2026-01-11 17:40:26.985328: stop profile (spmd_dp)
2026-01-11 17:40:26.985854: E external/xla/xla/python/profiler/internal/python_hooks.cc:416] Can't import tensorflow.python.profiler.trace
[spmd_dp 0025/0040] loss_sum=6.619992
[spmd_dp 0030/0040] loss_sum=5.782149
[spmd_dp 0035/0040] loss_sum=4.831417
[spmd_dp 0040/0040] loss_sum=3.508849
[spmd_dp] avg step time per step after profiling (steps 20..39): 1783.959 ms
```


##  running with JAXPP MPMD, TP, DP
```bash
python examples/tiny_gpt2_jaxpp_vs_spmd_dp.py   --system=jaxpp   --pp=2 --dp=1 --tp=4   --global-batch=32
```

```
=== JaxPP MPMD pipeline ===
... ...
[jaxpp warmup] loss_sum=10.874269
[jaxpp 0005/0040] loss_sum=10.056883
[jaxpp 0010/0040] loss_sum=9.022059
2026-01-11 17:45:41.355780: start profile (jaxpp)
2026-01-11 17:45:41.356591: E external/xla/xla/python/profiler/internal/python_hooks.cc:416] Can't import tensorflow.python.profiler.trace
[jaxpp 0015/0040] loss_sum=7.602057
[jaxpp 0020/0040] loss_sum=6.092763
2026-01-11 17:45:45.510771: stop profile (jaxpp)
2026-01-11 17:45:45.511169: E external/xla/xla/python/profiler/internal/python_hooks.cc:416] Can't import tensorflow.python.profiler.trace
[jaxpp 0025/0040] loss_sum=4.633161
[jaxpp 0030/0040] loss_sum=3.209099
[jaxpp 0035/0040] loss_sum=1.882708
[jaxpp 0040/0040] loss_sum=0.803130
[jaxpp] avg step time per step after profiling (steps 20..39): 353.541 ms
```

# Example

The example here shows the typical pattern used in a `flax` module to enable JaxPP.

```python
class ManualStagesModel(nn.Module):
    config: BertConfig
    pipeline_parallelism: int
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.layers = [
            FlaxBertLayer(
                self.config, name=f"flax_bert_layer_{i}", dtype=self.dtype
            )
            for i in range(self.config.num_hidden_layers)
        ]

    def __call__(self, hidden_states):
        num_layers_per_stage = self.config.num_hidden_layers // self.pipeline_parallelism
        stage_id = 0
        for i, layer in enumerate(self.layers):
            outs = layer(hidden_states, None, None)
            hidden_states = outs[0]

            # Mark that we are entering a new stage
            if (i + 1) % num_layers_per_stage == 0:
                hidden_states = jaxpp.pipeline_enter_stage(hidden_states)
                stage_id += 1

        return hidden_states
```

And the code snippet below shows a typical train step function with JaxPP.
```python
def loss(pars, batch):
    res = model.apply(pars, batch)
    return jnp.mean((res - batch) ** 2) / num_mubatches, (res, 4)

# The `mpmd_jit_with_loop` transformation, with `treduce`,
# will make this function execute in mpmd_jit_with_loop fashion over 2 devices
# using the `Eager1F1B` schedule
@partial(jaxpp.mpmd_jit_with_loop, mpmd_mesh=mpmd_mesh)
def pp_train_step(opt_state, pars, batch):
    mubatch_grad = partial(jax.value_and_grad(loss_fn, has_aux=True), params)
    # Compute loss and gradients
    (losses, (pred, _)), grad = jaxpp.treduce(
        mubatch_grad, batch, schedule=jaxpp.Std1F1B(mpmd_mesh.mpmd_dim)
    )
    # Apply the optimizer as usual
    (updates, opt_state) = optimizer.update(grad, opt_state, pars)
    new_pars = optax.apply_updates(pars, updates)
    return opt_state, new_pars, losses, pred
```

To run the train step, we need to create a `MpmdMesh` object, which
is a wrapper of a standard Jax `Mesh` describing which dimension is the
mpmd one.

```python
devices = np.array(jax.devices()[0]).reshape(2, 1, 4)
jax_mesh = jax.sharding.Mesh(devices, ("mpmd", "data", "model"))
mpmd_mesh = jaxpp.MpmdMesh(jax_mesh, "mpmd")
print(mpmd_mesh.lowering_mesh().shape) # OrderedDict([('mpmd', 1), ('data', 1), ('model', 4)])
```

[examples/basic.py](examples/basic.py) provides a complete example.

# Building and Testing Docker Container

JaxPP provides Docker containers for development and testing. Currently it works on `rocm/jax-training:maxtext-v25.9`.



## Running Tests

**Unit Tests**:
```bash
pytest tests
```


Note: The tests require 8 GPUs with sufficient GPU memory.


# Multi-node setup
JaxPP needs to be installed on all nodes that are participating in the parallel
execution and the [installation instruction](#installation-instructions) needs
to be repeated on each node.
In addition, all packages that are needed for the execution of the workload
needs to be installed on all nodes.

# Benchmarks

JaxPP has been tested with several models from MaxText.
We have integrated JaxPP into a [fork of MaxText](https://github.com/NVIDIA/maxtext-jaxpp/blob/jaxpp/main/jaxpp.README.md) with minimal changes.


# Citing JaxPP

```
@misc{jaxpp,
      title={Scaling Deep Learning Training with MPMD Pipeline Parallelism}, 
      author={Anxhelo Xhebraj and Sean Lee and Hanfeng Chen and Vinod Grover},
      year={2024},
      eprint={2412.14374},
      archivePrefix={arXiv},
      primaryClass={cs.DC},
      url={https://arxiv.org/abs/2412.14374}, 
}
```
