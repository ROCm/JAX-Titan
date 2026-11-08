"""Peak compute, memory, power and interconnect of training GPUs, from vendors' published
specs.

Peaks are dense (no structured sparsity), in TFLOP/s per GPU. A value marked as derived
in `notes` is not published as such (for example half of a sparse peak). Measured
attainable rates differ: XLA reaches 1,436 TFLOP/s of BF16 GEMM on MI355X
(`jax_titan.observability.ceilings`).

Interconnect is per GPU and per direction: `scaleup_gbs` to the other GPUs of its
scale-up domain (`scaleup_gpus` of them: an 8-GPU platform, or a 72-GPU Helios rack),
`scaleout_gbs` to other domains. On a fully connected platform (`scaleup_mesh`, the
MI300 and MI350 series) each link reaches one peer, so a collective over k of the GPUs
gets k - 1 links; behind a switch (NVLink, UALink) it gets all of them. Scale-out
depends on the system's NICs, not the GPU; the values assume one 400 Gb/s NIC per GPU
unless noted.
"""

import dataclasses


@dataclasses.dataclass(frozen=True)
class GPU:
  name: str
  bf16_tflops: float | None
  fp8_tflops: float | None
  fp4_tflops: float | None
  hbm_gb: float | None
  hbm_tbps: float | None
  board_power_w: float | None
  scaleup_gpus: int | None
  scaleup_gbs: float | None
  scaleout_gbs: float | None
  source: str
  notes: str = ""
  scaleup_mesh: bool = False

  def scaleup_gbs_for(self, ranks: int) -> float:
    """Per-direction bandwidth a collective over `ranks` GPUs of one domain gets."""
    if self.scaleup_mesh:
      return self.scaleup_gbs * (ranks - 1) / (self.scaleup_gpus - 1)
    return self.scaleup_gbs

  def peak_tflops(self, precision: str) -> float:
    value = {"bf16": self.bf16_tflops, "fp8": self.fp8_tflops,
             "fp4": self.fp4_tflops, "mxfp4": self.fp4_tflops}[precision]
    if not value:
      raise ValueError(f"{self.name} has no published dense {precision} peak")
    return value


_AMD = "https://www.amd.com/en/products/accelerators/instinct"
GPUS = {
    "MI300X": GPU("AMD Instinct MI300X", 1307.4, 2614.9, None, 192, 5.3, 750, 8, 448, 50,
                  f"{_AMD}/mi300/mi300x.html",
                  "scale-up derived: 7 Infinity Fabric links of 128 GB/s (both directions) "
                  "to the platform's other GPUs", scaleup_mesh=True),
    "MI325X": GPU("AMD Instinct MI325X", 1307.4, 2614.9, None, 256, 6.0, 1000, 8, 448, 50,
                  f"{_AMD}/mi300/mi325x.html",
                  "scale-up derived: 7 Infinity Fabric links of 128 GB/s (both directions) "
                  "to the platform's other GPUs", scaleup_mesh=True),
    "MI350X": GPU("AMD Instinct MI350X", 2306.9, 4613.7, 9227.5, 288, 8.0, 1000, 8, 504, 50,
                  f"{_AMD}/mi350/mi350x.html",
                  "peaks derived as 256 CUs x 2.2 GHz x 4,096, 8,192 and 16,384 FLOPs per "
                  "clock (the page rounds to 2.3, 4.6 and 9.2 PFLOP/s); scale-up derived: 7 "
                  "Infinity Fabric links of 144 GB/s (both directions)", scaleup_mesh=True),
    "MI355X": GPU("AMD Instinct MI355X", 2516.6, 5033.2, 10066.3, 288, 8.0, 1400, 8, 504, 50,
                  f"{_AMD}/mi350/mi355x.html",
                  "scale-up derived: 7 Infinity Fabric links of 144 GB/s (both directions)",
                  scaleup_mesh=True),
    "MI455X": GPU("AMD Instinct MI455X", 5033.2, 20132.7, 40265.3, 432, 23.3, None, 72, 1800,
                  300, f"{_AMD}/mi400/mi455x.html",
                  "AMD lists 5, 20.1 and 40.3 PFLOP/s: 256 work group processors x 2.4 GHz x "
                  "8,192, 32,768 and 65,536 FLOPs per clock; no board power published; "
                  "scale-up 3.6 TB/s and scale-out 600 GB/s, both directions, in a 72-GPU "
                  "Helios rack. AMD announced the Helios GPU as MI450 in 2025"),
    "H100": GPU("NVIDIA H100 SXM", 989.4, 1978.9, None, 80, 3.35, 700, 8, 450, 50,
                "https://www.nvidia.com/en-us/data-center/h100/",
                "dense peaks are half the published sparse ones; NVLink 900 GB/s both "
                "directions"),
    "H200": GPU("NVIDIA H200 SXM", 989.4, 1978.9, None, 141, 4.8, 700, 8, 450, 50,
                "https://www.nvidia.com/en-us/data-center/h200/",
                "dense peaks are half the published sparse ones; NVLink 900 GB/s both "
                "directions"),
    "B200": GPU("NVIDIA B200", 2250, 4500, 9000, 180, 8.0, 1000, 8, 900, 50,
                "https://docs.nvidia.com/dgx/dgxb200-user-guide/introduction-to-dgxb200.html",
                "derived per GPU from HGX and DGX B200 totals; dense is half of sparse; "
                "NVLink 1.8 TB/s both directions"),
    "TPUv5p": GPU("Google TPU v5p", 459, None, None, 95, 2.765, None, 8960, None, None,
                  "https://cloud.google.com/tpu/docs/v5p",
                  "the JAX Scaling Book's training examples use it"),
}

ALIASES = {"MI300": "MI300X", "MI325": "MI325X", "MI350": "MI350X", "MI355": "MI355X",
           "MI450": "MI455X", "MI450X": "MI455X", "MI455": "MI455X"}


def _key(name: str) -> str:
  return name.upper().replace("INSTINCT", "").replace("AMD", "").replace(" ", "").replace("-", "")


def get(name: str) -> GPU:
  key = _key(name)
  key = next((v for k, v in ALIASES.items() if k == key), key)
  for k, gpu in GPUS.items():
    if k.upper() == key:
      return gpu
  raise ValueError(f"unknown GPU {name!r}; known: {', '.join(GPUS)} "
                   f"(and {', '.join(ALIASES)})")
