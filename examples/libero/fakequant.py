"""
Fake-quantization wrapper for VLA models (OpenPI).

Usage:
    from fakequant import apply_fake_quant, FakeQuantLinear, CalibrationRunner

    apply_fake_quant(model, bits_w=4, bits_a=4, group_size=256)

    runner = CalibrationRunner(model)
    runner.start()
    for batch in calibration_data:
        runner.run_batch(batch)
    runner.finish()
    runner.save_calibration("./calib/")

    runner = CalibrationRunner(model)
    runner.load_calibration("./calib/")

    layer.set_w_bits(4)
    layer.set_a_bits(8)

    remove_fake_quant(model)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, Sequence


# ── Supported bit-widths (defined before FakeQuantLinear so helpers can use it) ──

SUPPORTED_BITS_GLOBAL = (2, 4, 8, 16)


# ── Global bit-width state ──────────────────────────────────────────────────────

_current_w_bits: int = 4
_current_a_bits: int = 4
_per_layer_w_bits_override: dict[int, int] = {}
_per_layer_a_bits_override: dict[int, int] = {}


def _validate_bits(bits: int) -> None:
    if bits not in SUPPORTED_BITS_GLOBAL:
        raise ValueError(f"Unsupported bits {bits}. Supported: {SUPPORTED_BITS_GLOBAL}")


def set_global_bits(bits: Optional[int] = None, *, w: Optional[int] = None, a: Optional[int] = None) -> None:
    global _current_w_bits, _current_a_bits
    if bits is not None:
        if w is not None or a is not None:
            raise ValueError("Use either set_global_bits(bits) or set_global_bits(w=..., a=...), not both.")
        _validate_bits(bits)
        _current_w_bits = bits
        _current_a_bits = bits
        return
    if w is not None:
        _validate_bits(w)
        _current_w_bits = w
    if a is not None:
        _validate_bits(a)
        _current_a_bits = a


def set_global_bits_w(bits: int) -> None:
    _validate_bits(bits)
    global _current_w_bits
    _current_w_bits = bits


def set_global_bits_a(bits: int) -> None:
    _validate_bits(bits)
    global _current_a_bits
    _current_a_bits = bits


# ── Helpers ─────────────────────────────────────────────────────────────────────

def stochastic_round(x: torch.Tensor) -> torch.Tensor:
    floor_x = torch.floor(x)
    return floor_x + (torch.rand_like(x - floor_x) < (x - floor_x)).to(dtype=x.dtype)


def _flatten_tokens(x: torch.Tensor) -> torch.Tensor:
    """Flatten batch dims for per-token quantization. 1D input gets fake batch dim."""
    if x.dim() == 1:
        return x.reshape(1, -1)
    return x.flatten(0, -2)


def _calc_asym_qparams_from_minmax(
    x_min_vals: torch.Tensor,
    x_max_vals: torch.Tensor,
    bits: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Asymmetric affine quantization params from per-token min/max.

    All internal computation is performed in float32 for numerical stability.
    Forces the real quantization range to include 0, handling all cases correctly:
      - Positive-only row: real range is [0, x_max], zero_point = 0.
      - Negative-only row: real range is [x_min, 0], zero_point = q_max.
      - Cross-zero row: normal [x_min, x_max] range.
      - All-zeros row: scale clamped to 1e-6, zero_point = 0.
      - Constant positive / constant negative rows preserve the value
        instead of being clamped to a single quant level.

    Returns scales and zero_pts as float32 tensors.
    """
    x_min_vals = x_min_vals.float()
    x_max_vals = x_max_vals.float()
    q_min = 0
    q_max = (2 ** bits) - 1
    q_range = q_max - q_min

    zeros = torch.zeros_like(x_min_vals)
    x_min = torch.minimum(x_min_vals, zeros)
    x_max = torch.maximum(x_max_vals, zeros)

    diff = x_max - x_min
    scales = torch.clamp(diff / q_range, min=1e-6)

    zero_pts = torch.round(q_min - x_min / scales)
    zero_pts = torch.clamp(zero_pts, q_min, q_max)

    return scales, zero_pts


def _compute_weight_scale_percentile(
    weight: torch.Tensor,
    bits: int,
    group_size: int,
    percentile: float,
) -> torch.Tensor:
    """Per-group symmetric signed weight scales using percentile clipping (float32)."""
    weight_f = weight.float()
    out_dim = weight_f.shape[0]
    num_groups = (out_dim + group_size - 1) // group_size
    scales = torch.ones(num_groups, 1, device=weight_f.device, dtype=torch.float32)
    levels = (2 ** (bits - 1)) - 1
    for g in range(num_groups):
        start, end = g * group_size, min((g + 1) * group_size, out_dim)
        w_g = weight_f[start:end].flatten()
        if w_g.numel() == 0:
            continue
        sorted_abs = torch.sort(torch.abs(w_g))[0]
        pct_idx = int(round(percentile / 100.0 * (sorted_abs.numel() - 1)))
        pct_idx = min(max(pct_idx, 0), sorted_abs.numel() - 1)
        pct_val = sorted_abs[pct_idx]
        if pct_val > 1e-9:
            scales[g] = pct_val / levels
    return scales


# ── Core wrapper ────────────────────────────────────────────────────────────────

class FakeQuantLinear(nn.Module):
    SUPPORTED_BITS = SUPPORTED_BITS_GLOBAL

    def __init__(
        self,
        linear: nn.Linear,
        group_size: int = 256,
        percentile_w: float = 99.9,
        percentile_a: float = 99.9,
        quantize_activations: bool = True,
        stochastic_round: bool = False,
        exclude_bits: Sequence[int] = (2,),
    ):
        super().__init__()
        self._linear = linear
        self._group_size = group_size
        self._percentile_w = percentile_w
        self._percentile_a = percentile_a  # currently unused; activation qparams use exact min/max
        self._quantize_activations = quantize_activations
        self._stochastic_round = stochastic_round
        # forbid 4 from exclude_bits because excluded bits fall back to 4
        exclude_bits = list(exclude_bits)
        if 4 in exclude_bits:
            raise ValueError("exclude_bits must not include 4 because excluded bits fall back to 4.")
        for b in exclude_bits:
            if b not in SUPPORTED_BITS_GLOBAL:
                raise ValueError(f"Invalid bit {b} in exclude_bits. Supported: {SUPPORTED_BITS_GLOBAL}")
        self._exclude_bits = set(exclude_bits)
        self._layer_id: int = -1

        self._w_scales_cache: dict[int, Optional[torch.Tensor]] = {}
        self._num_groups: int = 0
        self._w_initialized: bool = False

        # activation qparams: dict[bits -> (scales, zero_pts)]
        self._x_qparams_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

        # calibration
        self._calibrating: bool = False
        self._calib_x_mins: Optional[list[torch.Tensor]] = None
        self._calib_x_maxs: Optional[list[torch.Tensor]] = None

        # online collection
        self._collecting_online: bool = False
        self._online_x_mins: Optional[list[torch.Tensor]] = None
        self._online_x_maxs: Optional[list[torch.Tensor]] = None

    # ── properties ──────────────────────────────────────────────────────────────

    @property
    def in_features(self) -> int:
        return self._linear.in_features

    @property
    def out_features(self) -> int:
        return self._linear.out_features

    @property
    def weight(self) -> nn.Parameter:
        return self._linear.weight

    @property
    def bias(self) -> Optional[nn.Parameter]:
        return self._linear.bias

    @property
    def current_w_bits(self) -> int:
        return _per_layer_w_bits_override.get(self._layer_id, _current_w_bits)

    @property
    def current_a_bits(self) -> int:
        return _per_layer_a_bits_override.get(self._layer_id, _current_a_bits)

    @property
    def effective_w_bits(self) -> int:
        bits = self.current_w_bits
        return 4 if bits in self._exclude_bits else bits

    @property
    def effective_a_bits(self) -> int:
        bits = self.current_a_bits
        return 4 if bits in self._exclude_bits else bits

    # ── bit-width control ───────────────────────────────────────────────────────

    def set_bits(self, bits: int) -> None:
        self.set_w_bits(bits)
        self.set_a_bits(bits)

    def set_w_bits(self, bits: int) -> None:
        if bits not in self.SUPPORTED_BITS:
            raise ValueError(f"Unsupported bits {bits}. Supported: {self.SUPPORTED_BITS}")
        _per_layer_w_bits_override[self._layer_id] = bits

    def set_a_bits(self, bits: int) -> None:
        if bits not in self.SUPPORTED_BITS:
            raise ValueError(f"Unsupported bits {bits}. Supported: {self.SUPPORTED_BITS}")
        _per_layer_a_bits_override[self._layer_id] = bits

    # ── weight quantization ─────────────────────────────────────────────────────

    def _ensure_weight_scales(self) -> None:
        if self._w_initialized:
            return
        w = self._linear.weight.detach()
        self._num_groups = (w.shape[0] + self._group_size - 1) // self._group_size
        for bits in self.SUPPORTED_BITS:
            self._w_scales_cache[bits] = (
                None if bits in self._exclude_bits
                else _compute_weight_scale_percentile(w, bits, self._group_size, self._percentile_w)
                # returns float32, stays float32 in cache
            )
        self._w_initialized = True

    def _quantize_weight(self, w: torch.Tensor, bits: int) -> torch.Tensor:
        if bits >= 16:
            return w
        bits = 4 if bits in self._exclude_bits else bits
        self._ensure_weight_scales()
        orig_dtype = w.dtype
        w_f = w.float()
        scales = self._w_scales_cache[bits].to(device=w_f.device, dtype=torch.float32)
        self._w_scales_cache[bits] = scales
        levels = (2 ** (bits - 1)) - 1
        q_min, q_max = -levels, levels
        out_dim, in_dim = w_f.shape
        gs, ng = self._group_size, self._num_groups
        pad_out = ng * gs - out_dim
        w_padded = F.pad(w_f, (0, 0, 0, pad_out)) if pad_out else w_f
        w_groups = (w_padded.view(ng, gs, in_dim) / scales.view(-1, 1, 1))
        w_q = stochastic_round(w_groups) if self._stochastic_round else torch.round(w_groups)
        w_deq = (w_q.clamp(q_min, q_max) * scales.view(-1, 1, 1)).view(-1, in_dim)
        return w_deq[:-pad_out, :].to(dtype=orig_dtype) if pad_out else w_deq.to(dtype=orig_dtype)

    # ── activation quantization ──────────────────────────────────────────────────

    def _quantize_activation(self, x: torch.Tensor, bits: int) -> torch.Tensor:
        if not self._quantize_activations or bits >= 16:
            return x
        bits = 4 if bits in self._exclude_bits else bits
        orig_dtype = x.dtype
        x_f = _flatten_tokens(x).float()
        q_min, q_max = 0, (2 ** bits) - 1
        if bits in self._x_qparams_cache:
            num_tokens = x_f.shape[0]
            scales, zero_pts = self._x_qparams_cache[bits]
            scales = scales.to(device=x_f.device, dtype=torch.float32).expand(num_tokens, -1)
            zero_pts = zero_pts.to(device=x_f.device, dtype=torch.float32).expand(num_tokens, -1)
        else:
            x_mins, _ = x_f.min(dim=1, keepdim=True)
            x_maxs, _ = x_f.max(dim=1, keepdim=True)
            if self._collecting_online:
                self._online_x_mins.append(x_mins.detach().clone())
                self._online_x_maxs.append(x_maxs.detach().clone())
            scales, zero_pts = _calc_asym_qparams_from_minmax(x_mins, x_maxs, bits)
        x_norm = x_f / scales + zero_pts
        x_q = stochastic_round(x_norm) if self._stochastic_round else torch.round(x_norm)
        x_q = torch.clamp(x_q, q_min, q_max)
        x_deq = (x_q - zero_pts) * scales
        return x_deq.to(dtype=orig_dtype).reshape(x.shape)

    # ── calibration ─────────────────────────────────────────────────────────────

    def start_calibration(self) -> None:
        self._calibrating = True
        self._calib_x_mins = []
        self._calib_x_maxs = []
        self._x_qparams_cache.clear()

    def finish_calibration(self) -> None:
        if not self._calibrating or not self._calib_x_mins or not self._calib_x_maxs:
            self._calibrating = False
            self._calib_x_mins = None
            self._calib_x_maxs = None
            return
        self._finalize_qparams(self._calib_x_mins, self._calib_x_maxs)
        self._calibrating = False
        self._calib_x_mins = None
        self._calib_x_maxs = None

    # ── online collection ───────────────────────────────────────────────────────

    def start_online_collection(self) -> None:
        self._collecting_online = True
        self._online_x_mins = []
        self._online_x_maxs = []
        # clear calibration cache so _quantize_activation takes the no-cache path
        # and actually collects statistics into _online_x_mins/_online_x_maxs
        self._x_qparams_cache.clear()

    def finish_online_collection(self) -> None:
        if not self._collecting_online or not self._online_x_mins:
            self._collecting_online = False
            self._online_x_mins = None
            self._online_x_maxs = None
            return
        self._finalize_qparams(self._online_x_mins, self._online_x_maxs)
        self._collecting_online = False
        self._online_x_mins = None
        self._online_x_maxs = None

    def _finalize_qparams(self, mins_list: list[torch.Tensor], maxs_list: list[torch.Tensor]) -> None:
        x_mins_cat = torch.cat(mins_list, dim=0)
        x_maxs_cat = torch.cat(maxs_list, dim=0)
        x_min_agg, _ = x_mins_cat.min(dim=0, keepdim=True)
        x_max_agg, _ = x_maxs_cat.max(dim=0, keepdim=True)
        self._x_qparams_cache.clear()
        for bits in self.SUPPORTED_BITS:
            if bits in self._exclude_bits or bits >= 16:
                continue
            self._x_qparams_cache[bits] = _calc_asym_qparams_from_minmax(x_min_agg, x_max_agg, bits)

    # ── persistence ─────────────────────────────────────────────────────────────

    def save_calibration(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "x_qparams_cache": {bits: (s.cpu(), z.cpu()) for bits, (s, z) in self._x_qparams_cache.items()},
            "percentile_a": self._percentile_a,
        }, path)

    def load_calibration(self, path: str | Path) -> None:
        state = torch.load(path, map_location="cpu", weights_only=True)
        self._x_qparams_cache = {int(bits): (s, z) for bits, (s, z) in state["x_qparams_cache"].items()}

    # ── forward ────────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._calibrating:
            x_flat = _flatten_tokens(x).float()
            self._calib_x_mins.append(x_flat.min(dim=1, keepdim=True)[0].detach().clone())
            self._calib_x_maxs.append(x_flat.max(dim=1, keepdim=True)[0].detach().clone())
            return F.linear(x, self._linear.weight, self._linear.bias)
        w_bits, a_bits = self.current_w_bits, self.current_a_bits
        if w_bits >= 16 and a_bits >= 16:
            return self._linear(x)
        return F.linear(
            self._quantize_activation(x, a_bits),
            self._quantize_weight(self._linear.weight, w_bits),
            self._linear.bias,
        )


# ── Layer registry ───────────────────────────────────────────────────────────────

def get_all_fakequant_layers(model: nn.Module) -> list[FakeQuantLinear]:
    return [m for m in model.modules() if isinstance(m, FakeQuantLinear)]


def register_all_fakequant_layers(model: nn.Module) -> int:
    layers = get_all_fakequant_layers(model)
    # clear stale per-layer overrides so old IDs don't leak into a new model
    _per_layer_w_bits_override.clear()
    _per_layer_a_bits_override.clear()
    for idx, layer in enumerate(layers):
        layer._layer_id = idx
    return len(layers)


# ── CalibrationRunner ───────────────────────────────────────────────────────────

class CalibrationRunner:
    def __init__(self, model: nn.Module, percentile: float = 99.9):
        self._model = model
        self._percentile = percentile
        self._layers: list[FakeQuantLinear] = []

    def _ensure_layers(self) -> None:
        if not self._layers:
            self._layers = get_all_fakequant_layers(self._model)

    def start(self) -> None:
        self._ensure_layers()
        for layer in self._layers:
            layer.start_calibration()

    def run_batch(self, inputs) -> None:
        with torch.no_grad():
            self._model(**inputs) if isinstance(inputs, dict) else (
                self._model(*inputs) if isinstance(inputs, (tuple, list)) else self._model(inputs)
            )

    def finish(self) -> None:
        for layer in self._layers:
            layer.finish_calibration()

    def start_online_collection(self) -> None:
        self._ensure_layers()
        for layer in self._layers:
            layer.start_online_collection()

    def finish_online_collection(self) -> None:
        for layer in self._layers:
            layer.finish_online_collection()

    def save_calibration(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self._ensure_layers()
        for layer in self._layers:
            layer.save_calibration(path / f"layer_{layer._layer_id}.pt")

    def load_calibration(self, path: str | Path) -> None:
        self._ensure_layers()
        for layer in self._layers:
            layer.load_calibration(Path(path) / f"layer_{layer._layer_id}.pt")


# ── CalibrationHook ─────────────────────────────────────────────────────────────

class CalibrationHook:
    """
    Collects raw activations from nn.Linear modules via forward-pre hooks.

    NOTE: Call this BEFORE apply_fake_quant(), because apply_fake_quant replaces
    nn.Linear with FakeQuantLinear (which does not forward to the original linear's
    hook). For post-apply collection, use FakeQuantLinear's built-in
    CalibrationRunner.start() / start_online_collection() instead.
    """
    def __init__(self, model: nn.Module):
        self._model = model
        self._activations: dict[str, list[torch.Tensor]] = {}
        self._handles: list = []

    def _make_hook(self, name: str):
        def hook(module, input):
            x = input[0].detach()
            x = x.flatten(0, -2) if x.dim() > 2 else x.detach()
            self._activations.setdefault(name, []).append(x)
        return hook

    def __enter__(self):
        for name, module in self._model.named_modules():
            if isinstance(module, nn.Linear):
                self._handles.append(module.register_forward_pre_hook(self._make_hook(name)))
        return self

    def __exit__(self, *args):
        for h in self._handles:
            h.remove()

    def get_activations(self) -> dict[str, torch.Tensor]:
        return {name: torch.cat(acts, dim=0) for name, acts in self._activations.items()}


# ── Model-level apply / remove ──────────────────────────────────────────────────

def apply_fake_quant(
    model: nn.Module,
    bits_w: int = 4,
    bits_a: int = 4,
    group_size: int = 256,
    percentile_w: float = 99.9,
    percentile_a: float = 99.9,
    quantize_activations: bool = True,
    stochastic_round: bool = False,
    exclude_names: Optional[list[str]] = None,
    exclude_bits: Optional[Sequence[int]] = None,
) -> int:
    if isinstance(model, nn.Linear):
        raise ValueError("apply_fake_quant expects a parent module, not a bare nn.Linear.")
    _validate_bits(bits_w)
    _validate_bits(bits_a)
    exclude_names = exclude_names or []
    exclude_bits = [2] if exclude_bits is None else list(exclude_bits)
    count = 0

    def replace(module: nn.Module, name: str, linear: nn.Linear) -> None:
        nonlocal count
        setattr(module, name, FakeQuantLinear(
            linear,
            group_size=group_size,
            percentile_w=percentile_w,
            percentile_a=percentile_a,
            quantize_activations=quantize_activations,
            stochastic_round=stochastic_round,
            exclude_bits=exclude_bits,
        ))
        count += 1

    def walk(m: nn.Module, prefix: str = "") -> None:
        for name, child in list(m.named_children()):
            full = f"{prefix}.{name}" if prefix else name
            if any(ex in full for ex in exclude_names):
                continue
            # skip already-wrapped layers to prevent double-wrapping
            if isinstance(child, FakeQuantLinear):
                continue
            if isinstance(child, nn.Linear):
                replace(m, name, child)
            elif child._modules:
                walk(child, full)

    walk(model)
    register_all_fakequant_layers(model)
    set_global_bits(w=bits_w, a=bits_a)
    return count


def remove_fake_quant(model: nn.Module) -> None:
    def walk(m: nn.Module) -> None:
        for name, child in list(m.named_children()):
            if isinstance(child, FakeQuantLinear):
                setattr(m, name, child._linear)
            elif child._modules:
                walk(child)
    walk(model)


def count_parameters(model: nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    linear_params = sum(p.numel() for m in model.modules() if isinstance(m, nn.Linear) for p in m.parameters())
    fakequant_count = sum(1 for m in model.modules() if isinstance(m, FakeQuantLinear))
    return {"total": total, "linear_params": linear_params, "fakequant_layers": fakequant_count}


# ── Tests ──────────────────────────────────────────────────────────────────────

def _test():
    # ── Helper ──────────────────────────────────────────────────────────────────

    def test_case(name: str, x: torch.Tensor, bits: int, tol: float = 0.15):
        scales, zps = _calc_asym_qparams_from_minmax(x.min(), x.max(), bits)
        q_min, q_max = 0, (2 ** bits) - 1
        x_flat = x.flatten(0, -2)
        x_norm = x_flat / scales + zps
        x_q = torch.clamp(torch.round(x_norm), q_min, q_max)
        x_deq = ((x_q - zps) * scales)
        ok = torch.allclose(x_flat, x_deq, atol=tol)
        print(f"{'PASS' if ok else 'FAIL'} | {name} | x={x_flat.tolist()} | deq={x_deq.tolist()} | zp={zps.item():.2f}")

    # ── Float32 activation cases ───────────────────────────────────────────────

    test_case("positive row", torch.tensor([[1.0, 1.5, 2.0]]), 4)
    test_case("negative row", torch.tensor([[-2.0, -1.5, -1.0]]), 4)
    test_case("cross-zero row", torch.tensor([[-2.0, -1.0, 0.0, 1.0, 2.0]]), 4)
    test_case("const positive", torch.tensor([[0.5, 0.5, 0.5]]), 4)
    test_case("const negative", torch.tensor([[-2.0, -2.0, -2.0]]), 4)
    test_case("all zeros", torch.tensor([[0.0, 0.0, 0.0]]), 4)

    # ── FP16 activation edge cases ─────────────────────────────────────────────

    print("\n--- FP16 activation ---")

    x_fp16_zeros = torch.zeros(1, 3, dtype=torch.float16)
    fq = FakeQuantLinear(nn.Linear(3, 3), exclude_bits=[])
    fq._ensure_weight_scales()
    out = fq._quantize_activation(x_fp16_zeros, 4)
    ok = not torch.isnan(out).any() and out.dtype == torch.float16
    print(f"{'PASS' if ok else 'FAIL'} | fp16 all-zeros no NaN | out={out.flatten().tolist()}")

    x_fp16_tiny = torch.tensor([[1e-7, 1e-7, 1e-7]], dtype=torch.float16)
    out = fq._quantize_activation(x_fp16_tiny, 4)
    ok = not torch.isnan(out).any() and out.dtype == torch.float16
    print(f"{'PASS' if ok else 'FAIL'} | fp16 tiny values no NaN | out={out.flatten().tolist()}")

    x_fp16_pos = torch.tensor([[1.0, 1.5, 2.0]], dtype=torch.float16)
    out = fq._quantize_activation(x_fp16_pos, 4)
    vals = out.flatten().tolist()
    distinct = len(set(vals)) > 1
    ok = not torch.isnan(out).any() and out.dtype == torch.float16 and distinct
    print(f"{'PASS' if ok else 'FAIL'} | fp16 positive row no NaN, distinct={distinct} | out={vals}")

    x_fp16_neg = torch.tensor([[-2.0, -1.5, -1.0]], dtype=torch.float16)
    out = fq._quantize_activation(x_fp16_neg, 4)
    vals = out.flatten().tolist()
    distinct = len(set(vals)) > 1
    ok = not torch.isnan(out).any() and out.dtype == torch.float16 and distinct
    print(f"{'PASS' if ok else 'FAIL'} | fp16 negative row no NaN, distinct={distinct} | out={vals}")

    # ── FP16 FakeQuantLinear forward ──────────────────────────────────────────

    print("\n--- FP16 forward ---")

    linear_h = nn.Linear(4, 4).half()
    fq = FakeQuantLinear(linear_h, exclude_bits=[])
    x_h = torch.zeros(2, 4, dtype=torch.float16)
    fq._ensure_weight_scales()
    out = fq.forward(x_h)
    ok = out.dtype == torch.float16 and not torch.isnan(out).any()
    print(f"{'PASS' if ok else 'FAIL'} | fp16 forward dtype=fp16, no NaN")

    # ── Bit switching ─────────────────────────────────────────────────────────

    print("\n--- Bit switch ---")
    linear = nn.Linear(8, 8)
    fq = FakeQuantLinear(linear, group_size=4, exclude_bits=[])
    x = torch.randn(2, 8)
    fq._ensure_weight_scales()
    out4 = fq.forward(x)
    fq.set_a_bits(8)
    out8 = fq.forward(x)
    ok = out4.shape == out8.shape == x.shape
    print(f"{'PASS' if ok else 'FAIL'} | bit switch W4A4/W4A8 shape check")

    # ── Double apply ──────────────────────────────────────────────────────────

    print("\n--- Double apply ---")
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
    count1 = apply_fake_quant(model, bits_w=4, bits_a=4, exclude_bits=[])
    count2 = apply_fake_quant(model, bits_w=4, bits_a=4, exclude_bits=[])
    inner = model[0]
    ok = isinstance(inner, FakeQuantLinear) and not isinstance(inner._linear, FakeQuantLinear)
    print(f"{'PASS' if ok else 'FAIL'} | double apply no nesting (counts={count1},{count2})")

    # ── Per-layer override clear ───────────────────────────────────────────────

    print("\n--- Override clear ---")
    model2 = nn.Sequential(nn.Linear(4, 4))
    apply_fake_quant(model2, bits_w=4, bits_a=4, exclude_bits=[])
    layer0 = model2[0]
    layer0.set_w_bits(8)
    ok = layer0.current_w_bits == 8
    print(f"{'PASS' if ok else 'FAIL'} | per-layer override set (w_bits={layer0.current_w_bits})")
    apply_fake_quant(model2, bits_w=4, bits_a=4, exclude_bits=[])
    ok2 = model2[0].current_w_bits == 4
    print(f"{'PASS' if ok2 else 'FAIL'} | per-layer override cleared (w_bits={model2[0].current_w_bits})")

    # ── Global bit validation ─────────────────────────────────────────────────

    print("\n--- Bit validation ---")
    try:
        set_global_bits_w(99)
        print("FAIL | invalid bits_w accepted")
    except ValueError:
        print("PASS | invalid bits_w rejected")
    try:
        apply_fake_quant(nn.Linear(4, 4), bits_w=99, bits_a=4, exclude_bits=[])
        print("FAIL | invalid bits_w in apply_fake_quant accepted")
    except ValueError:
        print("PASS | invalid bits_w in apply_fake_quant rejected")

    # ── set_global_bits backward compat ───────────────────────────────────────

    print("\n--- set_global_bits compat ---")
    set_global_bits(4)
    ok = _current_w_bits == 4 and _current_a_bits == 4
    print(f"{'PASS' if ok else 'FAIL'} | set_global_bits(4) -> W4A4")

    set_global_bits(w=4, a=8)
    ok = _current_w_bits == 4 and _current_a_bits == 8
    print(f"{'PASS' if ok else 'FAIL'} | set_global_bits(w=4, a=8) -> W4A8")

    try:
        set_global_bits(4, w=4, a=8)
        print("FAIL | set_global_bits(4, w=..., a=...) accepted")
    except ValueError:
        print("PASS | set_global_bits(4, w=..., a=...) rejected")

    # ── exclude_bits cannot contain 4 ─────────────────────────────────────────

    print("\n--- exclude_bits validation ---")
    try:
        FakeQuantLinear(nn.Linear(4, 4), exclude_bits=[4])
        print("FAIL | exclude_bits=[4] accepted")
    except ValueError:
        print("PASS | exclude_bits=[4] rejected")
    try:
        FakeQuantLinear(nn.Linear(4, 4), exclude_bits=[2, 4])
        print("FAIL | exclude_bits=[2,4] accepted")
    except ValueError:
        print("PASS | exclude_bits=[2,4] rejected")
    try:
        FakeQuantLinear(nn.Linear(4, 4), exclude_bits=[99])
        print("FAIL | exclude_bits=[99] accepted")
    except ValueError:
        print("PASS | exclude_bits=[99] rejected")

    # ── exclude_bits=[] works correctly ──────────────────────────────────────

    print("\n--- exclude_bits=[] ---")
    fq = FakeQuantLinear(nn.Linear(4, 4), exclude_bits=[])
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    fq._ensure_weight_scales()
    out = fq._quantize_activation(x, 4)
    ok = out.dtype == x.dtype and out.shape == x.shape
    print(f"{'PASS' if ok else 'FAIL'} | exclude_bits=[] forward works | shape={out.shape}")

    # ── 1D input shape preserved ────────────────────────────────────────────

    print("\n--- 1D input ---")
    linear = nn.Linear(4, 3)
    fq = FakeQuantLinear(linear, exclude_bits=[])
    x_1d = torch.tensor([1.0, 2.0, 3.0, 4.0])
    fq._ensure_weight_scales()
    out_1d = fq.forward(x_1d)
    ok = out_1d.shape == torch.Size([3])
    print(f"{'PASS' if ok else 'FAIL'} | 1D input shape preserved | out={out_1d.shape}")

    # ── online_collection after calibration collects stats ────────────────────

    print("\n--- online_collection after calibration ---")
    linear = nn.Linear(4, 4)
    fq = FakeQuantLinear(linear, exclude_bits=[])
    # first run calibration to populate cache
    fq.start_calibration()
    fq.forward(torch.randn(2, 4))
    fq.finish_calibration()
    assert len(fq._x_qparams_cache) > 0, "calibration should populate cache"
    # now start online collection -- it must clear the cache and collect new stats
    fq.start_online_collection()
    assert len(fq._x_qparams_cache) == 0, "start_online_collection should clear cache"
    fq.forward(torch.randn(3, 4))
    fq.forward(torch.randn(3, 4))
    fq.finish_online_collection()
    ok = len(fq._x_qparams_cache) > 0
    print(f"{'PASS' if ok else 'FAIL'} | online_collection after calibration collected stats | cache_size={len(fq._x_qparams_cache)}")

    # ── apply_fake_quant exclude_bits=[] propagates empty set ─────────────────

    print("\n--- apply_fake_quant exclude_bits=[] ---")
    model3 = nn.Sequential(nn.Linear(4, 4))
    apply_fake_quant(model3, bits_w=4, bits_a=4, exclude_bits=[])
    layer = model3[0]
    ok = layer._exclude_bits == set()
    print(f"{'PASS' if ok else 'FAIL'} | apply_fake_quant exclude_bits=[] -> empty set | got={layer._exclude_bits}")

    # ── calibration with 1D input ────────────────────────────────────────────

    print("\n--- calibration 1D input ---")
    linear4 = nn.Linear(4, 4)
    fq = FakeQuantLinear(linear4, exclude_bits=[])
    fq.start_calibration()
    try:
        fq.forward(torch.tensor([1.0, 2.0, 3.0, 4.0]))
        fq.finish_calibration()
        ok = len(fq._x_qparams_cache) > 0
        print(f"{'PASS' if ok else 'FAIL'} | calibration 1D input no crash, cache populated")
    except Exception as e:
        print(f"FAIL | calibration 1D input crashed: {e}")

    # ── effective bits reflect fallback ───────────────────────────────────────

    print("\n--- effective bits fallback ---")
    fq = FakeQuantLinear(nn.Linear(4, 4), exclude_bits=[2])
    fq.set_bits(2)  # 2 is excluded, should fall back to 4
    ok_w = fq.effective_w_bits == 4
    ok_a = fq.effective_a_bits == 4
    print(f"{'PASS' if ok_w else 'FAIL'} | effective_w_bits with exclude=[2] and set_bits(2) = {fq.effective_w_bits}")
    print(f"{'PASS' if ok_a else 'FAIL'} | effective_a_bits with exclude=[2] and set_bits(2) = {fq.effective_a_bits}")

    # ── apply_fake_quant rejects bare nn.Linear ───────────────────────────────

    print("\n--- apply_fake_quant bare Linear rejection ---")
    try:
        apply_fake_quant(nn.Linear(4, 4))
        print("FAIL | bare nn.Linear accepted")
    except ValueError:
        print("PASS | bare nn.Linear rejected")

    print("\nAll tests done.")


if __name__ == "__main__":
    _test()
