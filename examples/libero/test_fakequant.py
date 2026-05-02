"""
test_fakequant.py -- Standalone benchmark comparing old vs new FakeQuant.

Tests:
  1. Per-Channel max-abs  vs  Per-Group percentile  (weight quantization)
  2. No act-quant  vs  Per-Token percentile act-quant  (activation quantization)
  3. With/without outliers in weight data (robustness test)
  4. Per-bit-width (4, 8, 16) comparison

Metrics:
  - MSE vs FP32 baseline
  - Relative L2 error
  - Cosine similarity
  - Max absolute error
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Literal

torch.manual_seed(42)
np.random.seed(42)


# ============================================================================
# Old FakeQuant (from experiment.py)
# ============================================================================

def compute_scale_per_channel_old(weight: torch.Tensor, bits: int) -> torch.Tensor:
    levels = (2 ** (bits - 1)) - 1
    max_val = torch.max(torch.abs(weight), dim=1, keepdim=True)[0]
    scale = torch.where(max_val < 1e-9, torch.ones_like(max_val), max_val / levels)
    return scale


class OldFakeQuantLinear(nn.Module):
    def __init__(self, linear: nn.Linear, bits: int, quantize_activations: bool = False):
        super().__init__()
        self._linear = linear
        self._bits = bits
        self._quantize_activations = quantize_activations
        self.register_buffer("_w_scale", compute_scale_per_channel_old(linear.weight.data, bits))

    @property
    def in_features(self): return self._linear.in_features
    @property
    def out_features(self): return self._linear.out_features
    @property
    def bias(self): return self._linear.bias
    @property
    def weight(self): return self._linear.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._bits >= 16:
            return self._linear(x)

        levels = (2 ** (self._bits - 1)) - 1
        q_min = -levels - 1
        q_max = levels

        w = self._linear.weight
        w_q = torch.round(w / self._w_scale)
        w_q = torch.clamp(w_q, q_min, q_max)
        w_deq = w_q * self._w_scale

        if self._quantize_activations:
            x_max = torch.max(torch.abs(x), dim=-1, keepdim=True)[0]
            x_scale = torch.where(x_max < 1e-9, torch.ones_like(x_max), x_max / levels)
            x_q = torch.round(x / x_scale)
            x_q = torch.clamp(x_q, q_min, q_max)
            x_deq = x_q * x_scale
        else:
            x_deq = x

        return F.linear(x_deq, w_deq, self._linear.bias)


def apply_old_fake_quant(model: nn.Module, bits: int, quantize_activations: bool = False):
    def _rec(module, prefix=""):
        for name in list(module._modules.keys()):
            child = module._modules[name]
            full = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear):
                module._modules[name] = OldFakeQuantLinear(child, bits, quantize_activations)
            elif hasattr(child, "_modules") and len(child._modules) > 0:
                _rec(child, full)
    _rec(model)


# ============================================================================
# Import new FakeQuant
# ============================================================================

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fakequant import (
    FakeQuantLinear as NewFakeQuantLinear,
    apply_fake_quant as apply_new_fake_quant,
    set_global_bits,
)


# ============================================================================
# Test model (simulates a few transformer layers)
# ============================================================================

class DummyTransformerBlock(nn.Module):
    """Simulates one transformer block: attention QKV + O + FFN."""

    def __init__(self, hidden: int, intermediate: int):
        super().__init__()
        self.q_proj = nn.Linear(hidden, hidden)
        self.k_proj = nn.Linear(hidden, hidden)
        self.v_proj = nn.Linear(hidden, hidden)
        self.o_proj = nn.Linear(hidden, hidden)
        self.gate_proj = nn.Linear(hidden, intermediate)
        self.up_proj = nn.Linear(hidden, intermediate)
        self.down_proj = nn.Linear(intermediate, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        # Simplified attention
        attn = torch.softmax(q @ k.transpose(-2, -1) / (x.shape[-1] ** 0.5), dim=-1)
        x = attn @ v
        x = self.o_proj(x)
        ff = self.gate_proj(x) * F.gelu(self.up_proj(x))
        x = x + self.down_proj(ff)
        return x


class DummyModel(nn.Module):
    def __init__(self, hidden: int = 256, intermediate: int = 1024, num_layers: int = 3):
        super().__init__()
        self.embed = nn.Linear(128, hidden)
        self.blocks = nn.ModuleList([
            DummyTransformerBlock(hidden, intermediate) for _ in range(num_layers)
        ])
        self.head = nn.Linear(hidden, 32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


# ============================================================================
# Metrics
# ============================================================================

def compute_metrics(output: torch.Tensor, baseline: torch.Tensor) -> dict[str, float]:
    mse = F.mse_loss(output, baseline).item()
    max_abs_err = (output - baseline).abs().max().item()
    rel_l2 = (output - baseline).norm() / baseline.norm().clamp(min=1e-8)
    rel_l2 = rel_l2.item()
    cos_sim = F.cosine_similarity(
        output.flatten(), baseline.flatten(), dim=0
    ).item()
    return {
        "mse": mse,
        "max_abs_err": max_abs_err,
        "rel_l2": rel_l2.item() if hasattr(rel_l2, 'item') else rel_l2,
        "cosine_sim": cos_sim,
    }


# ============================================================================
# Test 1: Weight quantization only (W4 vs W4 - new)
# ============================================================================

def test_weight_quant_only(hidden=256, intermediate=1024, num_layers=3,
                            bits=4, group_size=128):
    """Compare old vs new weight quantization (no activation quantization)."""

    # Generate random inputs
    torch.manual_seed(123)
    inputs = [torch.randn(2, 128) for _ in range(8)]

    configs = [
        ("BF16 (baseline)", None, False),
        ("Old FakeQuant W4 (max-abs, per-channel)", "old", False),
        ("New FakeQuant W4 (percentile, per-group)", "new", False),
    ]

    results = {}
    baselines = {}

    for label, quant_type, _ in configs:
        # Create fresh model for each test
        model = DummyModel(hidden, intermediate, num_layers)
        model.eval()

        if quant_type == "old":
            apply_old_fake_quant(model, bits=bits, quantize_activations=False)
            label = f"Old FakeQuant W{bits} (max-abs, per-channel)"
        elif quant_type == "new":
            apply_new_fake_quant(model, bits=bits, group_size=group_size,
                                 quantize_activations=False)
            label = f"New FakeQuant W{bits} (percentile, per-group, g={group_size})"

        if quant_type is None:
            label = f"BF16 (baseline)"

        with torch.no_grad():
            outputs = []
            for inp in inputs:
                out = model(inp)
                outputs.append(out)
            if quant_type is None:
                baselines["w_only"] = outputs
            results[label] = outputs

    # Compute metrics relative to baseline
    metrics = {}
    for label, outputs in results.items():
        if "baseline" in label.lower():
            continue
        m = compute_metrics(
            torch.cat(outputs, dim=0),
            torch.cat(baselines["w_only"], dim=0)
        )
        metrics[label] = m

    return metrics


# ============================================================================
# Test 2: Weight + Activation quantization (W4A4 vs W4A4 - new)
# ============================================================================

def test_full_quant(hidden=256, intermediate=1024, num_layers=3,
                    bits=4, group_size=128):
    """Compare old vs new with both weight AND activation quantization."""

    torch.manual_seed(123)
    inputs = [torch.randn(2, 128) for _ in range(8)]

    configs = [
        ("BF16 (baseline)", None, False),
        ("Old FakeQuant W4A4 (max-abs)", "old", True),
        ("New FakeQuant W4A4 (percentile, per-group)", "new", True),
    ]

    results = {}
    baselines = {}

    for label, quant_type, quant_act in configs:
        model = DummyModel(hidden, intermediate, num_layers)
        model.eval()

        if quant_type == "old":
            apply_old_fake_quant(model, bits=bits, quantize_activations=quant_act)
            label = f"Old FakeQuant W{bits}A{bits} (max-abs)"
        elif quant_type == "new":
            apply_new_fake_quant(model, bits=bits, group_size=group_size,
                                 quantize_activations=quant_act)
            label = f"New FakeQuant W{bits}A{bits} (percentile, per-group)"

        if quant_type is None:
            label = f"BF16 (baseline)"

        with torch.no_grad():
            outputs = []
            for inp in inputs:
                out = model(inp)
                outputs.append(out)
            if quant_type is None:
                baselines["full"] = outputs
            results[label] = outputs

    metrics = {}
    for label, outputs in results.items():
        if "baseline" in label.lower():
            continue
        m = compute_metrics(
            torch.cat(outputs, dim=0),
            torch.cat(baselines["full"], dim=0)
        )
        metrics[label] = m

    return metrics


# ============================================================================
# Test 3: Outlier robustness (show percentile is better when outliers present)
# ============================================================================

def test_outlier_robustness(hidden=256, intermediate=1024, outlier_ratio=0.005):
    """Inject outliers into a weight matrix and compare robustness."""

    torch.manual_seed(456)
    linear = nn.Linear(hidden, intermediate)
    inp = torch.randn(4, hidden)

    # Inject outliers
    w_data = linear.weight.data.clone()
    num_outliers = int(outlier_ratio * w_data.numel())
    outlier_indices = torch.randint(0, w_data.numel(), (num_outliers,))
    w_data.flatten()[outlier_indices] *= 20.0  # Huge outliers
    linear.weight.data = w_data

    # Baseline
    with torch.no_grad():
        baseline_out = F.linear(inp, linear.weight, linear.bias)

    # Old quant
    old_layer = OldFakeQuantLinear(linear, bits=4, quantize_activations=False)
    with torch.no_grad():
        old_out = old_layer(inp)

    # New quant
    new_layer = NewFakeQuantLinear(linear, bits=4, group_size=128, percentile=99.9,
                                    quantize_activations=False)
    with torch.no_grad():
        new_out = new_layer(inp)

    m_old = compute_metrics(old_out, baseline_out)
    m_new = compute_metrics(new_out, baseline_out)

    return {"Old (max-abs)": m_old, "New (percentile)": m_new}


# ============================================================================
# Test 4: Vary bit widths
# ============================================================================

def test_bitwidths(hidden=256, intermediate=1024, bits_list=[2, 4, 8]):
    """Compare old vs new across different bit widths."""

    torch.manual_seed(789)
    inputs = [torch.randn(2, 128) for _ in range(8)]

    metrics = {}

    for bits in bits_list:
        for label, apply_fn in [
            ("old", lambda m: apply_old_fake_quant(m, bits=bits, quantize_activations=False)),
            ("new", lambda m: apply_new_fake_quant(m, bits=bits, group_size=128,
                                                   quantize_activations=False)),
        ]:
            model = DummyModel(hidden, intermediate, num_layers=2)
            model.eval()

            # Baseline
            with torch.no_grad():
                baseline_outs = [model(inp.clone()) for inp in inputs]
                baseline = torch.cat(baseline_outs, dim=0)

            # Apply quant
            apply_fn(model)

            with torch.no_grad():
                quant_outs = [model(inp.clone()) for inp in inputs]
                quant = torch.cat(quant_outs, dim=0)

            m = compute_metrics(quant, baseline)
            key = f"W{bits} - {'Old' if label == 'old' else 'New'}"
            metrics[key] = m

    return metrics


# ============================================================================
# Test 5: Ablation on group_size
# ============================================================================

def test_group_size(hidden=256, intermediate=1024, group_sizes=[32, 64, 128, 256, 512]):
    """Ablation study: how does group_size affect accuracy?"""

    torch.manual_seed(789)
    inputs = [torch.randn(2, 128) for _ in range(8)]

    model = DummyModel(hidden, intermediate, num_layers=2)
    model.eval()

    with torch.no_grad():
        baseline = torch.cat([model(inp.clone()) for inp in inputs], dim=0)

    metrics = {}
    for gs in group_sizes:
        # Fresh model each time
        m = DummyModel(hidden, intermediate, num_layers=2)
        m.eval()
        apply_new_fake_quant(m, bits=4, group_size=gs, quantize_activations=False)

        with torch.no_grad():
            out = torch.cat([m(inp.clone()) for inp in inputs], dim=0)
        metrics[f"g={gs}"] = compute_metrics(out, baseline)

    return metrics


# ============================================================================
# Main
# ============================================================================

def print_metrics_table(metrics: dict, title: str):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")
    print(f"{'Method':<45} {'MSE':>10} {'MaxAbsErr':>10} {'RelL2':>8} {'CosSim':>8}")
    print(f"{'-'*80}")
    for label, m in metrics.items():
        print(f"{label:<45} {m['mse']:>10.6f} {m['max_abs_err']:>10.6f} "
              f"{m['rel_l2']:>8.4f} {m['cosine_sim']:>8.4f}")


def main():
    print("\n" + "="*80)
    print("  FakeQuant Benchmark: Old vs New (Improved)")
    print("="*80)

    print("\n[Test 1] Weight quantization only (W4)")
    r1 = test_weight_quant_only(bits=4)
    print_metrics_table(r1, "Weight-only quantization (W4)")

    print("\n[Test 2] Full W4A4 quantization")
    r2 = test_full_quant(bits=4)
    print_metrics_table(r2, "W4A4 quantization")

    print("\n[Test 3] Outlier robustness (0.5% outlier injection)")
    r3 = test_outlier_robustness()
    print_metrics_table(r3, "Outlier robustness")

    print("\n[Test 4] Bit-width ablation")
    r4 = test_bitwidths(bits_list=[2, 4, 8])
    print_metrics_table(r4, "Bit-width comparison")

    print("\n[Test 5] Group size ablation (W4)")
    r5 = test_group_size(group_sizes=[32, 64, 128, 256, 512])
    print_metrics_table(r5, "Group size ablation")

    # Summary
    print("\n" + "="*80)
    print("  Summary")
    print("="*80)
    w4_old = r1.get("Old FakeQuant W4 (max-abs, per-channel)", {})
    w4_new = r1.get("New FakeQuant W4 (percentile, per-group, g=128)", {})
    if w4_old and w4_new:
        print(f"\nW4 weight-only:")
        print(f"  Old: MSE={w4_old['mse']:.6f}, RelL2={w4_old['rel_l2']:.4f}")
        print(f"  New: MSE={w4_new['mse']:.6f}, RelL2={w4_new['rel_l2']:.4f}")
        print(f"  Improvement: MSE {w4_old['mse']/w4_new['mse']:.1f}x better")

    w4a4_old = r2.get("Old FakeQuant W4A4 (max-abs)", {})
    w4a4_new = r2.get("New FakeQuant W4A4 (percentile, per-group)", {})
    if w4a4_old and w4a4_new:
        print(f"\nW4A4 full quantization:")
        print(f"  Old: MSE={w4a4_old['mse']:.6f}, RelL2={w4a4_old['rel_l2']:.4f}")
        print(f"  New: MSE={w4a4_new['mse']:.6f}, RelL2={w4a4_new['rel_l2']:.4f}")
        print(f"  Improvement: MSE {w4a4_old['mse']/w4a4_new['mse']:.1f}x better")

    print("\n✅ Test complete.")


if __name__ == "__main__":
    main()
