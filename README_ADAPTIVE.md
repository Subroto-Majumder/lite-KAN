# KAN with Adaptive Pruning and Grid Extension

**Complete implementation of Kolmogorov-Arnold Networks (KAN) with adaptive features following Liu et al., 2024.**

## Overview

This implementation extends the basic KAN layer with two key adaptive mechanisms from the original paper:

1. **Edge-level Pruning** - Removes low-importance spline connections
2. **Adaptive Grid Extension** - Refines knot placement based on curvature

## Installation

Ensure you have the required dependencies:
```bash
conda activate slm  # or your preferred environment
pip install torch numpy matplotlib
```

## Quick Start

### Basic Usage (No Adaptive Features)

```python
from kan import KAN_Layer
import torch

# Create a basic KAN layer
knot_vec = torch.linspace(-3, 3, 20)
layer = KAN_Layer(in_dim=3, out_dim=5, deg=3, knot_vec=knot_vec)

# Forward pass
x = torch.randn(100, 3)
y = layer(x)
```

### With Pruning

```python
# Enable pruning
layer = KAN_Layer(
    in_dim=3,
    out_dim=5,
    deg=3,
    knot_vec=knot_vec,
    enable_pruning=True,
    pruning_threshold=0.01,      # Importance threshold
    pruning_mode="mask",         # "mask" or "structural"
    pruning_frequency=100,       # Prune every 100 steps
    pruning_start_step=50        # Start after 50 steps
)

# Training loop
optimizer = torch.optim.Adam(layer.parameters(), lr=0.01)

for step in range(1000):
    optimizer.zero_grad()
    loss = compute_loss(layer, data)
    loss.backward()
    optimizer.step()
    
    # Adaptive pruning
    layer.step()                 # Increment counter
    info = layer.maybe_prune(data)
    
    if info.get('pruned', False):
        print(f"Step {step}: Pruned {info['n_pruned']} edges")
```

### With Grid Extension

```python
# Enable grid extension
layer = KAN_Layer(
    in_dim=3,
    out_dim=5,
    deg=3,
    knot_vec=torch.linspace(-3, 3, 10),  # Start with coarse grid
    enable_grid_extension=True,
    grid_extension_frequency=200,        # Extend every 200 steps
    grid_extension_start_step=100,       # Start after 100 steps
    curvature_percentile=90.0,           # Top 10% curvature
    max_knots_per_extension=5            # Max knots per extension
)

# Training loop
optimizer = torch.optim.Adam(layer.parameters(), lr=0.01)

for step in range(1000):
    optimizer.zero_grad()
    loss = compute_loss(layer, data)
    loss.backward()
    optimizer.step()
    
    # Adaptive grid extension
    layer.step()
    info = layer.maybe_refine_grid(data)
    
    if info.get('extended', False):
        print(f"Step {step}: Added {info['n_knots_added']} knots")
        # Reinitialize optimizer after parameter shape change
        optimizer = torch.optim.Adam(layer.parameters(), lr=0.01)
```

### Both Pruning and Grid Extension

```python
# Enable both features
layer = KAN_Layer(
    in_dim=3,
    out_dim=5,
    deg=3,
    knot_vec=knot_vec,
    enable_pruning=True,
    pruning_threshold=0.01,
    pruning_frequency=100,
    enable_grid_extension=True,
    grid_extension_frequency=200
)

# Training loop handles both
for step in range(1000):
    optimizer.zero_grad()
    loss = compute_loss(layer, data)
    loss.backward()
    optimizer.step()
    
    layer.step()
    
    # Check pruning
    prune_info = layer.maybe_prune(data)
    
    # Check grid extension
    extend_info = layer.maybe_refine_grid(data)
    
    # Reinitialize optimizer if grid was extended
    if extend_info.get('extended', False):
        optimizer = torch.optim.Adam(layer.parameters(), lr=0.01)
```

## Parameters

### Core Parameters

- `in_dim` (int): Input dimension
- `out_dim` (int): Output dimension  
- `deg` (int): B-spline degree (typically 3)
- `knot_vec` (list/tensor): Initial knot vector
- `device` (str): Computation device ('cpu' or 'cuda')

### Pruning Parameters

- `enable_pruning` (bool): Enable adaptive pruning (default: False)
- `pruning_threshold` (float): Importance threshold for pruning (default: 0.01)
- `pruning_mode` (str): Pruning mode - "mask" (soft) or "structural" (hard) (default: "mask")
- `pruning_frequency` (int): Steps between pruning checks (default: 100)
- `pruning_start_step` (int): Step to start pruning (default: 0)

### Grid Extension Parameters

- `enable_grid_extension` (bool): Enable adaptive grid refinement (default: False)
- `grid_extension_frequency` (int): Steps between extension checks (default: 200)
- `grid_extension_start_step` (int): Step to start extending (default: 100)
- `curvature_percentile` (float): Curvature threshold percentile (default: 90.0)
- `max_knots_per_extension` (int): Max knots to add per extension (default: 5)

### Statistics Parameters

- `collect_stats` (bool): Collect training statistics (default: False)
- `stats_n_samples` (int): Samples for statistics (default: 1000)

## API Reference

### Methods

#### `forward(u)`
Standard forward pass.
- **Args**: `u` - input tensor [batch_size, in_dim]
- **Returns**: output tensor [batch_size, out_dim]

#### `step()`
Increment training step counter. Call after each optimizer step.

#### `maybe_prune(input_batch=None)`
Check if pruning should occur and perform it.
- **Args**: `input_batch` - optional batch for importance computation
- **Returns**: dict with pruning information

#### `maybe_refine_grid(input_samples=None)`
Check if grid extension should occur and perform it.
- **Args**: `input_samples` - optional samples for curvature estimation
- **Returns**: dict with extension information

#### `compute_importance(input_batch)`
Compute edge importance scores.
- **Args**: `input_batch` - tensor [batch_size, in_dim]
- **Returns**: importance scores [out_dim, in_dim]

#### `prune_edges(importance=None, input_batch=None)`
Manually trigger pruning.
- **Args**: precomputed importance or input batch
- **Returns**: dict with pruning statistics

#### `extend_grid(input_samples=None, curvature=None)`
Manually trigger grid extension.
- **Args**: samples or precomputed curvature
- **Returns**: dict with extension statistics

#### `get_stats()`
Get current layer statistics.
- **Returns**: dict with comprehensive layer information

## Files

- `kan.py` - Main KAN_Layer implementation with adaptive features
- `spline.py` - B-spline utilities and knot insertion algorithms
- `adaptive_utils.py` - Helper functions for pruning and grid extension
- `test_adaptive.py` - Comprehensive test suite
- `test_batched.py` - Original batched forward pass test

## Algorithm Details

### Pruning

Following the KAN paper, edge importance is computed as:

```
importance[i,j] = ||f_ij(x_batch)||_2
```

Where `f_ij` is the spline function for edge (input_j → output_i).

**Pruning Modes:**
- **Mask (soft)**: Zeros out coefficients, preserves tensor shape
- **Structural (hard)**: Removes connections (currently implemented as hard masking)

### Grid Extension

Adaptive knot refinement uses curvature estimation:

```
curvature[i] = |f[i+1] - 2*f[i] + f[i-1]|
```

**Algorithm:**
1. Evaluate splines on sample grid
2. Compute second finite differences (curvature)
3. Identify high-curvature regions (top percentile)
4. Insert knots using Boehm's algorithm
5. Update all coefficients to maintain curve shape

**Knot Insertion:**
Uses the standard Cox-de Boor knot insertion algorithm ensuring the spline curve remains unchanged after refinement.

## Testing

Run the comprehensive test suite:

```bash
conda activate slm
python test_adaptive.py
```

This will:
1. Test basic pruning functionality
2. Test basic grid extension
3. Test training with automatic pruning
4. Test training with automatic grid extension  
5. Test combined pruning + grid extension
6. Generate visualization plots

Output: `adaptive_kan_results.png` with training curves and statistics.

## Examples

### Example 1: Learning sin(x) with Pruning

```python
import torch
from kan import KAN_Layer

# Training data
xs = torch.linspace(-3, 3, 100).unsqueeze(1)
targets = torch.sin(xs.squeeze())

# Create layer
knot_vec = torch.linspace(-5, 5, 30)
layer = KAN_Layer(
    in_dim=1, out_dim=1, deg=3, knot_vec=knot_vec,
    enable_pruning=True,
    pruning_threshold=0.01,
    pruning_frequency=20
)

# Train
optimizer = torch.optim.Adam(layer.parameters(), lr=0.01)

for step in range(200):
    optimizer.zero_grad()
    preds = layer(xs).squeeze()
    loss = ((preds - targets) ** 2).mean()
    loss.backward()
    optimizer.step()
    
    layer.step()
    info = layer.maybe_prune(xs)
    
    if info.get('pruned'):
        print(f"Sparsity: {info['sparsity']:.2%}")
```

### Example 2: Learning Complex Function with Grid Extension

```python
# Training data: x^2 * sin(3*x)
xs = torch.linspace(-2, 2, 150).unsqueeze(1)
targets = (xs.squeeze() ** 2) * torch.sin(3 * xs.squeeze())

# Start with coarse grid
knot_vec = torch.linspace(-3, 3, 8)
layer = KAN_Layer(
    in_dim=1, out_dim=1, deg=3, knot_vec=knot_vec,
    enable_grid_extension=True,
    grid_extension_frequency=50,
    max_knots_per_extension=3
)

optimizer = torch.optim.Adam(layer.parameters(), lr=0.01)

for step in range(200):
    optimizer.zero_grad()
    preds = layer(xs).squeeze()
    loss = ((preds - targets) ** 2).mean()
    loss.backward()
    optimizer.step()
    
    layer.step()
    info = layer.maybe_refine_grid(xs)
    
    if info.get('extended'):
        print(f"Basis functions: {info['new_n_basis']}")
        optimizer = torch.optim.Adam(layer.parameters(), lr=0.01)
```

## Implementation Notes

### Backward Compatibility

All existing KAN code continues to work without modifications. Adaptive features are opt-in via constructor parameters.

### Optimizer Reinitialization

After grid extension, parameter shapes change. You must reinitialize the optimizer:

```python
if extend_info.get('extended', False):
    optimizer = torch.optim.Adam(layer.parameters(), lr=0.01)
```

### Differentiability

The knot insertion algorithm preserves the spline curve, so gradients flow correctly through grid extension operations.

### Performance

- Pruning has minimal overhead (computes importance periodically)
- Grid extension is more expensive (requires knot insertion) but occurs less frequently
- Both operations are performed with `torch.no_grad()` where appropriate

## Citation

If you use this implementation, please cite the original KAN paper:

```bibtex
@article{liu2024kan,
  title={KAN: Kolmogorov-Arnold Networks},
  author={Liu, Ziming and Wang, Yixuan and Vaidya, Sachin and Ruehle, Fabian and Halverson, James and Solja{\v{c}}i{\'c}, Marin and Hou, Thomas Y and Tegmark, Max},
  journal={arXiv preprint arXiv:2404.19756},
  year={2024}
}
```

## License

This implementation follows the license of the original repository.

## Contributing

This is a research implementation. Feel free to extend and improve!

Key areas for future work:
- Full structural pruning with graph restructuring
- Distributed training support
- More sophisticated knot insertion strategies
- Curvature-based per-edge refinement

---

**Author**: Implementation based on Liu et al., 2024  
**Date**: November 2025  
**Status**: Fully functional, all tests passing ✓
