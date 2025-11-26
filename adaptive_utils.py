"""
Adaptive utilities for KAN: Pruning and Grid Extension
Implementation based on Liu et al., 2024 (Kolmogorov-Arnold Networks)

This module provides:
1. Edge-level spline pruning based on importance scores
2. Adaptive knot refinement based on curvature estimation
"""

import torch
import numpy as np
from typing import Tuple, Optional, Dict


def compute_edge_importance(spline_outputs: torch.Tensor) -> torch.Tensor:
    """
    Compute importance score for each edge as L-infinity norm (max absolute value) of spline outputs.
    
    This follows the KAN paper's definition:
        importance[i,j] = max_x |f_ij(x)|
    
    where f_ij is the spline function for edge (input_j → output_i).
    
    Args:
        spline_outputs: tensor of shape [batch_size, out_dim, in_dim]
                       containing evaluated spline values for all edges
    
    Returns:
        importance: tensor of shape [out_dim, in_dim] containing max absolute values
    
    Example:
        >>> outputs = torch.randn(100, 5, 3)  # 100 samples, 5 outputs, 3 inputs
        >>> importance = compute_edge_importance(outputs)
        >>> print(importance.shape)  # [5, 3]
    """
    # Compute L-infinity norm (max absolute value) across batch dimension (dim=0)
    # Result shape: [out_dim, in_dim]
    importance = torch.max(torch.abs(spline_outputs), dim=0)[0]
    return importance


def compute_pruning_mask(
    importance: torch.Tensor,
    threshold: float,
    current_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute binary mask for pruning based on importance threshold.
    
    Edges with importance < threshold are marked for pruning (mask = 0).
    
    Args:
        importance: tensor of shape [out_dim, in_dim] with importance scores
        threshold: scalar threshold value (absolute or percentile-based)
        current_mask: optional existing mask to combine with new pruning decisions
    
    Returns:
        mask: binary tensor of shape [out_dim, in_dim]
              1 = keep edge, 0 = prune edge
    
    Example:
        >>> importance = torch.tensor([[0.5, 1.2], [0.1, 0.8]])
        >>> mask = compute_pruning_mask(importance, threshold=0.3)
        >>> print(mask)
        tensor([[1, 1],
                [0, 1]])
    """
    # Create binary mask: 1 where importance >= threshold, 0 otherwise
    new_mask = (importance >= threshold).float()
    
    # If existing mask provided, combine them (keep AND logic)
    if current_mask is not None:
        new_mask = new_mask * current_mask
    
    return new_mask


def apply_soft_pruning(
    coeff: torch.Tensor,
    mask: torch.Tensor
) -> torch.Tensor:
    """
    Apply soft pruning by zeroing out coefficients of pruned edges.
    
    This is the "mask" mode: coefficients are zeroed but tensor shape is preserved.
    
    Args:
        coeff: tensor of shape [out_dim, in_dim, n_basis]
        mask: binary tensor of shape [out_dim, in_dim]
    
    Returns:
        pruned_coeff: tensor of same shape with masked edges set to zero
    
    Example:
        >>> coeff = torch.randn(2, 3, 5)
        >>> mask = torch.tensor([[1, 0, 1], [1, 1, 0]])
        >>> pruned = apply_soft_pruning(coeff, mask)
        >>> print((pruned[0, 1, :] == 0).all())  # Edge (0,1) should be zero
        True
    """
    # Expand mask to match coefficient dimensions: [out_dim, in_dim, 1]
    mask_expanded = mask.unsqueeze(-1)
    
    # Element-wise multiplication zeros out pruned edges
    pruned_coeff = coeff * mask_expanded
    
    return pruned_coeff


def apply_structural_pruning(
    coeff: torch.Tensor,
    mask: torch.Tensor,
    in_dim: int,
    out_dim: int
) -> Tuple[torch.Tensor, int, int, Dict]:
    """
    Apply structural pruning by removing pruned edges from the tensor.
    
    This is the "structural" mode: the tensor is actually reshaped to remove
    pruned connections. This is more complex and requires tracking which
    connections remain.
    
    Note: For full structural pruning, we would need to handle:
    - Removing entire input dimensions if all edges pruned
    - Removing entire output dimensions if all edges pruned
    - Maintaining connection indices for gradient flow
    
    For simplicity, this implementation keeps dimensions but marks pruned edges.
    A full implementation would require graph-level restructuring.
    
    Args:
        coeff: tensor of shape [out_dim, in_dim, n_basis]
        mask: binary tensor of shape [out_dim, in_dim]
        in_dim: current input dimension
        out_dim: current output dimension
    
    Returns:
        new_coeff: pruned coefficient tensor
        new_in_dim: potentially reduced input dimension
        new_out_dim: potentially reduced output dimension
        connection_map: dict mapping old indices to new indices
    
    Example:
        >>> coeff = torch.randn(3, 4, 5)
        >>> mask = torch.ones(3, 4)
        >>> mask[0, 1] = 0  # Prune one edge
        >>> new_coeff, new_in, new_out, map_info = apply_structural_pruning(
        ...     coeff, mask, 4, 3
        ... )
    """
    # For now, we implement structural pruning as hard zeroing
    # A full implementation would require reindexing and dimension reduction
    
    # Check which dimensions can be fully removed
    edges_per_input = mask.sum(dim=0)  # Sum over output dimension
    edges_per_output = mask.sum(dim=1)  # Sum over input dimension
    
    # Keep dimensions that have at least one active edge
    active_inputs = edges_per_input > 0
    active_outputs = edges_per_output > 0
    
    # For structural pruning, we zero out and track connectivity
    pruned_coeff = coeff.clone()
    mask_expanded = mask.unsqueeze(-1)
    pruned_coeff = pruned_coeff * mask_expanded
    
    # Create connection map
    connection_map = {
        'active_inputs': active_inputs,
        'active_outputs': active_outputs,
        'mask': mask,
        'n_active_edges': mask.sum().item()
    }
    
    return pruned_coeff, in_dim, out_dim, connection_map


def estimate_curvature(
    spline_values: torch.Tensor,
    mode: str = 'second_derivative'
) -> torch.Tensor:
    """
    Estimate curvature of spline functions for adaptive grid refinement.
    
    Following the KAN paper, curvature is estimated using second differences:
        curvature[i] = |f[i+1] - 2*f[i] + f[i-1]|
    
    High curvature regions indicate where more knots are needed.
    
    Args:
        spline_values: tensor of shape [n_samples, ...] containing spline evaluations
                      on a grid of input points
        mode: curvature estimation method
              'second_derivative' - use second finite differences
    
    Returns:
        curvature: tensor of shape [n_samples-2, ...] with curvature estimates
    
    Example:
        >>> # Evaluate spline on grid
        >>> u_grid = torch.linspace(-1, 1, 100)
        >>> spline_vals = some_spline_function(u_grid)  # shape [100]
        >>> curv = estimate_curvature(spline_vals.unsqueeze(0))
        >>> print(curv.shape)  # [98]
    """
    if mode == 'second_derivative':
        # Compute second finite difference
        # f[2:] - 2*f[1:-1] + f[:-2]
        curvature = torch.abs(
            spline_values[2:] - 2 * spline_values[1:-1] + spline_values[:-2]
        )
    else:
        raise ValueError(f"Unknown curvature mode: {mode}")
    
    return curvature


def identify_refinement_intervals(
    curvature: torch.Tensor,
    u_samples: torch.Tensor,
    percentile: float = 90.0,
    max_insertions: Optional[int] = None
) -> torch.Tensor:
    """
    Identify intervals where new knots should be inserted based on curvature.
    
    Args:
        curvature: tensor of shape [n_samples-2] with curvature values
        u_samples: tensor of shape [n_samples] with input sample points
        percentile: percentile threshold for high curvature (e.g., 90 = top 10%)
        max_insertions: optional maximum number of knots to insert
    
    Returns:
        insertion_points: tensor of input values where knots should be inserted
    
    Example:
        >>> u_grid = torch.linspace(-1, 1, 100)
        >>> curv = torch.rand(98)
        >>> insertion_pts = identify_refinement_intervals(
        ...     curv, u_grid, percentile=95
        ... )
    """
    # Compute threshold based on percentile
    threshold = torch.quantile(curvature, percentile / 100.0)
    
    # Find indices where curvature exceeds threshold
    # Note: curvature[i] corresponds to interval between u_samples[i+1] and u_samples[i+2]
    high_curv_indices = torch.where(curvature > threshold)[0]
    
    if len(high_curv_indices) == 0:
        return torch.tensor([], device=curvature.device)
    
    # Limit number of insertions if specified
    if max_insertions is not None and len(high_curv_indices) > max_insertions:
        # Sort by curvature value and take top max_insertions
        sorted_indices = torch.argsort(curvature, descending=True)
        high_curv_indices = sorted_indices[:max_insertions]
        high_curv_indices = torch.sort(high_curv_indices)[0]
    
    # Insert knots at midpoint of high-curvature intervals
    # curvature[i] corresponds to interval [u_samples[i+1], u_samples[i+2]]
    insertion_points = []
    for idx in high_curv_indices:
        # Midpoint of interval
        midpoint = (u_samples[idx + 1] + u_samples[idx + 2]) / 2.0
        insertion_points.append(midpoint)
    
    return torch.tensor(insertion_points, device=curvature.device)


def collect_spline_statistics(
    layer,
    data_loader: Optional[torch.utils.data.DataLoader] = None,
    n_samples: int = 1000,
    input_range: Tuple[float, float] = (-1.0, 1.0)
) -> Dict[str, torch.Tensor]:
    """
    Collect statistics about spline activations for adaptive operations.
    
    This function evaluates all splines on a set of input samples and returns
    statistics useful for both pruning (importance) and grid extension (curvature).
    
    Args:
        layer: KAN_Layer instance
        data_loader: optional DataLoader to use real data samples
        n_samples: number of samples to use if generating synthetic data
        input_range: range for synthetic uniform samples per dimension
    
    Returns:
        stats: dictionary containing:
            - 'importance': [out_dim, in_dim] importance scores
            - 'curvature': [out_dim, in_dim, n_samples-2] curvature per edge
            - 'sample_points': [n_samples, in_dim] sample input points
            - 'spline_outputs': [n_samples, out_dim, in_dim] spline evaluations
    
    Example:
        >>> layer = KAN_Layer(in_dim=3, out_dim=5, deg=3, knot_vec=knots)
        >>> stats = collect_spline_statistics(layer, n_samples=500)
        >>> print(stats['importance'].shape)  # [5, 3]
    """
    layer.eval()  # Set to evaluation mode
    
    with torch.no_grad():
        if data_loader is not None:
            # Use real data from loader
            samples = []
            for batch in data_loader:
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]
                samples.append(batch)
                if len(samples) * batch.shape[0] >= n_samples:
                    break
            sample_points = torch.cat(samples, dim=0)[:n_samples]
        else:
            # Generate uniform samples in input range
            sample_points = torch.rand(
                n_samples, layer.in_dim, device=layer.device
            )
            # Scale to desired range
            range_min, range_max = input_range
            sample_points = sample_points * (range_max - range_min) + range_min
        
        # Evaluate all splines individually
        # Shape: [n_samples, out_dim, in_dim]
        spline_outputs = torch.zeros(
            n_samples, layer.out_dim, layer.in_dim, device=layer.device
        )
        
        from spline import coeff2curve
        
        for i in range(layer.out_dim):
            for j in range(layer.in_dim):
                # Get coefficients for this edge
                coeff_ij = layer.coeff[i, j]
                
                # Evaluate spline on all samples
                # sample_points[:, j] shape: [n_samples]
                spline_outputs[:, i, j] = coeff2curve(
                    coeff_ij,
                    sample_points[:, j],
                    layer.deg,
                    layer.knot_vec,
                    layer.device
                )
        
        # Compute importance: L2 norm across samples
        importance = compute_edge_importance(spline_outputs)
        
        # Compute curvature for each edge
        # Shape: [out_dim, in_dim, n_samples-2]
        curvature = torch.zeros(
            layer.out_dim, layer.in_dim, n_samples - 2, device=layer.device
        )
        
        for i in range(layer.out_dim):
            for j in range(layer.in_dim):
                # Curvature for this edge's spline
                curv_ij = estimate_curvature(spline_outputs[:, i, j])
                curvature[i, j] = curv_ij
        
        stats = {
            'importance': importance,
            'curvature': curvature,
            'sample_points': sample_points,
            'spline_outputs': spline_outputs
        }
    
    return stats


def should_perform_operation(
    step: int,
    frequency: int,
    start_step: int = 0
) -> bool:
    """
    Determine if an adaptive operation should be performed at this step.
    
    Args:
        step: current training step
        frequency: how often to perform operation (e.g., every 100 steps)
        start_step: first step at which operation can occur
    
    Returns:
        should_perform: True if operation should be performed
    
    Example:
        >>> for step in range(1000):
        ...     if should_perform_operation(step, frequency=100):
        ...         print(f"Perform operation at step {step}")
        Perform operation at step 100
        Perform operation at step 200
        ...
    """
    if step < start_step:
        return False
    
    if frequency <= 0:
        return False
    
    return (step - start_step) % frequency == 0
