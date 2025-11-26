import torch
import torch.nn as nn
import numpy as np
from spline import coeff2curve, curve2coeff, insert_knot, insert_multiple_knots
from adaptive_utils import (
    compute_edge_importance,
    compute_pruning_mask,
    apply_soft_pruning,
    apply_structural_pruning,
    estimate_curvature,
    identify_refinement_intervals,
    collect_spline_statistics,
    should_perform_operation
)
from typing import Optional, Dict, Tuple

class KAN_Layer(torch.nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        deg,
        knot_vec,
        device='cpu',
        # Pruning parameters
        enable_pruning: bool = False,
        pruning_threshold: float = 0.01,
        pruning_mode: str = "mask",  # "mask" or "structural"
        pruning_frequency: int = 100,
        pruning_start_step: int = 0,
        # Grid extension parameters
        enable_grid_extension: bool = False,
        grid_extension_frequency: int = 200,
        grid_extension_start_step: int = 100,
        curvature_percentile: float = 90.0,
        max_knots_per_extension: Optional[int] = 5,
        # Statistics collection
        collect_stats: bool = False,
        stats_n_samples: int = 1000
    ):
        """
        Kolmogorov-Arnold Network Layer with adaptive pruning and grid extension.
        
        This implementation follows Liu et al., 2024 (KAN paper) for:
        1. Edge-level pruning based on spline importance
        2. Adaptive knot refinement based on curvature
        
        Args:
            in_dim: input dimension
            out_dim: output dimension
            deg: B-spline degree
            knot_vec: initial knot vector (list or tensor)
            device: computation device
            
            # Pruning parameters:
            enable_pruning: whether to enable adaptive pruning
            pruning_threshold: importance threshold below which edges are pruned
            pruning_mode: "mask" (soft pruning) or "structural" (hard removal)
            pruning_frequency: how often to check for pruning (in steps)
            pruning_start_step: first step at which pruning can occur
            
            # Grid extension parameters:
            enable_grid_extension: whether to enable adaptive knot refinement
            grid_extension_frequency: how often to check for grid extension
            grid_extension_start_step: first step at which extension can occur
            curvature_percentile: percentile threshold for high curvature
            max_knots_per_extension: max number of knots to add per extension
            
            # Statistics:
            collect_stats: whether to collect training statistics
            stats_n_samples: number of samples for statistics collection
        
        Example:
            >>> # Basic usage without adaptive features
            >>> layer = KAN_Layer(in_dim=3, out_dim=5, deg=3, knot_vec=knots)
            
            >>> # With pruning enabled
            >>> layer = KAN_Layer(
            ...     in_dim=3, out_dim=5, deg=3, knot_vec=knots,
            ...     enable_pruning=True,
            ...     pruning_threshold=0.01,
            ...     pruning_frequency=100
            ... )
            
            >>> # With both pruning and grid extension
            >>> layer = KAN_Layer(
            ...     in_dim=3, out_dim=5, deg=3, knot_vec=knots,
            ...     enable_pruning=True,
            ...     enable_grid_extension=True
            ... )
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.deg = deg
        # Properly convert knot_vec to tensor
        if isinstance(knot_vec, torch.Tensor):
            self.knot_vec = knot_vec.clone().detach().to(device=device, dtype=torch.float32)
        else:
            self.knot_vec = torch.tensor(knot_vec, device=device, dtype=torch.float32)
        self.device = device

        # number of basis functions per spline
        self.n_basis = len(knot_vec) - deg - 1

        # Each (input j → node i) has its own spline coefficients
        # Shape: [out_dim, in_dim, n_basis]
        self.coeff = torch.nn.Parameter(
            torch.empty(out_dim, in_dim, self.n_basis, device=device)
        )
        torch.nn.init.xavier_uniform_(self.coeff)
        
        # Base function weights and spline weights (KAN paper: phi(x) = w_b * b(x) + w_s * spline(x))
        # where b(x) = silu(x) is the base function
        self.base_weight = torch.nn.Parameter(
            torch.ones(out_dim, in_dim, device=device)
        )
        self.spline_weight = torch.nn.Parameter(
            torch.ones(out_dim, in_dim, device=device)
        )
        
        # ===== Pruning state =====
        self.enable_pruning = enable_pruning
        self.pruning_threshold = pruning_threshold
        self.pruning_mode = pruning_mode
        self.pruning_frequency = pruning_frequency
        self.pruning_start_step = pruning_start_step
        
        # Pruning mask: 1 = active edge, 0 = pruned edge
        # Registered as buffer so it's saved with model but not trained
        self.register_buffer(
            'pruning_mask',
            torch.ones(out_dim, in_dim, device=device)
        )
        
        # ===== Grid extension state =====
        self.enable_grid_extension = enable_grid_extension
        self.grid_extension_frequency = grid_extension_frequency
        self.grid_extension_start_step = grid_extension_start_step
        self.curvature_percentile = curvature_percentile
        self.max_knots_per_extension = max_knots_per_extension
        
        # Track number of grid extensions performed
        self.n_grid_extensions = 0
        
        # ===== Statistics =====
        self.collect_stats = collect_stats
        self.stats_n_samples = stats_n_samples
        self.training_step = 0  # Track training steps for scheduling
        
        # Store latest statistics
        self.latest_stats = None


    def forward(self, u):
        """
        Forward pass with full batching support and pruning mask application.
        
        Args:
            u: input tensor of shape [batch_size, in_dim]
        
        Returns:
            output tensor of shape [batch_size, out_dim]
        """
        batch_size = u.shape[0]
        
        # Initialize output: [batch_size, out_dim]
        out = torch.zeros(batch_size, self.out_dim, device=self.device)
        
        # Process all output nodes and input dimensions in parallel
        for i in range(self.out_dim):
            for j in range(self.in_dim):
                # Check if this edge is active (not pruned)
                if self.enable_pruning and self.pruning_mask[i, j] == 0:
                    continue  # Skip pruned edges
                
                # Get coefficients for this (input_j → output_i) connection
                # coeff_ij shape: [n_basis]
                coeff_ij = self.coeff[i, j]
                
                # Apply pruning mask if enabled
                if self.enable_pruning:
                    coeff_ij = coeff_ij * self.pruning_mask[i, j]
                
                # Get input values for all samples in the batch for input dimension j
                # u[:, j] shape: [batch_size]
                u_j = u[:, j]
                
                # Compute base function: b(x) = silu(x)
                # base_ij shape: [batch_size]
                base_ij = torch.nn.functional.silu(u_j)
                
                # Evaluate spline for all batch samples at once
                # spline_ij shape: [batch_size]
                spline_ij = coeff2curve(coeff_ij, u_j, self.deg, self.knot_vec, self.device)
                
                # KAN paper: phi(x) = w_b * b(x) + w_s * spline(x)
                # Accumulate to output
                out[:, i] += self.base_weight[i, j] * base_ij + self.spline_weight[i, j] * spline_ij
        
        return out
    
    def compute_importance(self, input_batch: torch.Tensor) -> torch.Tensor:
        """
        Compute edge importance scores for pruning.
        
        Following KAN paper: importance[i,j] = ||f_ij(x_batch)||_2
        
        Args:
            input_batch: tensor of shape [batch_size, in_dim] or [batch_size]
                        If 1D, will be expanded to match in_dim
        
        Returns:
            importance: tensor of shape [out_dim, in_dim]
        
        Example:
            >>> layer = KAN_Layer(3, 5, 3, knots, enable_pruning=True)
            >>> batch = torch.randn(100, 3)
            >>> importance = layer.compute_importance(batch)
            >>> print(importance.shape)  # [5, 3]
        """
        # Handle 1D input batch
        if input_batch.ndim == 1:
            input_batch = input_batch.unsqueeze(1)
        
        # If input has only 1 dimension but layer expects more, expand
        if input_batch.shape[1] == 1 and self.in_dim > 1:
            input_batch = input_batch.expand(-1, self.in_dim)
        
        batch_size = input_batch.shape[0]
        
        # Evaluate each spline separately
        # Shape: [batch_size, out_dim, in_dim]
        spline_outputs = torch.zeros(
            batch_size, self.out_dim, self.in_dim, device=self.device
        )
        
        with torch.no_grad():
            for i in range(self.out_dim):
                for j in range(self.in_dim):
                    coeff_ij = self.coeff[i, j]
                    u_j = input_batch[:, j]
                    spline_outputs[:, i, j] = coeff2curve(
                        coeff_ij, u_j, self.deg, self.knot_vec, self.device
                    )
        
        # Compute L2 norm across batch dimension
        importance = compute_edge_importance(spline_outputs)
        
        return importance
    
    def prune_edges(
        self,
        importance: Optional[torch.Tensor] = None,
        input_batch: Optional[torch.Tensor] = None
    ) -> Dict[str, any]:
        """
        Perform edge pruning based on importance scores.
        
        Either provide precomputed importance or an input_batch to compute it.
        
        Args:
            importance: optional precomputed importance scores [out_dim, in_dim]
            input_batch: optional input batch to compute importance from
        
        Returns:
            info: dictionary with pruning statistics
        
        Example:
            >>> layer = KAN_Layer(3, 5, 3, knots, enable_pruning=True)
            >>> batch = torch.randn(100, 3)
            >>> info = layer.prune_edges(input_batch=batch)
            >>> print(f"Pruned {info['n_pruned']} edges")
        """
        if not self.enable_pruning:
            return {'pruned': False, 'reason': 'pruning not enabled'}
        
        # Compute importance if not provided
        if importance is None:
            if input_batch is None:
                return {'pruned': False, 'reason': 'no importance or batch provided'}
            importance = self.compute_importance(input_batch)
        
        # Get current number of active edges
        n_active_before = self.pruning_mask.sum().item()
        
        # Compute new pruning mask
        new_mask = compute_pruning_mask(
            importance,
            self.pruning_threshold,
            self.pruning_mask
        )
        
        n_active_after = new_mask.sum().item()
        n_pruned = n_active_before - n_active_after
        
        if n_pruned == 0:
            return {
                'pruned': False,
                'n_active': n_active_after,
                'n_pruned': 0,
                'reason': 'no edges below threshold'
            }
        
        # Apply pruning based on mode
        if self.pruning_mode == "mask":
            # Soft pruning: just update mask
            self.pruning_mask = new_mask
            
            # Zero out coefficients of pruned edges
            with torch.no_grad():
                mask_expanded = self.pruning_mask.unsqueeze(-1)
                self.coeff.data = self.coeff.data * mask_expanded
            
            return {
                'pruned': True,
                'mode': 'mask',
                'n_active': n_active_after,
                'n_pruned': n_pruned,
                'sparsity': 1.0 - (n_active_after / (self.out_dim * self.in_dim))
            }
        
        elif self.pruning_mode == "structural":
            # Structural pruning: remove connections
            # Note: Full structural pruning would require graph restructuring
            # For now, we do hard masking
            self.pruning_mask = new_mask
            
            with torch.no_grad():
                mask_expanded = self.pruning_mask.unsqueeze(-1)
                self.coeff.data = self.coeff.data * mask_expanded
            
            return {
                'pruned': True,
                'mode': 'structural',
                'n_active': n_active_after,
                'n_pruned': n_pruned,
                'sparsity': 1.0 - (n_active_after / (self.out_dim * self.in_dim)),
                'note': 'structural pruning implemented as hard masking'
            }
        
        else:
            raise ValueError(f"Unknown pruning mode: {self.pruning_mode}")
    
    def maybe_prune(self, input_batch: Optional[torch.Tensor] = None) -> Dict[str, any]:
        """
        Check if pruning should be performed at this step and do it if needed.
        
        This is the main entry point for scheduled pruning during training.
        
        Args:
            input_batch: optional batch for importance computation
                        If None, will generate synthetic samples
        
        Returns:
            info: pruning information dictionary
        
        Example:
            >>> layer = KAN_Layer(3, 5, 3, knots, enable_pruning=True, pruning_frequency=100)
            >>> for step in range(1000):
            ...     # ... training code ...
            ...     info = layer.maybe_prune(batch)
            ...     if info.get('pruned', False):
            ...         print(f"Pruned at step {step}")
        """
        if not self.enable_pruning:
            return {'pruned': False, 'reason': 'pruning not enabled'}
        
        # Check if we should prune at this step
        if not should_perform_operation(
            self.training_step,
            self.pruning_frequency,
            self.pruning_start_step
        ):
            return {'pruned': False, 'reason': 'not scheduled for this step'}
        
        # Generate batch if not provided
        if input_batch is None:
            # Generate random samples in a reasonable range
            input_batch = torch.randn(
                self.stats_n_samples, self.in_dim, device=self.device
            )
        
        # Perform pruning
        info = self.prune_edges(input_batch=input_batch)
        info['step'] = self.training_step
        
        return info
    
    def estimate_edge_curvature(
        self,
        input_samples: torch.Tensor
    ) -> torch.Tensor:
        """
        Estimate curvature for each edge's spline function.
        
        Args:
            input_samples: tensor of shape [n_samples, in_dim] or [n_samples]
                          If 1D, will be expanded to match in_dim
        
        Returns:
            curvature: tensor of shape [out_dim, in_dim, n_samples-2]
        
        Example:
            >>> layer = KAN_Layer(3, 5, 3, knots)
            >>> samples = torch.linspace(-1, 1, 100).unsqueeze(1).expand(-1, 3)
            >>> curv = layer.estimate_edge_curvature(samples)
            >>> print(curv.shape)  # [5, 3, 98]
        """
        # Handle 1D input samples
        if input_samples.ndim == 1:
            input_samples = input_samples.unsqueeze(1)
        
        # If input has only 1 dimension but layer expects more, expand
        if input_samples.shape[1] == 1 and self.in_dim > 1:
            input_samples = input_samples.expand(-1, self.in_dim)
        
        n_samples = input_samples.shape[0]
        
        # Evaluate all splines on the samples
        # Shape: [n_samples, out_dim, in_dim]
        spline_values = torch.zeros(
            n_samples, self.out_dim, self.in_dim, device=self.device
        )
        
        with torch.no_grad():
            for i in range(self.out_dim):
                for j in range(self.in_dim):
                    # Skip pruned edges
                    if self.enable_pruning and self.pruning_mask[i, j] == 0:
                        continue
                    
                    coeff_ij = self.coeff[i, j]
                    u_j = input_samples[:, j]
                    spline_values[:, i, j] = coeff2curve(
                        coeff_ij, u_j, self.deg, self.knot_vec, self.device
                    )
        
        # Compute curvature for each edge
        # Shape: [out_dim, in_dim, n_samples-2]
        curvature = torch.zeros(
            self.out_dim, self.in_dim, n_samples - 2, device=self.device
        )
        
        for i in range(self.out_dim):
            for j in range(self.in_dim):
                curv_ij = estimate_curvature(spline_values[:, i, j])
                curvature[i, j] = curv_ij
        
        return curvature
    
    def extend_grid(
        self,
        input_samples: Optional[torch.Tensor] = None,
        curvature: Optional[torch.Tensor] = None
    ) -> Dict[str, any]:
        """
        Perform adaptive grid extension by inserting knots in high-curvature regions.
        
        This implements the adaptive refinement strategy from the KAN paper.
        
        Args:
            input_samples: optional samples for curvature estimation [n_samples, in_dim]
            curvature: optional precomputed curvature [out_dim, in_dim, n_samples-2]
        
        Returns:
            info: dictionary with grid extension statistics
        
        Example:
            >>> layer = KAN_Layer(3, 5, 3, knots, enable_grid_extension=True)
            >>> samples = torch.linspace(-1, 1, 100).unsqueeze(1).expand(-1, 3)
            >>> info = layer.extend_grid(input_samples=samples)
            >>> print(f"Added {info['n_knots_added']} knots")
        """
        if not self.enable_grid_extension:
            return {'extended': False, 'reason': 'grid extension not enabled'}
        
        # Generate samples if not provided
        if input_samples is None:
            # Create uniform grid samples across knot range
            knot_min = self.knot_vec.min().item()
            knot_max = self.knot_vec.max().item()
            
            # Create samples for each input dimension
            u_samples_1d = torch.linspace(
                knot_min, knot_max, self.stats_n_samples, device=self.device
            )
            input_samples = u_samples_1d.unsqueeze(1).expand(-1, self.in_dim)
        
        # Compute curvature if not provided
        if curvature is None:
            curvature = self.estimate_edge_curvature(input_samples)
        
        # Aggregate curvature across all edges to find global high-curvature regions
        # Average curvature across output and input dimensions
        # Shape: [n_samples-2]
        avg_curvature = curvature.mean(dim=(0, 1))
        
        # Identify where to insert knots
        # Use samples from middle dimension (all should have same range ideally)
        u_samples_1d = input_samples[:, 0]
        
        insertion_points = identify_refinement_intervals(
            avg_curvature,
            u_samples_1d,
            percentile=self.curvature_percentile,
            max_insertions=self.max_knots_per_extension
        )
        
        if len(insertion_points) == 0:
            return {
                'extended': False,
                'reason': 'no high-curvature regions found',
                'n_knots_added': 0
            }
        
        # Insert knots and update all spline coefficients
        old_knot_vec = self.knot_vec
        old_n_basis = self.n_basis
        
        # Use the helper function to insert multiple knots
        # This handles the sequential insertion properly
        with torch.no_grad():
            # Start with current state
            current_knot_vec = old_knot_vec
            current_coeff = self.coeff.data.clone()
            
            # Insert knots one by one, updating all edges
            for new_knot in insertion_points:
                # Get the new knot vector after insertion
                new_knot_vec, _ = insert_knot(
                    current_knot_vec,
                    new_knot.item(),
                    self.deg,
                    torch.zeros(len(current_knot_vec) - self.deg - 1, device=self.device),
                    self.device
                )
                
                # Update coefficients for all edges
                new_n_basis_temp = len(new_knot_vec) - self.deg - 1
                new_coeff_temp = torch.zeros(
                    self.out_dim, self.in_dim, new_n_basis_temp, device=self.device
                )
                
                for i in range(self.out_dim):
                    for j in range(self.in_dim):
                        _, new_coeff_temp[i, j] = insert_knot(
                            current_knot_vec,
                            new_knot.item(),
                            self.deg,
                            current_coeff[i, j],
                            self.device
                        )
                
                # Update for next iteration
                current_knot_vec = new_knot_vec
                current_coeff = new_coeff_temp
            
            updated_coeff = current_coeff
            new_knot_vec = current_knot_vec
        
        # Calculate new basis dimension
        new_n_basis = len(new_knot_vec) - self.deg - 1
        
        # Update layer parameters
        self.knot_vec = new_knot_vec
        self.n_basis = new_n_basis
        
        # Replace parameter with new size
        del self.coeff
        self.coeff = torch.nn.Parameter(updated_coeff)
        
        # Update pruning mask if needed
        if self.enable_pruning:
            # Mask stays the same shape [out_dim, in_dim]
            pass
        
        self.n_grid_extensions += 1
        
        return {
            'extended': True,
            'n_knots_added': len(insertion_points),
            'old_n_basis': old_n_basis,
            'new_n_basis': new_n_basis,
            'old_n_knots': len(old_knot_vec),
            'new_n_knots': len(new_knot_vec),
            'insertion_points': insertion_points.tolist(),
            'total_extensions': self.n_grid_extensions
        }
    
    def maybe_refine_grid(
        self,
        input_samples: Optional[torch.Tensor] = None
    ) -> Dict[str, any]:
        """
        Check if grid refinement should be performed at this step and do it if needed.
        
        This is the main entry point for scheduled grid extension during training.
        
        Args:
            input_samples: optional samples for curvature estimation
        
        Returns:
            info: grid extension information dictionary
        
        Example:
            >>> layer = KAN_Layer(
            ...     3, 5, 3, knots,
            ...     enable_grid_extension=True,
            ...     grid_extension_frequency=200
            ... )
            >>> for step in range(1000):
            ...     # ... training code ...
            ...     info = layer.maybe_refine_grid()
            ...     if info.get('extended', False):
            ...         print(f"Extended grid at step {step}")
        """
        if not self.enable_grid_extension:
            return {'extended': False, 'reason': 'grid extension not enabled'}
        
        # Check if we should extend at this step
        if not should_perform_operation(
            self.training_step,
            self.grid_extension_frequency,
            self.grid_extension_start_step
        ):
            return {'extended': False, 'reason': 'not scheduled for this step'}
        
        # Perform grid extension
        info = self.extend_grid(input_samples=input_samples)
        info['step'] = self.training_step
        
        return info
    
    def step(self):
        """
        Increment training step counter.
        
        Call this after each optimizer step to keep track of training progress.
        
        Example:
            >>> layer = KAN_Layer(3, 5, 3, knots, enable_pruning=True)
            >>> optimizer = torch.optim.Adam(layer.parameters())
            >>> for epoch in range(100):
            ...     loss = compute_loss(layer, data)
            ...     loss.backward()
            ...     optimizer.step()
            ...     layer.step()  # Increment counter
            ...     layer.maybe_prune()
            ...     layer.maybe_refine_grid()
        """
        self.training_step += 1
    
    def get_stats(self) -> Dict[str, any]:
        """
        Get current layer statistics.
        
        Returns:
            stats: dictionary with layer information
        
        Example:
            >>> layer = KAN_Layer(3, 5, 3, knots, enable_pruning=True)
            >>> stats = layer.get_stats()
            >>> print(f"Active edges: {stats['n_active_edges']}")
            >>> print(f"Sparsity: {stats['sparsity']:.2%}")
        """
        n_total_edges = self.out_dim * self.in_dim
        n_active_edges = self.pruning_mask.sum().item() if self.enable_pruning else n_total_edges
        
        stats = {
            'in_dim': self.in_dim,
            'out_dim': self.out_dim,
            'deg': self.deg,
            'n_basis': self.n_basis,
            'n_knots': len(self.knot_vec),
            'n_total_edges': n_total_edges,
            'n_active_edges': int(n_active_edges),
            'sparsity': 1.0 - (n_active_edges / n_total_edges),
            'n_parameters': self.coeff.numel(),
            'training_step': self.training_step,
            'n_grid_extensions': self.n_grid_extensions,
            'enable_pruning': self.enable_pruning,
            'enable_grid_extension': self.enable_grid_extension
        }
        
        return stats    


