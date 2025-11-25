import torch
import torch.nn as nn
import numpy as np
from spline import coeff2curve, curve2coeff, Cox_De_Boor

class KAN_Layer(nn.Module):
    """
    Enhanced KAN Layer with pruning and grid extension capabilities.
    
    Key improvements:
    - Activity tracking for pruning
    - Grid extension for adaptive refinement
    - Regularization for sparsity
    - OPTIMIZED: Vectorized forward pass for batch processing
    """
    def __init__(self, in_dim, out_dim, deg=3, grid_size=5, grid_range=[-1, 1], device='cpu'):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.deg = deg
        self.grid_size = grid_size
        self.grid_range = grid_range
        self.device = device
        
        # Create initial uniform knot vector
        self.knot_vec = self._create_knot_vector(grid_size, grid_range, deg)
        self.n_basis = len(self.knot_vec) - deg - 1
        
        # Learnable spline coefficients: [out_dim, in_dim, n_basis]
        self.coeff = nn.Parameter(
            torch.empty(out_dim, in_dim, self.n_basis, device=device)
        )
        nn.init.xavier_uniform_(self.coeff)
        
        # Activity tracking for pruning (non-trainable)
        self.register_buffer('edge_scores', torch.ones(out_dim, in_dim, device=device))
        self.register_buffer('forward_count', torch.tensor(0, device=device))
        
        # Mask for pruned connections
        self.register_buffer('mask', torch.ones(out_dim, in_dim, device=device))
        
    def _create_knot_vector(self, grid_size, grid_range, deg):
        """Create uniform B-spline knot vector with proper padding."""
        # Interior knots
        interior_knots = torch.linspace(grid_range[0], grid_range[1], 
                                       grid_size, device=self.device)
        # Pad with boundary knots (deg times on each side)
        knot_vec = torch.cat([
            torch.full((deg,), grid_range[0], device=self.device),
            interior_knots,
            torch.full((deg,), grid_range[1], device=self.device)
        ])
        return knot_vec
    
    def forward(self, u, track_activity=True):
        """
        OPTIMIZED forward pass with vectorized batch processing.
        
        Args:
            u: [batch_size, in_dim]
            track_activity: whether to update edge activity scores
        
        Returns:
            [batch_size, out_dim]
        """
        batch_size = u.shape[0]
        
        # Precompute all basis function values for all inputs
        # Shape: [batch_size, in_dim, n_basis]
        all_basis = torch.zeros(batch_size, self.in_dim, self.n_basis, device=self.device)
        
        for j in range(self.in_dim):
            u_j = u[:, j]  # [batch_size]
            for k in range(self.n_basis):
                all_basis[:, j, k] = Cox_De_Boor(u_j, k, self.deg, self.knot_vec, self.device)
        
        # Apply mask to coefficients
        masked_coeff = self.coeff * self.mask.unsqueeze(-1)  # [out_dim, in_dim, n_basis]
        
        # Compute all spline outputs at once using einsum
        # masked_coeff: [out_dim, in_dim, n_basis]
        # all_basis: [batch_size, in_dim, n_basis]
        # result: [batch_size, out_dim]
        # n = batch, o = out_dim, i = in_dim, k = n_basis
        out = torch.einsum('oik,nik->no', masked_coeff, all_basis)
        
        # Track activity
        if track_activity and self.training:
            # Compute per-edge outputs: [batch_size, out_dim, in_dim]
            edge_outputs = torch.einsum('oik,nik->noi', self.coeff, all_basis)
            # Get maximum activation per edge: [out_dim, in_dim]
            max_activations = edge_outputs.abs().max(dim=0)[0]
            
            # Update activity scores with exponential moving average
            self.forward_count += 1
            alpha = 0.9  # EMA decay factor
            self.edge_scores = alpha * self.edge_scores + (1 - alpha) * max_activations
        
        return out
    
    def prune(self, threshold=0.01):
        """
        Prune edges with low activity scores.
        
        Args:
            threshold: activity threshold below which edges are pruned
        
        Returns:
            Number of edges pruned
        """
        # Normalize scores to [0, 1]
        normalized_scores = self.edge_scores / (self.edge_scores.max() + 1e-8)
        
        # Create new mask
        new_mask = (normalized_scores > threshold).float()
        
        # Count pruned edges
        n_pruned = (self.mask - new_mask).sum().item()
        
        # Update mask
        self.mask = new_mask
        
        print(f"Pruned {int(n_pruned)} edges. "
              f"Remaining: {int(self.mask.sum())}/{self.in_dim * self.out_dim}")
        
        return int(n_pruned)
    
    def extend_grid(self, new_grid_size):
        """
        Extend grid resolution for finer approximation.
        
        Args:
            new_grid_size: new number of interior grid points
        """
        if new_grid_size <= self.grid_size:
            print(f"Warning: new_grid_size ({new_grid_size}) <= current ({self.grid_size})")
            return
        
        old_knot_vec = self.knot_vec
        old_n_basis = self.n_basis
        
        # Create new knot vector
        new_knot_vec = self._create_knot_vector(new_grid_size, self.grid_range, self.deg)
        new_n_basis = len(new_knot_vec) - self.deg - 1
        
        # Sample points for interpolation (use old knot positions)
        u_samples = old_knot_vec[self.deg:-self.deg].clone()
        
        # Initialize new coefficients
        new_coeff = torch.zeros(self.out_dim, self.in_dim, new_n_basis, 
                               device=self.device)
        
        with torch.no_grad():
            for i in range(self.out_dim):
                for j in range(self.in_dim):
                    # Evaluate old spline at sample points
                    old_curve = coeff2curve(self.coeff[i, j], u_samples, 
                                          self.deg, old_knot_vec, self.device)
                    
                    # Fit new spline to these values
                    new_coeff[i, j] = curve2coeff(old_curve, u_samples, 
                                                 self.deg, new_knot_vec, self.device)
        
        # Update layer parameters
        self.coeff = nn.Parameter(new_coeff)
        self.knot_vec = new_knot_vec
        self.n_basis = new_n_basis
        self.grid_size = new_grid_size
        
        print(f"Extended grid: {old_n_basis} → {new_n_basis} basis functions")
    
    def get_l1_norm(self):
        """Get L1 norm of active coefficients (for regularization)."""
        active_coeff = self.coeff * self.mask.unsqueeze(-1)
        return torch.abs(active_coeff).sum()
    
    def get_sparsity(self):
        """Get percentage of pruned edges."""
        total = self.in_dim * self.out_dim
        active = self.mask.sum().item()
        return (total - active) / total * 100


class KAN(nn.Module):
    """
    Full KAN network with multiple layers.
    """
    def __init__(self, width, deg=3, grid_size=5, grid_range=[-1, 1], device='cpu', activation='tanh', residual=False):
        """
        Args:
            width: list of layer widths, e.g., [2, 5, 1] for 2→5→1
            deg: B-spline degree
            grid_size: initial grid size
            grid_range: input range for normalization
            device: computation device
        """
        super().__init__()
        self.width = width
        self.depth = len(width) - 1
        self.device = device
        self.activation = activation
        self.residual = residual
        
        # Create layers
        self.layers = nn.ModuleList([
            KAN_Layer(width[i], width[i+1], deg, grid_size, grid_range, device)
            for i in range(self.depth)
        ])
    
    def forward(self, x, track_activity=True):
        """
        Forward pass through all layers.
        
        Args:
            x: [batch_size, input_dim]
        
        Returns:
            [batch_size, output_dim]
        """
        for i, layer in enumerate(self.layers):
            x_in = x
            out = layer(x, track_activity=track_activity)

            # residual connection when input and output dims match and enabled
            if self.residual and out.shape == x_in.shape:
                x = x_in + out
            else:
                x = out

            # apply activation after each layer except last
            if self.activation is not None and i < (self.depth - 1):
                if self.activation == 'tanh':
                    x = torch.tanh(x)
                elif self.activation == 'relu':
                    x = torch.relu(x)
                # add other activations as needed

        return x
    
    def prune(self, threshold=0.01):
        """Prune all layers."""
        total_pruned = 0
        for i, layer in enumerate(self.layers):
            print(f"Layer {i}:")
            n_pruned = layer.prune(threshold)
            total_pruned += n_pruned
        return total_pruned
    
    def extend_grid(self, new_grid_size):
        """Extend grid in all layers."""
        for i, layer in enumerate(self.layers):
            print(f"Layer {i}:")
            layer.extend_grid(new_grid_size)
    
    def get_l1_regularization(self):
        """Get L1 regularization term for all layers."""
        return sum(layer.get_l1_norm() for layer in self.layers)
    
    def get_sparsity_stats(self):
        """Get sparsity statistics for all layers."""
        stats = {}
        for i, layer in enumerate(self.layers):
            stats[f'layer_{i}'] = layer.get_sparsity()
        return stats
    
    def count_parameters(self):
        """Count active parameters."""
        total = 0
        active = 0
        for layer in self.layers:
            layer_params = layer.coeff.numel()
            layer_active = (layer.mask.sum() * layer.n_basis).item()
            total += layer_params
            active += layer_active
        return int(active), int(total)