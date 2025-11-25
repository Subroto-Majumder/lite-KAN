import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

class PoissonPDE:
    """
    2D Poisson equation: ∇²u = f(x,y)
    Domain: [0,1] × [0,1]
    Boundary condition: u = 0 on boundary
    """
    def __init__(self, source_function='sin'):
        """
        Args:
            source_function: type of source term
                'sin': f(x,y) = -2π²sin(πx)sin(πy) 
                       [exact solution: u = sin(πx)sin(πy)]
                'gaussian': f(x,y) = Gaussian-like source
        """
        self.source_type = source_function
        
    def source_term(self, x, y):
        """Compute f(x,y)."""
        if self.source_type == 'sin':
            # For exact solution u = sin(πx)sin(πy)
            return -2 * np.pi**2 * torch.sin(np.pi * x) * torch.sin(np.pi * y)
        elif self.source_type == 'gaussian':
            # Gaussian-like source
            return 10 * torch.exp(-((x-0.5)**2 + (y-0.5)**2) / 0.1)
        else:
            raise ValueError(f"Unknown source type: {self.source_type}")
    
    def exact_solution(self, x, y):
        """Exact solution (only for sin case)."""
        if self.source_type == 'sin':
            return torch.sin(np.pi * x) * torch.sin(np.pi * y)
        else:
            return None
    
    def generate_training_data(self, n_interior=20, n_boundary=20, device='cpu'):
        """
        Generate collocation points for PDE solving.
        
        Args:
            n_interior: points per dimension in interior
            n_boundary: points per boundary edge
        
        Returns:
            interior_points: [n_interior², 2]
            boundary_points: [4*n_boundary, 2]
            source_values: [n_interior²]
        """
        # Interior points (exclude boundaries)
        x_int = torch.linspace(0, 1, n_interior + 2, device=device)[1:-1]
        y_int = torch.linspace(0, 1, n_interior + 2, device=device)[1:-1]
        xx_int, yy_int = torch.meshgrid(x_int, y_int, indexing='ij')
        interior_points = torch.stack([xx_int.flatten(), yy_int.flatten()], dim=1)
        
        # Compute source values
        source_values = self.source_term(interior_points[:, 0], interior_points[:, 1])
        
        # Boundary points
        x_bd = torch.linspace(0, 1, n_boundary, device=device)
        y_bd = torch.linspace(0, 1, n_boundary, device=device)
        
        boundary_points = torch.cat([
            torch.stack([x_bd, torch.zeros_like(x_bd)], dim=1),  # bottom
            torch.stack([x_bd, torch.ones_like(x_bd)], dim=1),   # top
            torch.stack([torch.zeros_like(y_bd), y_bd], dim=1),  # left
            torch.stack([torch.ones_like(y_bd), y_bd], dim=1),   # right
        ], dim=0)
        
        return interior_points, boundary_points, source_values


class PDESolver:
    """
    Physics-Informed Neural Network solver using KAN.
    """
    def __init__(self, model, pde, device='cpu'):
        """
        Args:
            model: KAN network
            pde: PoissonPDE instance
            device: computation device
        """
        self.model = model
        self.pde = pde
        self.device = device
        # Optional learnable basis coefficient for simple analytic basis
        # (helps capture dominant mode like sin(pi x)sin(pi y)). Stored
        # as a Parameter so it can be optimized alongside model params.
        if self.pde.source_type == 'sin':
            # initialize small so model learns correction
            self.basis_coeff = nn.Parameter(torch.tensor(0.0, device=self.device))
        else:
            self.basis_coeff = None
        
    def compute_derivatives(self, u_pred, x, y):
        """
        Compute second derivatives for Laplacian using vectorized differentiation.
        
        Returns:
            u_xx + u_yy (Laplacian) [batch_size, 1]
        """
        # Compute gradients w.r.t x and y simultaneously
        grad_u = torch.autograd.grad(outputs=u_pred.sum(), inputs=[x, y],
                                     create_graph=True, allow_unused=True)
        u_x = grad_u[0]
        u_y = grad_u[1]
        
        # Compute second derivatives
        u_xx = torch.autograd.grad(outputs=u_x.sum(), inputs=x,
                                   create_graph=True, allow_unused=True)[0]
        u_yy = torch.autograd.grad(outputs=u_y.sum(), inputs=y,
                                   create_graph=True, allow_unused=True)[0]
        
        # Handle None gradients
        if u_xx is None:
            u_xx = torch.zeros_like(x)
        if u_yy is None:
            u_yy = torch.zeros_like(y)
            
        return u_xx + u_yy
    
    def pde_loss(self, interior_points, source_values):
        """
        Physics loss: ||∇²u - f||²
        """
        # Normalize interior points to [-1, 1] for KAN input
        x_norm = 2 * interior_points[:, 0:1] - 1  # [0,1] -> [-1,1]
        y_norm = 2 * interior_points[:, 1:2] - 1  # [0,1] -> [-1,1]
        x_norm.requires_grad_(True)
        y_norm.requires_grad_(True)
        xy_norm = torch.cat([x_norm, y_norm], dim=1)
        
        # Predict model contribution u_model
        u_model = self.model(xy_norm, track_activity=True)

        # Compute Laplacian of model component in normalized space, scale by 4 (chain rule)
        laplacian_model_norm = self.compute_derivatives(u_model, x_norm, y_norm)
        laplacian_model = laplacian_model_norm * 4

        # If using analytic sine basis, include its Laplacian contribution
        if self.basis_coeff is not None:
            # basis laplacian: a * (-2*pi^2) * sin(pi x) sin(pi y)
            x_orig = interior_points[:, 0:1]
            y_orig = interior_points[:, 1:2]
            basis_lap = (-2.0 * np.pi**2) * torch.sin(np.pi * x_orig) * torch.sin(np.pi * y_orig)
            laplacian_total = laplacian_model + self.basis_coeff * basis_lap
        else:
            laplacian_total = laplacian_model

        # PDE residual
        residual = laplacian_total - source_values.unsqueeze(1)

        return torch.mean(residual ** 2)
    
    def boundary_loss(self, boundary_points):
        """
        Boundary loss: ||u||² on boundary (Dirichlet BC: u=0)
        """
        # Normalize boundary points to [-1, 1] for KAN input
        boundary_norm = 2 * boundary_points - 1  # [0,1] -> [-1,1]
        u_model = self.model(boundary_norm, track_activity=True)

        # Add analytic basis if present (evaluate on original boundary points)
        if self.basis_coeff is not None:
            x_orig = boundary_points[:, 0:1]
            y_orig = boundary_points[:, 1:2]
            basis_val = torch.sin(np.pi * x_orig) * torch.sin(np.pi * y_orig)
            u_total = u_model + self.basis_coeff * basis_val
        else:
            u_total = u_model

        return torch.mean(u_total ** 2)
    
    def total_loss(self, interior_points, boundary_points, source_values, 
                   lambda_pde=1.0, lambda_bc=1.0, lambda_l1=0.0):
        """
        Total loss with regularization.
        """
        loss_pde = self.pde_loss(interior_points, source_values)
        loss_bc = self.boundary_loss(boundary_points)
        loss_l1 = self.model.get_l1_regularization() if lambda_l1 > 0 else 0.0
        
        total = lambda_pde * loss_pde + lambda_bc * loss_bc + lambda_l1 * loss_l1
        
        return total, {
            'pde': loss_pde.item(),
            'boundary': loss_bc.item(),
            'l1': loss_l1.item() if lambda_l1 > 0 else 0.0,
            'total': total.item()
        }
    
    def train(self, n_interior=20, n_boundary=20, epochs=1000, lr=0.01,
              lambda_pde=1.0, lambda_bc=10.0, lambda_l1=1e-4, 
              prune_interval=None, prune_threshold=0.01,
              grid_extension_schedule=None):
        """
        Train the model to solve PDE.
        
        Args:
            n_interior: interior collocation points per dimension
            n_boundary: boundary points per edge
            epochs: training epochs
            lr: learning rate
            lambda_pde: PDE loss weight
            lambda_bc: boundary loss weight
            lambda_l1: L1 regularization weight
            prune_interval: epochs between pruning (None = no pruning)
            prune_threshold: activity threshold for pruning
            grid_extension_schedule: dict {epoch: new_grid_size}
        """
        # Generate training data
        interior_pts, boundary_pts, source_vals = self.pde.generate_training_data(
            n_interior, n_boundary, self.device
        )

        # If we have an analytic basis coefficient, initialize it by projecting
        # the source onto the basis in a least-squares sense. This gives a
        # strong prior for dominant modes (e.g., sin(pi x)sin(pi y)).
        if self.basis_coeff is not None:
            x_orig = interior_pts[:, 0:1]
            y_orig = interior_pts[:, 1:2]
            basis_lap = (-2.0 * np.pi**2) * torch.sin(np.pi * x_orig) * torch.sin(np.pi * y_orig)
            # flatten to 1D
            b = basis_lap.reshape(-1)
            s = source_vals.reshape(-1)
            denom = (b * b).sum()
            if denom.abs() > 0:
                a_init = (b * s).sum() / denom
                try:
                    with torch.no_grad():
                        self.basis_coeff.data = a_init.to(self.basis_coeff.device)
                except Exception:
                    self.basis_coeff.data = torch.tensor(a_init, device=self.device)
                print(f"Initialized basis_coeff = {self.basis_coeff.item():.6f}")
        
        # Optimizer with weight decay (AdamW) and cosine annealing scheduler
        # Include basis_coeff in optimizer if present
        params = list(self.model.parameters())
        if self.basis_coeff is not None:
            params.append(self.basis_coeff)
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-6)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        
        # Training history
        history = {
            'pde_loss': [],
            'boundary_loss': [],
            'l1_loss': [],
            'total_loss': [],
            'l2_error': []  # if exact solution available
        }
        
        print(f"Starting training for {epochs} epochs...")
        print(f"Interior points: {interior_pts.shape[0]}, Boundary points: {boundary_pts.shape[0]}")
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Compute loss
            total_loss, losses = self.total_loss(
                interior_pts, boundary_pts, source_vals,
                lambda_pde, lambda_bc, lambda_l1
            )
            
            # Backward pass
            total_loss.backward()
            # Gradient clipping to stabilize training
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Learning rate scheduling (step once per epoch)
            scheduler.step()
            
            # Record history
            history['pde_loss'].append(losses['pde'])
            history['boundary_loss'].append(losses['boundary'])
            history['l1_loss'].append(losses['l1'])
            history['total_loss'].append(losses['total'])
            
            # Compute L2 error if exact solution available
            if self.pde.source_type == 'sin':
                with torch.no_grad():
                    # Normalize interior points for model evaluation (consistent with training)
                    x_norm = 2 * interior_pts[:, 0:1] - 1
                    y_norm = 2 * interior_pts[:, 1:2] - 1
                    xy_norm = torch.cat([x_norm, y_norm], dim=1)

                    u_pred = self.model(xy_norm, track_activity=False)
                    # Include analytic basis contribution when reporting L2
                    if self.basis_coeff is not None:
                        u_model = self.model(xy_norm, track_activity=False)
                        u_basis = self.pde.exact_solution(
                            interior_pts[:, 0], interior_pts[:, 1]
                        ).unsqueeze(1)
                        u_pred = u_model + self.basis_coeff * u_basis
                    else:
                        u_pred = self.model(xy_norm, track_activity=False)
                    u_exact = self.pde.exact_solution(
                        interior_pts[:, 0], interior_pts[:, 1]
                    ).unsqueeze(1)
                    l2_error = torch.sqrt(torch.mean((u_pred - u_exact) ** 2))
                    history['l2_error'].append(l2_error.item())
            
            # Pruning
            if prune_interval and (epoch + 1) % prune_interval == 0:
                print(f"\nEpoch {epoch+1}: Pruning...")
                self.model.prune(prune_threshold)
            
            # Grid extension
            if grid_extension_schedule and epoch in grid_extension_schedule:
                new_size = grid_extension_schedule[epoch]
                print(f"\nEpoch {epoch}: Extending grid to {new_size}...")
                self.model.extend_grid(new_size)
                # Re-initialize optimizer with new parameters
                optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-6)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
            
            # Logging
            if (epoch + 1) % 10 == 0 or epoch == 0:
                sparsity = self.model.get_sparsity_stats()
                active, total = self.model.count_parameters()
                current_lr = optimizer.param_groups[0]['lr']
                
                print(f"Epoch {epoch+1}/{epochs}")
                print(f"  Losses - PDE: {losses['pde']:.6f}, BC: {losses['boundary']:.6f}, "
                      f"L1: {losses['l1']:.6f}, Total: {losses['total']:.6f}")
                if history['l2_error']:
                    print(f"  L2 Error: {history['l2_error'][-1]:.6e}")
                print(f"  Active params: {active}/{total} ({active/total*100:.1f}%)")
                print(f"  Sparsity: {sparsity}")
                print(f"  LR: {current_lr:.6f}")
        
        return history
    
    def evaluate_on_grid(self, n_points=50):
        """
        Evaluate solution on uniform grid.
        
        Returns:
            x, y, u_pred, u_exact (if available)
        """
        x = torch.linspace(0, 1, n_points, device=self.device)
        y = torch.linspace(0, 1, n_points, device=self.device)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        
        xy = torch.stack([xx.flatten(), yy.flatten()], dim=1)

        # Normalize grid points to [-1,1] for model input
        xy_norm = 2 * xy - 1

        with torch.no_grad():
            u_model = self.model(xy_norm, track_activity=False).reshape(n_points, n_points)
            if self.basis_coeff is not None:
                # evaluate analytic basis on original grid points
                u_basis = self.pde.exact_solution(xx, yy)
                u_pred = u_model + (self.basis_coeff * u_basis)
            else:
                u_pred = u_model
        
        u_exact = None
        if self.pde.source_type == 'sin':
            u_exact = self.pde.exact_solution(xx, yy)
        
        return xx.cpu().numpy(), yy.cpu().numpy(), u_pred.cpu().numpy(), \
               u_exact.cpu().numpy() if u_exact is not None else None


def plot_results(xx, yy, u_pred, u_exact=None, save_path=None):
    """Visualize PDE solution."""
    if u_exact is not None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), subplot_kw={'projection': '3d'})
        
        # Predicted solution
        axes[0].plot_surface(xx, yy, u_pred, cmap=cm.viridis)
        axes[0].set_title('Predicted Solution')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('y')
        axes[0].set_zlabel('u')
        
        # Exact solution
        axes[1].plot_surface(xx, yy, u_exact, cmap=cm.viridis)
        axes[1].set_title('Exact Solution')
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('y')
        axes[1].set_zlabel('u')
        
        # Error
        error = np.abs(u_pred - u_exact)
        axes[2].plot_surface(xx, yy, error, cmap=cm.plasma)
        axes[2].set_title(f'Absolute Error (max: {error.max():.2e})')
        axes[2].set_xlabel('x')
        axes[2].set_ylabel('y')
        axes[2].set_zlabel('|error|')
    else:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(xx, yy, u_pred, cmap=cm.viridis)
        ax.set_title('Predicted Solution')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('u')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    plt.close()


def plot_training_history(history, save_path=None):
    """Plot loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss components
    axes[0].semilogy(history['total_loss'], label='Total Loss', linewidth=2)
    axes[0].semilogy(history['pde_loss'], label='PDE Loss', alpha=0.7)
    axes[0].semilogy(history['boundary_loss'], label='Boundary Loss', alpha=0.7)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (log scale)')
    axes[0].set_title('Training Loss Components')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # L2 error (if available)
    if history['l2_error']:
        axes[1].semilogy(history['l2_error'], color='red', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('L2 Error (log scale)')
        axes[1].set_title('Solution Error vs Exact')
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    plt.close()