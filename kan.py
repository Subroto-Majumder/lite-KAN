import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from spline import b_splines, curve2coeff

class KANLayer(nn.Module):
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3, scale_noise=0.1, scale_base=1.0, scale_spline=1.0, grid_range=[-1, 1]):
        super(KANLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.grid_range = grid_range

        self.update_grid_buffer(grid_size, device='cpu') 

        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(torch.Tensor(out_features, in_features, grid_size + spline_order))
        self.spline_scaler = nn.Parameter(torch.ones(out_features, in_features))
        
        self.reset_parameters(scale_noise, scale_base, scale_spline)
        self.acts = None

    def update_grid_buffer(self, grid_size, device='cpu'):
        h = (self.grid_range[1] - self.grid_range[0]) / grid_size
        grid = ((torch.arange(-self.spline_order, grid_size + self.spline_order + 1, device=device) * h + self.grid_range[0]).expand(self.in_features, -1).contiguous())
        self.register_buffer("grid", grid)

    def reset_parameters(self, scale_noise, scale_base, scale_spline):
        nn.init.kaiming_uniform_(self.base_weight, a=np.sqrt(5) * scale_base)
        with torch.no_grad():
            noise = (torch.rand(self.grid_size + 1, self.in_features, self.out_features) - 0.5) * scale_noise / self.grid_size
            self.spline_weight.data.copy_(
                (scale_spline if scale_spline is not None else 1.0) * curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order], 
                    noise,
                    self.grid,
                    self.spline_order
                )
            )

    def forward(self, x):
        base_output = F.linear(F.silu(x), self.base_weight)
        bs = b_splines(x, self.grid, self.spline_order)
        
        # Calculate scaled weights. 
        # Crucial: Pruned connections have spline_scaler=0, effectively killing the spline.
        scaled_weight = self.spline_weight * self.spline_scaler.unsqueeze(-1)
        
        self.acts = torch.einsum('bij,oij->boi', bs, scaled_weight)
        return base_output + self.acts.sum(dim=2)

    def prune(self, threshold):
        with torch.no_grad():
            mask = (self.spline_scaler.abs() > threshold).float()
            self.spline_scaler.mul_(mask)
            self.base_weight.mul_(mask)
            return (self.spline_scaler == 0).sum().item()

    def extend_grid(self, new_grid_size):
        current_device = self.grid.device
        x_sample = torch.linspace(self.grid_range[0], self.grid_range[1], 1000, device=current_device).unsqueeze(1).expand(-1, self.in_features)
        
        with torch.no_grad():
            old_bs = b_splines(x_sample, self.grid, self.spline_order)
            old_spline_out = torch.einsum('bij,oij->boi', old_bs, self.spline_weight)
            y_target = old_spline_out.permute(0, 2, 1) 

        self.grid_size = new_grid_size
        self.update_grid_buffer(new_grid_size, device=current_device)
        self.spline_weight = nn.Parameter(torch.Tensor(self.out_features, self.in_features, new_grid_size + self.spline_order).to(current_device))
        
        with torch.no_grad():
            new_coeffs = curve2coeff(x_sample, y_target, self.grid, self.spline_order)
            self.spline_weight.data.copy_(new_coeffs)

class KAN(nn.Module):
    def __init__(self, layers_hidden, grid_size=5, spline_order=3, scale_noise=0.1, scale_base=1.0, scale_spline=1.0):
        super(KAN, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers_hidden) - 1):
            self.layers.append(KANLayer(
                layers_hidden[i], 
                layers_hidden[i+1], 
                grid_size=grid_size, 
                spline_order=spline_order,
                scale_noise=scale_noise,
                scale_base=scale_base,
                scale_spline=scale_spline
            ))
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def extend_grid(self, new_grid_size):
        for layer in self.layers:
            layer.extend_grid(new_grid_size)

    def prune(self, threshold=1e-2):
        total_pruned = 0
        for layer in self.layers:
            total_pruned += layer.prune(threshold)
        return total_pruned

    def get_reg_loss(self):
        return sum(layer.spline_scaler.abs().mean() for layer in self.layers)
