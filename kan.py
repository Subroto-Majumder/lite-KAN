import torch
import torch.nn as nn
import numpy as np
from spline import coeff2curve, curve2coeff

class KAN_Layer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, deg, knot_vec, device='cpu'):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.deg = deg
        self.knot_vec = torch.tensor(knot_vec, device=device)
        self.device = device

        # number of basis functions per spline
        self.n_basis = len(knot_vec) - deg - 1

        # Each (input j → node i) has its own spline coefficients
        # Shape: [out_dim, in_dim, n_basis]
        self.coeff = torch.nn.Parameter(
        torch.empty(out_dim, in_dim, self.n_basis, device=device)
        )
        torch.nn.init.xavier_uniform_(self.coeff)


    def forward(self, u):
        """
        Forward pass with full batching support.
        
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
                # Get coefficients for this (input_j → output_i) connection
                # coeff_ij shape: [n_basis]
                coeff_ij = self.coeff[i, j]
                
                # Get input values for all samples in the batch for input dimension j
                # u[:, j] shape: [batch_size]
                u_j = u[:, j]
                
                # Evaluate spline for all batch samples at once
                # curve_ij shape: [batch_size]
                curve_ij = coeff2curve(coeff_ij, u_j, self.deg, self.knot_vec, self.device)
                
                # Accumulate to output
                out[:, i] += curve_ij
        
        return out
        
    


