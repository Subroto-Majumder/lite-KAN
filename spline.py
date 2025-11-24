import numpy as np
import torch

def Cox_De_Boor(u, k, deg, knot_vec, device='cpu'):
    """
    Cox-De Boor recursion for B-spline basis functions.
    
    Args:
        u: input tensor of shape [...] (supports batched inputs)
        k: basis function index
        deg: spline degree
        knot_vec: knot vector tensor
        device: computation device
    
    Returns:
        B-spline basis function values of shape [...]
    """
    if deg == 0:  # Base case: degree 0 (piecewise constant)
        return ((u >= knot_vec[k]) & (u < knot_vec[k+1])).float()
    else:
        denom1 = knot_vec[k+deg] - knot_vec[k]
        denom2 = knot_vec[k+deg+1] - knot_vec[k+1]
        term1 = torch.zeros_like(u)
        term2 = torch.zeros_like(u)
        if denom1 > 0:
            term1 = ((u - knot_vec[k]) / denom1) * Cox_De_Boor(u, k, deg-1, knot_vec, device)
        if denom2 > 0:
            term2 = ((knot_vec[k+deg+1] - u) / denom2) * Cox_De_Boor(u, k+1, deg-1, knot_vec, device)
        return term1 + term2

def coeff2curve(coeff, u, deg, knot_vec, device='cpu'):
    """
    Evaluate B-spline curve given coefficients and input points.
    
    Args:
        coeff: coefficients tensor of shape [n_basis] or [..., n_basis]
        u: input tensor of shape [...] (supports batched inputs)
        deg: spline degree
        knot_vec: knot vector tensor
        device: computation device
    
    Returns:
        curve values of shape [...]
    """
    n_basis = len(knot_vec) - deg - 1
    curve = torch.zeros_like(u)
    
    # Handle both batched and non-batched coefficients
    coeff_is_batched = coeff.ndim > 1
    
    for k in range(n_basis):
        b_k = Cox_De_Boor(u, k, deg, knot_vec, device)
        if coeff_is_batched:
            # coeff shape: [..., n_basis], select coeff[..., k]
            curve = curve + coeff[..., k] * b_k
        else:
            # coeff shape: [n_basis], select coeff[k]
            curve = curve + coeff[k] * b_k
    return curve

def curve2coeff(curve,u_samples,deg,knot_vec,device='cpu'):
    # Use pseudo inverse to get least squares values for coefficients
    n_basis=len(knot_vec)-deg-1
    n_samples=len(u_samples)
    B_mat=torch.zeros((n_samples,n_basis), device=device)
    for i,u in enumerate(u_samples):
        for k in range(n_basis):
            B_mat[i,k]=Cox_De_Boor(u,k,deg,knot_vec,device)
    B_pinv=torch.pinverse(B_mat)
    coeffs=B_pinv @ curve
    return coeffs