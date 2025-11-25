# Code to build basis functions/matrix for B splines
import numpy as np
import torch

def Cox_De_Boor(u, k, deg, knot_vec, device='cpu'):
    """
    Evaluate B-spline basis function recursively.
    
    Args:
        u: evaluation points [batch_size] or scalar
        k: basis function index
        deg: spline degree
        knot_vec: knot vector
        device: computation device
    
    Returns:
        Basis function values at u
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
    Evaluate spline curve given coefficients.
    
    Args:
        coeff: spline coefficients [n_basis]
        u: evaluation points [batch_size] or scalar
        deg: spline degree
        knot_vec: knot vector
        device: computation device
    
    Returns:
        Curve values at u
    """
    n_basis = len(knot_vec) - deg - 1
    curve = torch.zeros_like(u)
    
    for k in range(n_basis):
        b_k = Cox_De_Boor(u, k, deg, knot_vec, device)
        curve = curve + coeff[k] * b_k
    
    return curve


def curve2coeff(curve, u_samples, deg, knot_vec, device='cpu'):
    """
    Fit spline coefficients to curve values using least squares.
    
    Args:
        curve: target curve values at sample points [n_samples]
        u_samples: sample point locations [n_samples]
        deg: spline degree
        knot_vec: knot vector
        device: computation device
    
    Returns:
        Fitted coefficients [n_basis]
    """
    n_basis = len(knot_vec) - deg - 1
    n_samples = len(u_samples)
    
    # Build basis matrix
    B_mat = torch.zeros((n_samples, n_basis), device=device)
    for i, u in enumerate(u_samples):
        for k in range(n_basis):
            B_mat[i, k] = Cox_De_Boor(u, k, deg, knot_vec, device)
    
    # Solve least squares
    B_pinv = torch.pinverse(B_mat)
    coeffs = B_pinv @ curve
    
    return coeffs