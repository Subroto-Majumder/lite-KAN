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


def insert_knot(knot_vec, new_knot, deg, coeff, device='cpu'):
    """
    Insert a new knot into the knot vector and update coefficients.
    
    This implements the Boehm knot insertion algorithm for B-splines.
    The algorithm ensures that the spline curve remains unchanged after
    knot insertion, only its representation changes.
    
    Mathematical background:
    When inserting knot u_new, the new control points (coefficients) are
    computed as linear combinations of the old ones:
        c'_i = alpha_i * c_i + (1 - alpha_i) * c_{i-1}
    where alpha_i depends on the knot positions and degree.
    
    Args:
        knot_vec: current knot vector tensor of length m
        new_knot: scalar value of the knot to insert
        deg: degree of the B-spline
        coeff: current coefficient tensor of shape [..., n_basis]
               where n_basis = m - deg - 1
        device: computation device
    
    Returns:
        new_knot_vec: updated knot vector of length m+1
        new_coeff: updated coefficients of shape [..., n_basis+1]
    
    Example:
        >>> knot_vec = torch.tensor([0, 0, 0, 0.5, 1, 1, 1])
        >>> coeff = torch.tensor([1.0, 2.0, 3.0, 4.0])
        >>> new_knot_vec, new_coeff = insert_knot(knot_vec, 0.25, 2, coeff)
        >>> print(new_knot_vec.shape)  # [8]
        >>> print(new_coeff.shape)  # [5]
    
    Reference:
        Boehm, W. (1980). "Inserting new knots into B-spline curves."
        Computer-Aided Design, 12(4), 199-201.
    """
    knot_vec = knot_vec.to(device)
    coeff = coeff.to(device)
    
    # Find insertion index: where new_knot should be inserted
    # knot_vec[k] <= new_knot < knot_vec[k+1]
    k = torch.searchsorted(knot_vec, new_knot, right=False).item()
    
    # Ensure we don't insert outside valid range
    if k == 0:
        k = 1
    if k >= len(knot_vec):
        k = len(knot_vec) - 1
    
    # Insert the new knot into knot vector
    new_knot_vec = torch.cat([
        knot_vec[:k],
        torch.tensor([new_knot], device=device),
        knot_vec[k:]
    ])
    
    # Compute new coefficients using Boehm's algorithm
    n_basis = len(knot_vec) - deg - 1
    new_n_basis = n_basis + 1
    
    # Handle batched coefficients: [..., n_basis]
    coeff_shape = coeff.shape
    batch_dims = coeff_shape[:-1]
    
    # Initialize new coefficients
    new_coeff = torch.zeros(*batch_dims, new_n_basis, device=device)
    
    # Boehm's knot insertion algorithm
    # The affected range is [k-deg, k]
    for i in range(new_n_basis):
        if i <= k - deg - 1:
            # Coefficients before affected region remain unchanged
            if i < n_basis:
                new_coeff[..., i] = coeff[..., i]
        elif i >= k + 1:
            # Coefficients after affected region shift by one
            if i - 1 < n_basis:
                new_coeff[..., i] = coeff[..., i - 1]
        else:
            # Coefficients in affected region [k-deg, k] are blended
            # Compute alpha_i for blending
            if i + deg < len(knot_vec):
                denominator = knot_vec[i + deg] - knot_vec[i]
                if denominator > 1e-10:
                    alpha = (new_knot - knot_vec[i]) / denominator
                else:
                    alpha = 0.5
            else:
                alpha = 0.5
            
            # Blend: new_coeff[i] = (1-alpha)*coeff[i-1] + alpha*coeff[i]
            if i - 1 >= 0 and i - 1 < n_basis:
                new_coeff[..., i] = (1 - alpha) * coeff[..., i - 1]
            if i < n_basis:
                new_coeff[..., i] += alpha * coeff[..., i]
    
    return new_knot_vec, new_coeff


def insert_multiple_knots(knot_vec, new_knots, deg, coeff, device='cpu'):
    """
    Insert multiple knots sequentially into a B-spline.
    
    This is a convenience function that calls insert_knot multiple times.
    Knots are inserted in sorted order to maintain knot vector validity.
    
    Args:
        knot_vec: current knot vector
        new_knots: tensor of knot values to insert
        deg: degree of the B-spline
        coeff: current coefficients
        device: computation device
    
    Returns:
        new_knot_vec: updated knot vector
        new_coeff: updated coefficients
    
    Example:
        >>> knot_vec = torch.tensor([0, 0, 0, 1, 1, 1])
        >>> coeff = torch.tensor([1.0, 2.0, 3.0])
        >>> new_knots = torch.tensor([0.25, 0.5, 0.75])
        >>> new_kv, new_c = insert_multiple_knots(knot_vec, new_knots, 2, coeff)
        >>> print(new_kv.shape)  # [9] = original 6 + 3 new knots
    """
    # Sort new knots to maintain knot vector properties
    new_knots_sorted = torch.sort(new_knots)[0]
    
    current_knot_vec = knot_vec
    current_coeff = coeff
    
    for new_knot in new_knots_sorted:
        current_knot_vec, current_coeff = insert_knot(
            current_knot_vec, new_knot.item(), deg, current_coeff, device
        )
    
    return current_knot_vec, current_coeff


def refine_knot_vector_uniform(knot_vec, deg, coeff, refinement_factor=2, device='cpu'):
    """
    Uniformly refine a knot vector by inserting knots at midpoints.
    
    This is a simple refinement strategy that doubles the number of intervals.
    More sophisticated strategies can insert knots based on curvature.
    
    Args:
        knot_vec: current knot vector
        deg: degree of the B-spline
        coeff: current coefficients
        refinement_factor: how many new knots per interval (default 2 = midpoint)
        device: computation device
    
    Returns:
        new_knot_vec: refined knot vector
        new_coeff: updated coefficients
    
    Example:
        >>> knot_vec = torch.tensor([0, 0, 0, 1, 1, 1])
        >>> coeff = torch.tensor([1.0, 2.0, 3.0])
        >>> new_kv, new_c = refine_knot_vector_uniform(knot_vec, 2, coeff)
    """
    # Find unique interior knots (excluding repeated boundary knots)
    unique_knots = torch.unique(knot_vec)
    
    # Generate midpoints between consecutive unique knots
    new_knots = []
    for i in range(len(unique_knots) - 1):
        left = unique_knots[i]
        right = unique_knots[i + 1]
        
        # Insert refinement_factor-1 points in this interval
        for j in range(1, refinement_factor):
            t = j / refinement_factor
            new_knot = left * (1 - t) + right * t
            new_knots.append(new_knot.item())
    
    if len(new_knots) == 0:
        return knot_vec, coeff
    
    new_knots_tensor = torch.tensor(new_knots, device=device)
    
    return insert_multiple_knots(knot_vec, new_knots_tensor, deg, coeff, device)