import torch

def b_splines(x, grid, spline_order):
    """
    Compute B-spline bases for input x given a grid and spline order.
    Args:
        x: (batch_size, in_features)
        grid: (in_features, grid_size + 2 * spline_order + 1)
        spline_order: int
    Returns:
        bases: (batch_size, in_features, grid_size + spline_order)
    """
    x = x.unsqueeze(-1)
    bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
    for k in range(1, spline_order + 1):
        bases = ((x - grid[:, : -(k + 1)]) / (grid[:, k:-1] - grid[:, : -(k + 1)]) * bases[:, :, :-1]) + \
                ((grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:(-k)]) * bases[:, :, 1:])
    return bases.contiguous()

def curve2coeff(x, y, grid, spline_order):
    """
    Compute coefficients for B-splines to fit the curve (x, y).
    Args:
        x: (batch_size, in_features)
        y: (batch_size, in_features, out_features)
        grid: (in_features, grid_size + 2 * spline_order + 1)
        spline_order: int
    Returns:
        coefficients: (out_features, in_features, grid_size + spline_order)
    """
    # A: (in_features, batch_size, grid_size + spline_order)
    A = b_splines(x, grid, spline_order).transpose(0, 1)
    # B: (in_features, batch_size, out_features)
    B = y.transpose(0, 1)
    # solution: (in_features, grid_size + spline_order, out_features)
    solution = torch.linalg.lstsq(A, B).solution
    # result: (out_features, in_features, grid_size + spline_order)
    return solution.permute(2, 0, 1).contiguous()
