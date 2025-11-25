"""
Complete experimental pipeline for KAN Part 2 presentation.

Experiments:
1. Baseline: Train KAN without pruning/extension
2. With Pruning: Show parameter reduction
3. With Grid Extension: Show accuracy improvement
4. Hyperparameter Study: Grid size, network width, learning rate
5. Comparison with paper results
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from enhanced_kan import KAN
from pde_solver import PoissonPDE, PDESolver, plot_results, plot_training_history

# Create results directory if it doesn't exist
os.makedirs('results', exist_ok=True)

def experiment_1_baseline(device='cpu'):
    """
    Experiment 1: Baseline KAN for Poisson equation.
    
    Purpose: Establish baseline performance without enhancements.
    """
    print("="*80)
    print("EXPERIMENT 1: Baseline KAN")
    print("="*80)
    
    # Setup
    pde = PoissonPDE(source_function='sin')
    model = KAN(width=[2, 5, 5, 1], deg=3, grid_size=5, grid_range=[-1, 1], device=device, activation='tanh', residual=True)
    solver = PDESolver(model, pde, device)
    
    # Training
    history = solver.train(
        n_interior=20,
        n_boundary=40,
        epochs=2000,
        lr=0.01,
        lambda_pde=1.0,
        lambda_bc=100.0,
        lambda_l1=0.0
    )
    
    # Evaluation
    xx, yy, u_pred, u_exact = solver.evaluate_on_grid(n_points=50)
    
    # Compute final metrics
    l2_error = np.sqrt(np.mean((u_pred - u_exact) ** 2))
    max_error = np.abs(u_pred - u_exact).max()
    active, total = model.count_parameters()
    
    print(f"\nFinal Results:")
    print(f"  L2 Error: {l2_error:.6e}")
    print(f"  Max Error: {max_error:.6e}")
    print(f"  Parameters: {active}")
    
    # Visualization
    plot_training_history(history, save_path='results/exp1_training.png')
    plot_results(xx, yy, u_pred, u_exact, save_path='results/exp1_solution.png')
    
    return history, model, {'l2_error': l2_error, 'max_error': max_error, 'params': active}


def experiment_2_pruning(device='cpu'):
    """
    Experiment 2: KAN with pruning.
    
    Purpose: Demonstrate parameter reduction while maintaining accuracy.
    """
    print("="*80)
    print("EXPERIMENT 2: KAN with Pruning")
    print("="*80)
    
    # Setup
    pde = PoissonPDE(source_function='sin')
    model = KAN(width=[2, 10, 10, 1], deg=3, grid_size=5, grid_range=[-1, 1], device=device, activation='tanh', residual=True)
    solver = PDESolver(model, pde, device)
    
    initial_params = model.count_parameters()[1]
    print(f"Initial parameters: {initial_params}")
    
    # Training with pruning
    history = solver.train(
        n_interior=20,
        n_boundary=20,
        epochs=2000,
        lr=0.01,
        lambda_pde=1.0,
        lambda_bc=10.0,
        lambda_l1=1e-4,  # L1 regularization encourages sparsity
        prune_interval=500,  # Prune every 500 epochs
        prune_threshold=0.05
    )
    
    # Evaluation
    xx, yy, u_pred, u_exact = solver.evaluate_on_grid(n_points=50)
    
    # Compute final metrics
    l2_error = np.sqrt(np.mean((u_pred - u_exact) ** 2))
    active, total = model.count_parameters()
    sparsity = (total - active) / total * 100
    
    print(f"\nFinal Results:")
    print(f"  L2 Error: {l2_error:.6e}")
    print(f"  Active Parameters: {active}/{total} ({100-sparsity:.1f}%)")
    print(f"  Sparsity: {sparsity:.1f}%")
    print(f"  Compression: {initial_params/active:.2f}x")
    
    # Visualization
    plot_training_history(history, save_path='results/exp2_training.png')
    plot_results(xx, yy, u_pred, u_exact, save_path='results/exp2_solution.png')
    
    return history, model, {'l2_error': l2_error, 'sparsity': sparsity, 
                           'compression': initial_params/active}


def experiment_3_grid_extension(device='cpu'):
    """
    Experiment 3: KAN with grid extension.
    
    Purpose: Show adaptive refinement improves accuracy.
    """
    print("="*80)
    print("EXPERIMENT 3: KAN with Grid Extension")
    print("="*80)
    
    # Setup
    pde = PoissonPDE(source_function='sin')
    model = KAN(width=[2, 5, 5, 1], deg=3, grid_size=3, grid_range=[-1, 1], device=device, activation='tanh', residual=True)
    solver = PDESolver(model, pde, device)
    
    # Training with grid extension schedule
    grid_schedule = {
        500: 5,   # Extend to grid_size=5 at epoch 500
        1000: 7,  # Extend to grid_size=7 at epoch 1000
        1500: 10  # Extend to grid_size=10 at epoch 1500
    }
    
    history = solver.train(
        n_interior=20,
        n_boundary=20,
        epochs=2000,
        lr=0.01,
        lambda_pde=1.0,
        lambda_bc=10.0,
        lambda_l1=0.0,
        grid_extension_schedule=grid_schedule
    )
    
    # Evaluation
    xx, yy, u_pred, u_exact = solver.evaluate_on_grid(n_points=50)
    
    # Compute final metrics
    l2_error = np.sqrt(np.mean((u_pred - u_exact) ** 2))
    active, total = model.count_parameters()
    
    print(f"\nFinal Results:")
    print(f"  L2 Error: {l2_error:.6e}")
    print(f"  Final Grid Size: {model.layers[0].grid_size}")
    print(f"  Final Parameters: {active}")
    
    # Visualization
    plot_training_history(history, save_path='results/exp3_training.png')
    plot_results(xx, yy, u_pred, u_exact, save_path='results/exp3_solution.png')
    
    return history, model, {'l2_error': l2_error, 'params': active}


def experiment_4_hyperparameter_study(device='cpu'):
    """
    Experiment 4: Hyperparameter sensitivity analysis.
    
    Purpose: Show how grid size and width affect performance.
    """
    print("="*80)
    print("EXPERIMENT 4: Hyperparameter Study")
    print("="*80)
    
    pde = PoissonPDE(source_function='sin')
    
    # Vary grid size
    grid_sizes = [3, 5, 7, 10]
    grid_results = []
    
    print("\nVarying Grid Size (fixed width=[2,5,1]):")
    for g in grid_sizes:
        model = KAN(width=[2, 5, 1], deg=3, grid_size=g, grid_range=[-1, 1], device=device, activation='tanh', residual=True)
        solver = PDESolver(model, pde, device)
        
        history = solver.train(
            n_interior=20, n_boundary=20, epochs=500, lr=0.01,
            lambda_pde=1.0, lambda_bc=10.0, lambda_l1=0.0
        )
        
        xx, yy, u_pred, u_exact = solver.evaluate_on_grid(n_points=50)
        l2_error = np.sqrt(np.mean((u_pred - u_exact) ** 2))
        active, _ = model.count_parameters()
        
        grid_results.append({'grid_size': g, 'l2_error': l2_error, 'params': active})
        print(f"  Grid={g}: L2={l2_error:.6e}, Params={active}")
    
    # Vary width
    widths = [[2, 3, 1], [2, 5, 1], [2, 10, 1], [2, 5, 5, 1]]
    width_results = []
    
    print("\nVarying Width (fixed grid_size=5):")
    for w in widths:
        model = KAN(width=w, deg=3, grid_size=5, grid_range=[-1, 1], device=device, activation='tanh', residual=True)
        solver = PDESolver(model, pde, device)
        
        history = solver.train(
            n_interior=20, n_boundary=20, epochs=500, lr=0.01,
            lambda_pde=1.0, lambda_bc=10.0, lambda_l1=0.0
        )
        
        xx, yy, u_pred, u_exact = solver.evaluate_on_grid(n_points=50)
        l2_error = np.sqrt(np.mean((u_pred - u_exact) ** 2))
        active, _ = model.count_parameters()
        
        width_results.append({'width': str(w), 'l2_error': l2_error, 'params': active})
        print(f"  Width={w}: L2={l2_error:.6e}, Params={active}")
    
    # Visualize results
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Grid size effect
    axes[0].semilogy([r['grid_size'] for r in grid_results],
                     [r['l2_error'] for r in grid_results], 
                     'o-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Grid Size')
    axes[0].set_ylabel('L2 Error (log scale)')
    axes[0].set_title('Effect of Grid Size')
    axes[0].grid(True, alpha=0.3)
    
    # Width effect
    axes[1].semilogy(range(len(width_results)),
                     [r['l2_error'] for r in width_results], 
                     'o-', linewidth=2, markersize=8)
    axes[1].set_xticks(range(len(width_results)))
    axes[1].set_xticklabels([r['width'] for r in width_results], rotation=45)
    axes[1].set_xlabel('Network Width')
    axes[1].set_ylabel('L2 Error (log scale)')
    axes[1].set_title('Effect of Network Width')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/exp4_hyperparameter_study.png', dpi=100, bbox_inches='tight')
    print("Plot saved to results/exp4_hyperparameter_study.png")
    plt.close()
    
    return grid_results, width_results


def experiment_5_paper_comparison(device='cpu'):
    """
    Experiment 5: Reproduce paper results.
    
    Purpose: Compare with reported results from KAN paper Section 3.2.
    Paper reports: 2-layer width-10 KAN achieves ~10^-7 MSE
    """
    print("="*80)
    print("EXPERIMENT 5: Paper Result Reproduction")
    print("="*80)
    print("Target: Reproduce Section 3.2 results")
    print("Paper: 2-layer width-10 KAN, MSE ~10^-7")
    
    pde = PoissonPDE(source_function='sin')
    
    # Paper configuration (approximately)
    model = KAN(width=[2, 10, 1], deg=3, grid_size=5, grid_range=[-1, 1], device=device, activation='tanh', residual=True)
    solver = PDESolver(model, pde, device)
    
    active, total = model.count_parameters()
    print(f"Model: {model.width}, Parameters: {active}")
    
    # Train longer for convergence
    history = solver.train(
        n_interior=30,  # More collocation points
        n_boundary=30,
        epochs=3000,
        lr=0.01,
        lambda_pde=1.0,
        lambda_bc=100.0,
        lambda_l1=0.0
    )
    
    # Evaluation
    xx, yy, u_pred, u_exact = solver.evaluate_on_grid(n_points=50)
    l2_error = np.sqrt(np.mean((u_pred - u_exact) ** 2))
    mse = np.mean((u_pred - u_exact) ** 2)
    
    print(f"\nFinal Results:")
    print(f"  MSE: {mse:.6e} (Paper target: ~10^-7)")
    print(f"  L2 Error: {l2_error:.6e}")
    print(f"  Parameters: {active}")
    
    # Visualization
    plot_training_history(history, save_path='results/exp5_training.png')
    plot_results(xx, yy, u_pred, u_exact, save_path='results/exp5_solution.png')
    
    return history, {'mse': mse, 'l2_error': l2_error, 'params': active}


def run_all_experiments(device='cpu'):
    """Run complete experimental suite."""
    results = {}
    
    print("\n" + "="*80)
    print("RUNNING COMPLETE EXPERIMENTAL SUITE")
    print("="*80 + "\n")
    
    # Experiment 1
    #/ _, _, results['baseline'] = experiment_1_baseline(device)
    
    # Experiment 2
    _, _, results['pruning'] = experiment_2_pruning(device)
    
    # Experiment 3
    _, _, results['grid_extension'] = experiment_3_grid_extension(device)
    
    # Experiment 4
    results['grid_study'], results['width_study'] = experiment_4_hyperparameter_study(device)
    
    # Experiment 5
    _, results['paper_comparison'] = experiment_5_paper_comparison(device)
    
    # Summary
    print("\n" + "="*80)
    print("EXPERIMENTAL SUMMARY")
    print("="*80)
    print(f"\nBaseline: L2={results['baseline']['l2_error']:.6e}, Params={results['baseline']['params']}")
    print(f"Pruning: L2={results['pruning']['l2_error']:.6e}, Sparsity={results['pruning']['sparsity']:.1f}%")
    print(f"Grid Ext: L2={results['grid_extension']['l2_error']:.6e}")
    print(f"Paper Target: MSE~10^-7, Achieved: {results['paper_comparison']['mse']:.6e}")
    
    return results


if __name__ == "__main__":
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Run experiments
    # Option 1: Run individual experiment
    # experiment_1_baseline(device)
    
    # Option 2: Run all experiments
    results = run_all_experiments(device)