"""
Test script for KAN with adaptive pruning and grid extension.

This demonstrates the new features:
1. Edge-level pruning based on importance
2. Adaptive knot refinement based on curvature
3. Training with automatic adaptive operations

Following Liu et al., 2024 (KAN paper)
"""

import torch
import torch.nn as nn
from kan import KAN_Layer
import matplotlib.pyplot as plt
import numpy as np


def test_basic_pruning():
    """Test basic pruning functionality."""
    print("=" * 60)
    print("TEST 1: Basic Edge Pruning")
    print("=" * 60)
    
    # Create layer with pruning enabled
    knot_vec = torch.linspace(-3, 3, 20)
    layer = KAN_Layer(
        in_dim=3,
        out_dim=4,
        deg=3,
        knot_vec=knot_vec,
        enable_pruning=True,
        pruning_threshold=0.05,
        pruning_mode="mask"
    )
    
    print(f"Initial state:")
    stats = layer.get_stats()
    print(f"  Active edges: {stats['n_active_edges']}/{stats['n_total_edges']}")
    print(f"  Sparsity: {stats['sparsity']:.2%}")
    
    # Generate random batch
    batch = torch.randn(100, 3)
    
    # Compute importance
    importance = layer.compute_importance(batch)
    print(f"\nImportance scores (shape {importance.shape}):")
    print(f"  Min: {importance.min().item():.4f}")
    print(f"  Max: {importance.max().item():.4f}")
    print(f"  Mean: {importance.mean().item():.4f}")
    
    # Perform pruning
    info = layer.prune_edges(importance=importance)
    print(f"\nPruning results:")
    print(f"  Pruned: {info['pruned']}")
    if info['pruned']:
        print(f"  Mode: {info['mode']}")
        print(f"  Edges pruned: {info['n_pruned']}")
        print(f"  Active edges: {info['n_active']}/{stats['n_total_edges']}")
        print(f"  Sparsity: {info['sparsity']:.2%}")
    
    print("\n✓ Basic pruning test passed\n")


def test_basic_grid_extension():
    """Test basic grid extension functionality."""
    print("=" * 60)
    print("TEST 2: Basic Grid Extension")
    print("=" * 60)
    
    # Create layer with grid extension enabled
    knot_vec = torch.linspace(-2, 2, 10)  # Start with few knots
    layer = KAN_Layer(
        in_dim=2,
        out_dim=3,
        deg=3,
        knot_vec=knot_vec,
        enable_grid_extension=True,
        curvature_percentile=85.0,
        max_knots_per_extension=3
    )
    
    print(f"Initial state:")
    stats = layer.get_stats()
    print(f"  Knots: {stats['n_knots']}")
    print(f"  Basis functions: {stats['n_basis']}")
    print(f"  Parameters: {stats['n_parameters']}")
    
    # Generate samples for curvature estimation
    samples = torch.linspace(-2, 2, 200).unsqueeze(1).expand(-1, 2)
    
    # Compute curvature
    curvature = layer.estimate_edge_curvature(samples)
    print(f"\nCurvature (shape {curvature.shape}):")
    print(f"  Min: {curvature.min().item():.4f}")
    print(f"  Max: {curvature.max().item():.4f}")
    print(f"  Mean: {curvature.mean().item():.4f}")
    
    # Perform grid extension
    info = layer.extend_grid(input_samples=samples, curvature=curvature)
    print(f"\nGrid extension results:")
    print(f"  Extended: {info['extended']}")
    if info['extended']:
        print(f"  Knots added: {info['n_knots_added']}")
        print(f"  Old basis functions: {info['old_n_basis']}")
        print(f"  New basis functions: {info['new_n_basis']}")
        print(f"  Insertion points: {[f'{x:.3f}' for x in info['insertion_points']]}")
    
    print(f"\nFinal state:")
    stats = layer.get_stats()
    print(f"  Knots: {stats['n_knots']}")
    print(f"  Basis functions: {stats['n_basis']}")
    print(f"  Parameters: {stats['n_parameters']}")
    
    print("\n✓ Basic grid extension test passed\n")


def test_training_with_pruning():
    """Test training with automatic pruning."""
    print("=" * 60)
    print("TEST 3: Training with Automatic Pruning")
    print("=" * 60)
    
    # Training data: learn sin(x)
    xs = torch.linspace(-3, 3, 100).unsqueeze(1)
    targets = torch.sin(xs.squeeze())
    
    # Create layer with pruning
    knot_vec = torch.linspace(-5, 5, 30)
    layer = KAN_Layer(
        in_dim=1,
        out_dim=1,
        deg=3,
        knot_vec=knot_vec,
        enable_pruning=True,
        pruning_threshold=0.01,
        pruning_frequency=20,  # Prune every 20 steps
        pruning_start_step=50  # Start pruning after 50 steps
    )
    
    optimizer = torch.optim.Adam(layer.parameters(), lr=0.01)
    
    losses = []
    pruning_events = []
    
    print("Training for 200 steps...")
    for step in range(200):
        optimizer.zero_grad()
        preds = layer(xs).squeeze()
        loss = 0.5 * ((preds - targets) ** 2).mean()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        # Increment step counter and check for pruning
        layer.step()
        info = layer.maybe_prune(xs)
        
        if info.get('pruned', False):
            pruning_events.append(step)
            print(f"  Step {step}: Pruned {info['n_pruned']} edges, "
                  f"sparsity={info['sparsity']:.2%}")
        
        if step % 50 == 0:
            print(f"  Step {step}: Loss={loss.item():.6f}")
    
    print(f"\nTraining complete!")
    print(f"  Final loss: {losses[-1]:.6f}")
    print(f"  Pruning events: {len(pruning_events)}")
    
    stats = layer.get_stats()
    print(f"  Active edges: {stats['n_active_edges']}/{stats['n_total_edges']}")
    print(f"  Final sparsity: {stats['sparsity']:.2%}")
    
    print("\n✓ Training with pruning test passed\n")
    
    return losses, pruning_events


def test_training_with_grid_extension():
    """Test training with automatic grid extension."""
    print("=" * 60)
    print("TEST 4: Training with Grid Extension")
    print("=" * 60)
    
    # Training data: learn x^2 * sin(3*x)
    xs = torch.linspace(-2, 2, 150).unsqueeze(1)
    targets = (xs.squeeze() ** 2) * torch.sin(3 * xs.squeeze())
    
    # Start with coarse grid
    knot_vec = torch.linspace(-3, 3, 8)
    layer = KAN_Layer(
        in_dim=1,
        out_dim=1,
        deg=3,
        knot_vec=knot_vec,
        enable_grid_extension=True,
        grid_extension_frequency=50,  # Extend every 50 steps
        grid_extension_start_step=25,  # Start after 25 steps
        curvature_percentile=90.0,
        max_knots_per_extension=3
    )
    
    optimizer = torch.optim.Adam(layer.parameters(), lr=0.01)
    
    losses = []
    extension_events = []
    n_basis_history = [layer.n_basis]
    
    print(f"Starting with {layer.n_basis} basis functions")
    print("Training for 200 steps...")
    
    for step in range(200):
        optimizer.zero_grad()
        preds = layer(xs).squeeze()
        loss = 0.5 * ((preds - targets) ** 2).mean()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        # Increment step and check for grid extension
        layer.step()
        info = layer.maybe_refine_grid(xs)
        
        if info.get('extended', False):
            extension_events.append(step)
            print(f"  Step {step}: Added {info['n_knots_added']} knots, "
                  f"basis: {info['old_n_basis']} → {info['new_n_basis']}")
            n_basis_history.append(info['new_n_basis'])
            
            # Need to reinitialize optimizer after parameter shape change
            optimizer = torch.optim.Adam(layer.parameters(), lr=0.01)
        
        if step % 50 == 0:
            print(f"  Step {step}: Loss={loss.item():.6f}, "
                  f"n_basis={layer.n_basis}")
    
    print(f"\nTraining complete!")
    print(f"  Final loss: {losses[-1]:.6f}")
    print(f"  Extension events: {len(extension_events)}")
    
    stats = layer.get_stats()
    print(f"  Final basis functions: {stats['n_basis']}")
    print(f"  Final knots: {stats['n_knots']}")
    print(f"  Total extensions: {stats['n_grid_extensions']}")
    
    print("\n✓ Training with grid extension test passed\n")
    
    return losses, extension_events, n_basis_history


def test_combined_adaptive():
    """Test training with both pruning and grid extension."""
    print("=" * 60)
    print("TEST 5: Combined Pruning + Grid Extension")
    print("=" * 60)
    
    # Training data: learn sin(x) with 2-layer network
    xs = torch.linspace(-3, 3, 100).unsqueeze(1)
    targets = torch.sin(xs.squeeze())
    
    # Create network with both features
    knot_vec = torch.linspace(-5, 5, 12)
    
    model = nn.Sequential(
        KAN_Layer(
            in_dim=1,
            out_dim=3,
            deg=3,
            knot_vec=knot_vec,
            enable_pruning=True,
            pruning_threshold=0.01,
            pruning_frequency=30,
            pruning_start_step=60,
            enable_grid_extension=True,
            grid_extension_frequency=40,
            grid_extension_start_step=20,
            max_knots_per_extension=2
        ),
        KAN_Layer(
            in_dim=3,
            out_dim=1,
            deg=3,
            knot_vec=knot_vec,
            enable_pruning=True,
            pruning_threshold=0.01,
            pruning_frequency=30,
            pruning_start_step=60,
            enable_grid_extension=True,
            grid_extension_frequency=40,
            grid_extension_start_step=20,
            max_knots_per_extension=2
        )
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    losses = []
    events = {'prune': [], 'extend': []}
    
    print("Training 2-layer network for 200 steps...")
    
    for step in range(200):
        optimizer.zero_grad()
        preds = model(xs).squeeze()
        loss = 0.5 * ((preds - targets) ** 2).mean()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        # Update both layers
        for i, layer in enumerate(model):
            layer.step()
            
            # Check pruning
            prune_info = layer.maybe_prune(xs)
            if prune_info.get('pruned', False):
                events['prune'].append((step, i))
                print(f"  Step {step}, Layer {i}: Pruned {prune_info['n_pruned']} edges")
            
            # Check grid extension
            extend_info = layer.maybe_refine_grid(xs)
            if extend_info.get('extended', False):
                events['extend'].append((step, i))
                print(f"  Step {step}, Layer {i}: Added {extend_info['n_knots_added']} knots")
                
                # Reinitialize optimizer after parameter change
                optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        if step % 50 == 0:
            print(f"  Step {step}: Loss={loss.item():.6f}")
    
    print(f"\nTraining complete!")
    print(f"  Final loss: {losses[-1]:.6f}")
    print(f"  Pruning events: {len(events['prune'])}")
    print(f"  Extension events: {len(events['extend'])}")
    
    print("\nLayer 0 stats:")
    stats0 = model[0].get_stats()
    print(f"  Active edges: {stats0['n_active_edges']}/{stats0['n_total_edges']}")
    print(f"  Sparsity: {stats0['sparsity']:.2%}")
    print(f"  Basis functions: {stats0['n_basis']}")
    
    print("\nLayer 1 stats:")
    stats1 = model[1].get_stats()
    print(f"  Active edges: {stats1['n_active_edges']}/{stats1['n_total_edges']}")
    print(f"  Sparsity: {stats1['sparsity']:.2%}")
    print(f"  Basis functions: {stats1['n_basis']}")
    
    print("\n✓ Combined adaptive test passed\n")
    
    return losses, events


def visualize_results():
    """Create visualizations of adaptive features."""
    print("=" * 60)
    print("Creating Visualizations")
    print("=" * 60)
    
    # Run training experiments
    print("\nRunning pruning experiment...")
    losses_prune, prune_events = test_training_with_pruning()
    
    print("\nRunning grid extension experiment...")
    losses_extend, extend_events, n_basis_hist = test_training_with_grid_extension()
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Training loss with pruning
    ax = axes[0, 0]
    ax.plot(losses_prune, linewidth=2)
    for event in prune_events:
        ax.axvline(x=event, color='red', alpha=0.3, linestyle='--', linewidth=1)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Loss')
    ax.set_title('Training with Pruning\n(Red lines = pruning events)')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Plot 2: Training loss with grid extension
    ax = axes[0, 1]
    ax.plot(losses_extend, linewidth=2, color='green')
    for event in extend_events:
        ax.axvline(x=event, color='blue', alpha=0.3, linestyle='--', linewidth=1)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Loss')
    ax.set_title('Training with Grid Extension\n(Blue lines = extension events)')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Plot 3: Basis function growth
    ax = axes[1, 0]
    extension_steps = [0] + extend_events
    ax.step(extension_steps, n_basis_hist, where='post', linewidth=2, color='purple')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Number of Basis Functions')
    ax.set_title('Adaptive Grid Extension:\nBasis Function Growth')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Combined comparison
    ax = axes[1, 1]
    ax.plot(losses_prune, label='With Pruning', linewidth=2, alpha=0.7)
    ax.plot(losses_extend, label='With Grid Extension', linewidth=2, alpha=0.7)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Loss')
    ax.set_title('Comparison of Adaptive Strategies')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('adaptive_kan_results.png', dpi=150, bbox_inches='tight')
    print("\n✓ Visualization saved to 'adaptive_kan_results.png'\n")
    plt.show()


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("KAN ADAPTIVE FEATURES TEST SUITE")
    print("Implementation based on Liu et al., 2024")
    print("=" * 60 + "\n")
    
    # Run individual tests
    test_basic_pruning()
    test_basic_grid_extension()
    test_training_with_pruning()
    test_training_with_grid_extension()
    test_combined_adaptive()
    
    # Create visualizations
    try:
        visualize_results()
    except Exception as e:
        print(f"Visualization skipped: {e}")
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED! ✓")
    print("=" * 60)
    print("\nYou can now use adaptive KAN features:")
    print("\n  # Enable pruning:")
    print("  layer = KAN_Layer(..., enable_pruning=True, pruning_threshold=0.01)")
    print("\n  # Enable grid extension:")
    print("  layer = KAN_Layer(..., enable_grid_extension=True)")
    print("\n  # During training:")
    print("  layer.step()            # Increment counter")
    print("  layer.maybe_prune()     # Check & prune")
    print("  layer.maybe_refine_grid()  # Check & extend")
    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
