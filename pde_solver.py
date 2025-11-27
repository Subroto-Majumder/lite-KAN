import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from kan import KAN
from mlp import MLP

# ==========================================
# 1. SETUP
# ==========================================
plt.switch_backend('Agg')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.5, device=0)
    print(f" Running on {torch.cuda.get_device_name(0)}")

# ==========================================
# 2. PDE DEFINITION
# ==========================================
class PoissonPDE:
    def __init__(self, num_domain=2500, num_boundary=500):
        self.x = torch.rand(num_domain, 1) * 2 - 1
        self.y = torch.rand(num_domain, 1) * 2 - 1
        self.xy = torch.cat([self.x, self.y], dim=1)
        
        x_edge = torch.linspace(-1, 1, num_boundary//4).unsqueeze(1)
        ones = torch.ones_like(x_edge)
        self.xy_b = torch.cat([
            torch.cat([x_edge, ones], dim=1),
            torch.cat([x_edge, -ones], dim=1),
            torch.cat([-ones, x_edge], dim=1),
            torch.cat([ones, x_edge], dim=1)
        ], dim=0)

    def to_device(self, device):
        self.xy = self.xy.to(device).requires_grad_(True)
        self.xy_b = self.xy_b.to(device)
        return self
        
    def exact_solution(self, xy):
        return torch.sin(np.pi * xy[:, 0:1]) * torch.sin(np.pi * xy[:, 1:2])
    
    def source_term(self, xy):
        return -2 * (np.pi**2) * torch.sin(np.pi * xy[:, 0:1]) * torch.sin(np.pi * xy[:, 1:2])

# ==========================================
# 3. TRAINING ENGINES
# ==========================================
def compute_loss(model, pde):
    u = model(pde.xy)
    
    # Gradients
    grads = torch.autograd.grad(u, pde.xy, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x, u_y = grads[:, 0:1], grads[:, 1:2]
    
    u_xx = torch.autograd.grad(u_x, pde.xy, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
    u_yy = torch.autograd.grad(u_y, pde.xy, grad_outputs=torch.ones_like(u_y), create_graph=True)[0][:, 1:2]
    
    f = pde.source_term(pde.xy)
    loss_f = torch.mean((u_xx + u_yy - f) ** 2)
    
    u_b = model(pde.xy_b)
    loss_b = torch.mean(u_b ** 2)
    
    return loss_f + 100.0 * loss_b

# MLP uses Adam
def train_adam(model, pde, name, epochs=2000):
    print(f"\n--- Training {name} (Adam) ---")
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
    
    history = {'step': [], 'loss': [], 'error': []}
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = compute_loss(model, pde)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if epoch % 100 == 0:
            with torch.no_grad():
                u_pred = model(pde.xy)
                u_true = pde.exact_solution(pde.xy)
                l2_err = torch.sqrt(torch.mean((u_pred - u_true)**2)).item()
            
            history['step'].append(epoch)
            history['loss'].append(loss.item())
            history['error'].append(l2_err)
            print(f"Epoch {epoch} | Loss: {loss.item():.2e} | L2 Error: {l2_err:.2e}")
    return history

# KAN uses L-BFGS + Grid Extension
def train_lbfgs_kan(model, pde, name, total_steps=100):
    print(f"\n--- Training {name} (L-BFGS + Grid Ext) ---")
    
    # Init L-BFGS
    optimizer = optim.LBFGS(model.parameters(), lr=1, history_size=10, line_search_fn="strong_wolfe")
    
    history = {'step': [], 'loss': [], 'error': []}
    
    grid_schedule = {20: 10, 50: 20} 
    
    for step in range(total_steps):
        
        if step in grid_schedule:
            new_g = grid_schedule[step]
            print(f" Step {step}: Extending Grid to G={new_g}...")
            model.extend_grid(new_g)
            optimizer = optim.LBFGS(model.parameters(), lr=5e-2, history_size=10, line_search_fn="strong_wolfe")

        def closure():
            optimizer.zero_grad()
            loss = compute_loss(model, pde)
            loss.backward()
            return loss
        
        optimizer.step(closure)
        
        if step % 5 == 0:
            loss_val = compute_loss(model, pde).item()
            with torch.no_grad():
                u_pred = model(pde.xy)
                u_true = pde.exact_solution(pde.xy)
                l2_err = torch.sqrt(torch.mean((u_pred - u_true)**2)).item()
            
            history['step'].append(step) 
            history['loss'].append(loss_val)
            history['error'].append(l2_err)
            print(f"Step {step} | Loss: {loss_val:.2e} | L2 Error: {l2_err:.2e}")

    return history

def plot_comparisons(hist_mlp, hist_kan):
    plt.figure(figsize=(10, 6))
    
    # MLP (Adam)
    plt.semilogy(hist_mlp['error'], label='MLP (Adam)', color='gray', linestyle='--')
    
    # KAN (L-BFGS) - stretched to match width
    kan_x = np.linspace(0, len(hist_mlp['error']), len(hist_kan['error']))
    plt.semilogy(kan_x, hist_kan['error'], label='KAN (L-BFGS + Grid Ext)', color='blue', linewidth=2)
    
    plt.title("Fair Fight: MLP vs KAN (No Basis Trick)")
    plt.xlabel("Training Duration")
    plt.ylabel("L2 Relative Error")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("pde_fair_fight.png")
    print(" Comparison plot saved.")

if __name__ == "__main__":
    pde = PoissonPDE().to_device(device)
    
    mlp = MLP().to(device)
    hist_mlp = train_adam(mlp, pde, "MLP", epochs=2000)
    
    kan = KAN([2, 5, 1], grid_size=5).to(device)
    hist_kan = train_lbfgs_kan(kan, pde, "KAN", total_steps=200)
    
    plot_comparisons(hist_mlp, hist_kan)
