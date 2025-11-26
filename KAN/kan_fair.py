import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import time

# ==========================================
# 1. SETUP
# ==========================================
plt.switch_backend('Agg')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.5, device=0)
    print(f"✅ Running on {torch.cuda.get_device_name(0)}")

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
# 3. KAN (NO CHEAT CODE)
# ==========================================
class KANLayer(nn.Module):
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3, scale_base=1.0, scale_spline=1.0, grid_range=[-1, 1]):
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
        
        nn.init.kaiming_uniform_(self.base_weight, a=np.sqrt(5) * scale_base)
        with torch.no_grad():
            noise = (torch.rand(self.grid_size + 1, self.in_features, self.out_features) - 0.5) * 0.1 / self.grid_size
            self.spline_weight.data.copy_(
                (scale_spline if scale_spline is not None else 1.0) * self.curve2coeff(self.grid.T[self.spline_order : -self.spline_order], noise)
            )

    def update_grid_buffer(self, grid_size, device='cpu'):
        h = (self.grid_range[1] - self.grid_range[0]) / grid_size
        grid = ((torch.arange(-self.spline_order, grid_size + self.spline_order + 1, device=device) * h + self.grid_range[0]).expand(self.in_features, -1).contiguous())
        self.register_buffer("grid", grid)

    def b_splines(self, x):
        grid = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = ((x - grid[:, : -(k + 1)]) / (grid[:, k:-1] - grid[:, : -(k + 1)]) * bases[:, :, :-1]) + \
                    ((grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:(-k)]) * bases[:, :, 1:])
        return bases.contiguous()

    def curve2coeff(self, x, y):
        A = self.b_splines(x).transpose(0, 1)
        B = y.transpose(0, 1)
        solution = torch.linalg.lstsq(A, B).solution
        return solution.permute(2, 0, 1).contiguous()

    def forward(self, x):
        base_output = F.linear(F.silu(x), self.base_weight)
        bs = self.b_splines(x)
        scaled_weight = self.spline_weight * self.spline_scaler.unsqueeze(-1)
        spline_output = torch.einsum('bij,oij->boi', bs, scaled_weight).sum(dim=2)
        return base_output + spline_output

    def extend_grid(self, new_grid_size):
        current_device = self.grid.device
        x_sample = torch.linspace(self.grid_range[0], self.grid_range[1], 1000, device=current_device).unsqueeze(1).expand(-1, self.in_features)
        
        with torch.no_grad():
            old_bs = self.b_splines(x_sample)
            old_spline_out = torch.einsum('bij,oij->boi', old_bs, self.spline_weight)
            y_target = old_spline_out.permute(0, 2, 1) 

        self.grid_size = new_grid_size
        self.update_grid_buffer(new_grid_size, device=current_device)
        self.spline_weight = nn.Parameter(torch.Tensor(self.out_features, self.in_features, new_grid_size + self.spline_order).to(current_device))
        
        with torch.no_grad():
            new_coeffs = self.curve2coeff(x_sample, y_target)
            self.spline_weight.data.copy_(new_coeffs)

class KAN(nn.Module):
    def __init__(self, layers_hidden, grid_size=5):
        super(KAN, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers_hidden) - 1):
            self.layers.append(KANLayer(layers_hidden[i], layers_hidden[i+1], grid_size=grid_size))
        
        # --- REMOVED THE CHEAT CODE ---
        # No basis_coeff here! KAN must learn sin(x) from scratch.

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

    def extend_grid(self, new_grid_size):
        for layer in self.layers:
            layer.extend_grid(new_grid_size)

# ==========================================
# 4. MLP BASELINE
# ==========================================
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 1)
        )
    def forward(self, x):
        return self.net(x)

# ==========================================
# 5. TRAINING ENGINES
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
            print(f"⚡ Step {step}: Extending Grid to G={new_g}...")
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
    print("✅ Comparison plot saved.")

if __name__ == "__main__":
    pde = PoissonPDE().to_device(device)
    
    mlp = MLP().to(device)
    hist_mlp = train_adam(mlp, pde, "MLP", epochs=2000)
    
    kan = KAN([2, 5, 1], grid_size=5).to(device)
    hist_kan = train_lbfgs_kan(kan, pde, "KAN", total_steps=200)
    
    plot_comparisons(hist_mlp, hist_kan)
