import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. SETUP
# ==========================================
plt.switch_backend('Agg')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.3, device=0)

# ==========================================
# 2. DATASET
# ==========================================
class FeynmanDataset:
    def __init__(self, num_samples=10000):
        self.u = torch.rand(num_samples, 1) * 2 - 1
        self.v = torch.rand(num_samples, 1) * 2 - 1
        self.z = torch.rand(num_samples, 1) * 2 - 1 # Noise
        
        denom = 1 + self.u * self.v
        # Handle singularity slightly more aggressively for stability
        denom = torch.where(denom.abs() < 0.05, torch.sign(denom)*0.05, denom) 
        self.y = (self.u + self.v) / denom
        self.x = torch.cat([self.u, self.v, self.z], dim=1)
        
    def to_device(self, device):
        self.x = self.x.to(device)
        self.y = self.y.to(device)
        return self

# ==========================================
# 3. COMPETITOR 1: MLP
# ==========================================
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        # 3 -> 128 -> 128 -> 1
        self.net = nn.Sequential(
            nn.Linear(3, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, x):
        return self.net(x)

# ==========================================
# 4. COMPETITOR 2: OPTIMIZED KAN
# ==========================================
class KANLinear(nn.Module):
    # OPTIMIZATION 1: Increased grid_size default to 10 (better precision)
    def __init__(self, in_features, out_features, grid_size=10, spline_order=3, scale_noise=0.1, scale_base=1.0, scale_spline=1.0, grid_range=[-1, 1]):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = ((torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0]).expand(in_features, -1).contiguous())
        self.register_buffer("grid", grid)

        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(torch.Tensor(out_features, in_features, grid_size + spline_order))
        self.spline_scaler = nn.Parameter(torch.ones(out_features, in_features)) 

        self.reset_parameters(scale_noise, scale_base, scale_spline)

    def reset_parameters(self, scale_noise, scale_base, scale_spline):
        nn.init.kaiming_uniform_(self.base_weight, a=np.sqrt(5) * scale_base)
        with torch.no_grad():
            noise = (torch.rand(self.grid_size + 1, self.in_features, self.out_features) - 0.5) * scale_noise / self.grid_size
            self.spline_weight.data.copy_((scale_spline if scale_spline is not None else 1.0) * self.curve2coeff(self.grid.T[self.spline_order : -self.spline_order], noise))

    def b_splines(self, x):
        assert x.dim() == 2 and x.size(1) == self.in_features
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
        scaled_spline_weight = self.spline_weight * self.spline_scaler.unsqueeze(-1)
        spline_output = torch.einsum('bij,oij->bo', bs, scaled_spline_weight)
        return base_output + spline_output

    def prune_connections(self, threshold):
        with torch.no_grad():
            mask = (self.spline_scaler.abs() > threshold).float()
            self.spline_scaler.mul_(mask)
            self.base_weight.mul_(mask)

class KAN(nn.Module):
    def __init__(self, layers_hidden):
        super(KAN, self).__init__()
        self.layers = nn.ModuleList()
        # Increased grid size passed here
        for i in range(len(layers_hidden) - 1):
            self.layers.append(KANLinear(layers_hidden[i], layers_hidden[i+1], grid_size=10))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def get_reg_loss(self):
        return sum(layer.spline_scaler.abs().mean() for layer in self.layers)
    
    def prune(self, threshold=1e-2):
        for layer in self.layers:
            layer.prune_connections(threshold)

# ==========================================
# 5. TRAINING ROUTINE
# ==========================================
def train_model(model, data, name, epochs=1000, prune=False):
    print(f"\n--- Training {name} ---")
    optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-5) # Reduced weight decay
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    criterion = nn.MSELoss()
    
    history = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(data.x)
        mse = criterion(pred, data.y)
        
        loss = mse
        
        # OPTIMIZATION 2: Only apply regularization if NOT finely tuning
        if "KAN" in name:
            reg = model.get_reg_loss()
            # Strong regularization initially to force sparsity
            loss += 0.01 * reg 
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if epoch % 100 == 0:
            history.append(mse.item())
            print(f"Epoch {epoch} | MSE: {mse.item():.2e}")

    if prune and "KAN" in name:
        print("✂️  Pruning weak connections (Threshold < 0.1)...")
        model.prune(threshold=0.1)
        
        print("🔧 Fine-tuning (Regularization OFF)...")
        # Reset optimizer for fine-tuning with fresh learning rate
        optimizer_ft = optim.AdamW(model.parameters(), lr=0.01) 
        
        for epoch in range(500): # More epochs for fine-tuning
            optimizer_ft.zero_grad()
            pred = model(data.x)
            mse = criterion(pred, data.y)
            
            # OPTIMIZATION 3: PURE MSE LOSS. No Regularization.
            # This allows the remaining params to fit perfectly.
            loss = mse 
            
            loss.backward()
            optimizer_ft.step()
            
            if epoch % 100 == 0:
                print(f"Fine-tune Epoch {epoch} | MSE: {mse.item():.2e}")
                
        print(f"Final Fine-Tuned MSE: {mse.item():.2e}")
        history.append(mse.item())

    return history, model

if __name__ == "__main__":
    # Generate Data
    data = FeynmanDataset(10000).to_device(device)
    
    # 1. Train MLP
    mlp = MLP().to(device)
    mlp_hist, _ = train_model(mlp, data, "MLP (Baseline)")
    
    # 2. Train KAN
    # Note: KAN is initialized with grid_size=10 inside the class now
    kan = KAN([3, 5, 1]).to(device)
    kan_hist, _ = train_model(kan, data, "KAN (Pruned)", prune=True)

    # 3. Plot
    plt.figure(figsize=(10, 6))
    plt.plot(mlp_hist, label=f'MLP (Final: {mlp_hist[-1]:.2e})', linestyle='--', color='gray')
    plt.plot(kan_hist, label=f'KAN (Final: {kan_hist[-1]:.2e})', color='blue', linewidth=2)
    plt.axvline(x=len(kan_hist)-1, color='red', linestyle=':', label='Pruning Start')
    
    plt.yscale('log')
    plt.title("Comparison: MLP vs KAN (Optimized)")
    plt.xlabel("Epochs (x100)")
    plt.ylabel("MSE Loss (Log Scale)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("kan_vs_mlp_v2.png")
    print("\n✅ Saved 'kan_vs_mlp_v2.png'")
    
    print(f"\n--- Stats ---")
    print(f"MLP Parameters: {sum(p.numel() for p in mlp.parameters())}")
    print(f"KAN Parameters: {sum(p.numel() for p in kan.parameters())}")
