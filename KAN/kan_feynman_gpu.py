import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os

plt.switch_backend('Agg') 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available():
    # SETTING MEMORY LIMIT: 30% of RTX 6000 Ada (approx ~14GB, plenty for this)
    fraction = 0.3
    torch.cuda.set_per_process_memory_fraction(fraction, device=0)
    print(f"✅ GPU Detected: {torch.cuda.get_device_name(0)}")
    print(f"🛡️  Memory limited to {fraction*100}% to protect shared resources.")
else:
    print("⚠️  Running on CPU (Slow).")

# ==========================================
# 2. DATASET: FEYNMAN RELATIVISTIC VELOCITY
# ==========================================
class FeynmanDataset:
    def __init__(self, num_samples=10000):
        # u and v are real physics variables
        self.u = torch.rand(num_samples, 1) * 1.8 - 0.9
        self.v = torch.rand(num_samples, 1) * 1.8 - 0.9
        
        # z is PURE NOISE (The dummy variable)
        self.z = torch.rand(num_samples, 1) * 1.8 - 0.9
        
        # Ground Truth still only depends on u and v
        self.y = (self.u + self.v) / (1 + self.u * self.v)
        
        # Input is now 3 dimensions: [u, v, z]
        self.x = torch.cat([self.u, self.v, self.z], dim=1)

    def to_device(self, device):
        self.x = self.x.to(device)
        self.y = self.y.to(device)
        return self

# ==========================================
# 3. VECTORIZED KAN LAYER (Replaces spline.py)
# ==========================================
class KANLinear(nn.Module):
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3, 
                 scale_noise=0.1, scale_base=1.0, scale_spline=1.0, 
                 grid_range=[-1, 1]):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        # 1. Create Grid
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = ((torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0])
                .expand(in_features, -1).contiguous())
        self.register_buffer("grid", grid)

        # 2. Learnable Weights
        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(torch.Tensor(out_features, in_features, grid_size + spline_order))
        self.spline_scaler = nn.Parameter(torch.ones(out_features, in_features)) 

        self.reset_parameters(scale_noise, scale_base, scale_spline)

    def reset_parameters(self, scale_noise, scale_base, scale_spline):
        nn.init.kaiming_uniform_(self.base_weight, a=np.sqrt(5) * scale_base)
        with torch.no_grad():
            # FIX 1: Noise shape matches grid_size + 1
            noise = (torch.rand(self.grid_size + 1, self.in_features, self.out_features) - 0.5) * scale_noise / self.grid_size
            
            self.spline_weight.data.copy_(
                (scale_spline if scale_spline is not None else 1.0) * self.curve2coeff(self.grid.T[self.spline_order : -self.spline_order], noise)
            )

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
        # Base (SiLU) path
        base_output = F.linear(F.silu(x), self.base_weight)
        
        # B-Splines
        bs = self.b_splines(x) 
        
        # FIX 2: Apply the scaler to the weights *before* the matrix multiplication
        # scaler: (out, in) -> unsqueeze -> (out, in, 1) to broadcast over grid points
        scaled_spline_weight = self.spline_weight * self.spline_scaler.unsqueeze(-1)
        
        # Efficient Matrix Multiplication (Einstein Summation)
        # b=batch, i=input_dim, j=grid_points, o=output_dim
        spline_output = torch.einsum('bij,oij->bo', bs, scaled_spline_weight)
        
        return base_output + spline_output

    def regularization_loss(self):
        return self.spline_scaler.abs().mean()

class KAN(nn.Module):
    def __init__(self, layers_hidden):
        super(KAN, self).__init__()
        self.layers = nn.ModuleList()
        # Create layers: e.g., [2, 5, 1] -> Layer(2,5), Layer(5,1)
        for i in range(len(layers_hidden) - 1):
            self.layers.append(KANLinear(layers_hidden[i], layers_hidden[i+1]))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def get_reg_loss(self):
        return sum(layer.regularization_loss() for layer in self.layers)

# ==========================================
# 4. TRAINING & INTERPRETABILITY
# ==========================================
def train_and_visualize():
    # A. Setup
    print("\n--- Generating Feynman Dataset ---")
    data = FeynmanDataset(num_samples=5000)
    data.to_device(device)
    
    # Paper Architecture: [Input=2, Hidden=5, Output=1]
    # Small grid_size=5 is usually enough for smooth physics functions
    model = KAN([3, 5, 1]).to(device)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    criterion = nn.MSELoss()
    
    epochs = 1500
    lambda_l1 = 0.01 # Sparsity penalty strength
    
    history = []

    print(f"--- Starting Training on {device} ---")
    
    # B. Training Loop
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(data.x)
        
        mse = criterion(pred, data.y)
        reg = model.get_reg_loss()
        loss = mse + lambda_l1 * reg
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if epoch % 100 == 0:
            history.append(mse.item())
            print(f"Epoch {epoch:04d} | MSE: {mse.item():.2e} | L1 Reg: {reg.item():.2e}")

    print("--- Training Complete ---")

    # C. Visualization (Saved to disk)
    print("--- Generating Plots (Headless) ---")
    model.eval()
    
    # 1. Prediction Plot
    plt.figure(figsize=(10, 5))
    
    # Move to CPU for plotting
    y_true = data.y.cpu().detach().numpy().flatten()
    y_pred = model(data.x).cpu().detach().numpy().flatten()
    
    plt.subplot(1, 2, 1)
    plt.plot(history)
    plt.title('Training MSE Loss')
    plt.yscale('log')
    plt.xlabel('Epochs (x100)')
    
    plt.subplot(1, 2, 2)
    plt.scatter(y_true, y_pred, alpha=0.1, s=1, c='blue')
    # Perfect fit line
    min_val, max_val = min(y_true), max(y_true)
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.title(f'Feynman: Relativistic Velocity\nCorr: {np.corrcoef(y_true, y_pred)[0,1]:.4f}')
    plt.xlabel('True')
    plt.ylabel('Predicted')
    
    plt.tight_layout()
    plt.savefig('kan_results.png')
    print("✅ Saved 'kan_results.png'")
    
    # D. Interpretability Showcase (Sparsity)
    print("\n--- INTERPRETABILITY: Sparsity Analysis ---")
    print("In KANs, if a connection weight is close to 0, it means that variable is NOT used.")
    
    np.set_printoptions(precision=2, suppress=True)
    
    # Check Layer 1 (Inputs u,v -> Hidden Neurons)
    # Shape: [Out_features, In_features]
    l1_weights = model.layers[0].spline_scaler.data.cpu().numpy()
    
    print("\nLayer 1 Weights (Rows=Hidden Nodes, Cols=Input u,v):")
    print(l1_weights)
    print("\nInterpretation:")
    print("- High values (>0.1) mean the input is active for that node.")
    print("- Near-zero values mean the network 'pruned' that connection.")

if __name__ == "__main__":
    train_and_visualize()
