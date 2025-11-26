import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. CONFIGURATION
# ==========================================
plt.switch_backend('Agg') 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.3, device=0)
    print(f"✅ Running on {torch.cuda.get_device_name(0)}")

# ==========================================
# 2. DATASET
# ==========================================
class FeynmanDataset:
    def __init__(self, num_samples=2000, mode='train'):
        if mode == 'test': torch.manual_seed(42)
        
        self.u = torch.rand(num_samples, 1) * 1.8 - 0.9
        self.v = torch.rand(num_samples, 1) * 1.8 - 0.9
        self.z = torch.rand(num_samples, 1) * 1.8 - 0.9 
        
        self.y = (self.u + self.v) / (1 + self.u * self.v)
        self.x = torch.cat([self.u, self.v, self.z], dim=1)
        
    def to_device(self, device):
        self.x = self.x.to(device)
        self.y = self.y.to(device)
        return self

# ==========================================
# 3. KAN LAYER WITH GRID EXTENSION
# ==========================================
class KANLinearExtended(nn.Module):
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3, scale_noise=0.1, scale_base=1.0, scale_spline=1.0, grid_range=[-1, 1]):
        super(KANLinearExtended, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.grid_range = grid_range

        # Initialize Grid (Default to CPU initially, moved to GPU later by .to(device))
        self.update_grid_buffer(grid_size, device='cpu')

        # Weights
        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(torch.Tensor(out_features, in_features, grid_size + spline_order))
        self.spline_scaler = nn.Parameter(torch.ones(out_features, in_features)) 

        self.reset_parameters(scale_noise, scale_base, scale_spline)
        self.acts = None

    def update_grid_buffer(self, grid_size, device='cpu'):
        h = (self.grid_range[1] - self.grid_range[0]) / grid_size
        # FIX: We now pass the 'device' argument to torch.arange
        grid = ((torch.arange(-self.spline_order, grid_size + self.spline_order + 1, device=device) * h + self.grid_range[0]).expand(self.in_features, -1).contiguous())
        self.register_buffer("grid", grid)

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
        A = self.b_splines(x).transpose(0, 1) # (In, Batch, Coeffs)
        B = y.transpose(0, 1)                 # (In, Batch, Out)
        solution = torch.linalg.lstsq(A, B).solution
        return solution.permute(2, 0, 1).contiguous()

    def forward(self, x):
        base_output = F.linear(F.silu(x), self.base_weight)
        bs = self.b_splines(x)
        scaled_weight = self.spline_weight * self.spline_scaler.unsqueeze(-1)
        self.acts = torch.einsum('bij,oij->boi', bs, scaled_weight)
        return base_output + self.acts.sum(dim=2)

    def extend_grid(self, new_grid_size):
        # 1. Sample x values to capture the shape of the OLD spline
        # FIX: Ensure samples are on the current GPU
        current_device = self.grid.device
        num_samples = 1000 
        x_sample = torch.linspace(self.grid_range[0], self.grid_range[1], num_samples, device=current_device)
        x_sample = x_sample.unsqueeze(1).expand(-1, self.in_features)
        
        # 2. Compute the output of the OLD spline at these points
        with torch.no_grad():
            old_bs = self.b_splines(x_sample)
            old_spline_out = torch.einsum('bij,oij->boi', old_bs, self.spline_weight)
            y_target = old_spline_out.permute(0, 2, 1) 

        # 3. Update Grid Parameters
        self.grid_size = new_grid_size
        # FIX: Pass the current device so the new grid is created on GPU
        self.update_grid_buffer(new_grid_size, device=current_device)
        
        # 4. Create New Parameter Tensor (Ensure it's on the correct device)
        self.spline_weight = nn.Parameter(torch.Tensor(self.out_features, self.in_features, new_grid_size + self.spline_order).to(current_device))
        
        # 5. Least Squares Fit
        with torch.no_grad():
            new_coeffs = self.curve2coeff(x_sample, y_target)
            self.spline_weight.data.copy_(new_coeffs)

class KAN(nn.Module):
    def __init__(self, layers_hidden, initial_grid=5):
        super(KAN, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers_hidden) - 1):
            self.layers.append(KANLinearExtended(layers_hidden[i], layers_hidden[i+1], grid_size=initial_grid))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def extend_grid(self, new_grid_size):
        for layer in self.layers:
            layer.extend_grid(new_grid_size)

# ==========================================
# 4. TRAINING WITH EXTENSION
# ==========================================
def train_with_extension(model, train_data, test_data):
    print(f"\n--- Training with Grid Extension (Adam + RMSE) ---")
    
    # Start with Adam
    optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    
    history = {'epoch': [], 'rmse': [], 'grid_size': []}
    
    def get_rmse(pred, target):
        return torch.sqrt(torch.mean((pred - target) ** 2))

    # Schedule:
    # 0-200: Grid 5 (Coarse)
    # 200: Extend to 10
    # 200-400: Grid 10 (Medium)
    # 400: Extend to 20
    # 400-600: Grid 20 (Fine)
    
    total_epochs = 600
    grid_updates = {200: 10, 400: 20}
    
    for epoch in range(total_epochs):
        
        # Check for Grid Extension
        if epoch in grid_updates:
            new_g = grid_updates[epoch]
            print(f"\n⚡ Extending Grid to G={new_g}...")
            model.extend_grid(new_g)
            
            # IMPORTANT: Re-initialize optimizer because parameter shapes changed!
            optimizer = optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-5)
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
            print("   Model resized successfully.")

        optimizer.zero_grad()
        pred = model(train_data.x)
        rmse = get_rmse(pred, train_data.y)
        loss = rmse # Pure RMSE training
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if epoch % 10 == 0:
            with torch.no_grad():
                test_rmse = get_rmse(model(test_data.x), test_data.y)
            
            current_grid = model.layers[0].grid_size
            history['epoch'].append(epoch)
            history['rmse'].append(test_rmse.item())
            history['grid_size'].append(current_grid)
            
            if epoch % 50 == 0:
                print(f"Ep {epoch} | Grid {current_grid} | RMSE: {test_rmse:.2e}")

    return history

# ==========================================
# 5. EXECUTION
# ==========================================
if __name__ == "__main__":
    train_data = FeynmanDataset(2000, mode='train').to_device(device)
    test_data = FeynmanDataset(1000, mode='test').to_device(device)
    
    # Start with Coarse Grid G=5
    model = KAN([3, 5, 1], initial_grid=5).to(device)
    
    history = train_with_extension(model, train_data, test_data)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(history['epoch'], history['rmse'], color='blue', linewidth=2, label='Test RMSE')
    
    # Draw vertical lines for extensions
    plt.axvline(x=200, color='red', linestyle='--', label='Extend to G=10')
    plt.axvline(x=400, color='green', linestyle='--', label='Extend to G=20')
    
    plt.yscale('log')
    plt.ylabel("RMSE (Log Scale)")
    plt.xlabel("Epochs")
    plt.title("KAN Grid Extension: Coarse to Fine Training")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.savefig("kan_grid_extension.png")
    print("\n✅ Results saved to kan_grid_extension.png")
