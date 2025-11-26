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
# 2. DATASET (Physics + Noise)
# ==========================================
class FeynmanDataset:
    def __init__(self, num_samples=10000, mode='train'):
        if mode == 'test': torch.manual_seed(42)
        
        # u, v = Physics. z = Noise.
        self.u = torch.rand(num_samples, 1) * 1.8 - 0.9
        self.v = torch.rand(num_samples, 1) * 1.8 - 0.9
        self.z = torch.rand(num_samples, 1) * 1.8 - 0.9 
        
        # Target: f = (u+v)/(1+uv)
        self.y = (self.u + self.v) / (1 + self.u * self.v)
        self.x = torch.cat([self.u, self.v, self.z], dim=1)
        
    def to_device(self, device):
        self.x = self.x.to(device)
        self.y = self.y.to(device)
        return self

# ==========================================
# 3. KAN ARCHITECTURE
# ==========================================
class KANLinear(nn.Module):
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
        # The logic: If scaler < threshold, kill the connection entirely
        with torch.no_grad():
            mask = (self.spline_scaler.abs() > threshold).float()
            self.spline_scaler.mul_(mask)
            self.base_weight.mul_(mask)

class KAN(nn.Module):
    def __init__(self, layers_hidden):
        super(KAN, self).__init__()
        self.layers = nn.ModuleList()
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
# 4. TRAINING LOGIC
# ==========================================
def train_experiment(model, train_data, test_data, label, do_pruning=False):
    print(f"\n--- Starting Experiment: {label} ---")
    optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    criterion = nn.MSELoss()
    
    history = {'epoch': [], 'loss': []}
    
    # PHASE 1: COARSE TRAINING (With L1 Regularization)
    # We train for 1000 epochs to let the model figure out structure
    epochs_phase1 = 1000
    
    for epoch in range(epochs_phase1):
        model.train()
        optimizer.zero_grad()
        pred = model(train_data.x)
        mse = criterion(pred, train_data.y)
        
        # L1 Regularization encourages sparsity
        reg = model.get_reg_loss()
        loss = mse + 0.01 * reg 
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(test_data.x), test_data.y).item()
            history['epoch'].append(epoch)
            history['loss'].append(val_loss)
            if epoch % 100 == 0: print(f"Ep {epoch} | Val MSE: {val_loss:.2e}")

    # PHASE 2: PRUNING (Optional)
    if do_pruning:
        print("✂️  Applying Pruning Mask (Threshold 0.1)...")
        model.prune(threshold=0.1)
        
        print("🔧 Fine-tuning (No Regularization)...")
        # Reset optimizer for fresh start on fine-tuning
        optimizer_ft = optim.AdamW(model.parameters(), lr=0.01)
        
        # Train for another 500 epochs purely on MSE (No L1 penalty)
        for epoch in range(500):
            model.train()
            optimizer_ft.zero_grad()
            pred = model(train_data.x)
            mse = criterion(pred, train_data.y)
            loss = mse # Note: No +0.01*reg here!
            loss.backward()
            optimizer_ft.step()
            
            if epoch % 10 == 0:
                model.eval()
                with torch.no_grad():
                    val_loss = criterion(model(test_data.x), test_data.y).item()
                history['epoch'].append(epochs_phase1 + epoch)
                history['loss'].append(val_loss)
                if epoch % 100 == 0: print(f"FineTune Ep {epoch} | Val MSE: {val_loss:.2e}")
    else:
        # If no pruning, just keep training the "old way" for comparison
        print("➡️  Continuing without pruning...")
        for epoch in range(500):
            model.train()
            optimizer.zero_grad()
            pred = model(train_data.x)
            mse = criterion(pred, train_data.y)
            loss = mse + 0.01 * model.get_reg_loss() # Keep punishing weights
            loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0:
                model.eval()
                with torch.no_grad():
                    val_loss = criterion(model(test_data.x), test_data.y).item()
                history['epoch'].append(epochs_phase1 + epoch)
                history['loss'].append(val_loss)

    return history

# ==========================================
# 5. EXECUTION
# ==========================================
if __name__ == "__main__":
    train_data = FeynmanDataset(10000, mode='train').to_device(device)
    test_data = FeynmanDataset(2000, mode='test').to_device(device)
    
    # Experiment 1: Unpruned
    model_unpruned = KAN([3, 5, 1]).to(device)
    hist_unpruned = train_experiment(model_unpruned, train_data, test_data, "Unpruned KAN", do_pruning=False)
    
    # Experiment 2: Pruned
    model_pruned = KAN([3, 5, 1]).to(device)
    hist_pruned = train_experiment(model_pruned, train_data, test_data, "Pruned KAN", do_pruning=True)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(hist_unpruned['epoch'], hist_unpruned['loss'], label="Unpruned KAN", color='orange', alpha=0.8)
    plt.plot(hist_pruned['epoch'], hist_pruned['loss'], label="Pruned & Finetuned KAN", color='blue', linewidth=2)
    
    plt.axvline(x=1000, color='red', linestyle=':', label='Pruning Point')
    
    plt.yscale('log')
    plt.title("Impact of Pruning on Relativistic Velocity Discovery")
    plt.xlabel("Epochs")
    plt.ylabel("Validation MSE (Log Scale)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("kan_pruning.png")
    print("✅ Results saved to kan_pruning.png")
