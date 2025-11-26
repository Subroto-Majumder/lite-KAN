import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# ==========================================
# 1. SETUP
# ==========================================
plt.switch_backend('Agg') 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.3, device=0)
    print(f"✅ Running on {torch.cuda.get_device_name(0)}")

# ==========================================
# 2. DATASET (Complex Nested Function)
# ==========================================
class ComplexDataset:
    def __init__(self, num_samples=5000, mode='train'):
        if mode == 'test': torch.manual_seed(42)
        
        # 4 Inputs: x1, x2 (Signal), x3, x4 (Noise)
        self.inputs = torch.rand(num_samples, 4) * 2 - 1 # Range [-1, 1]
        
        x1 = self.inputs[:, 0:1]
        x2 = self.inputs[:, 1:2]
        
        # Equation: exp( sin(pi*x1) + x2^2 )
        # This requires capturing periodicity AND exponential growth
        inner = torch.sin(np.pi * x1) + x2**2
        self.y = torch.exp(inner)
        
        # Normalize target to [0,1] range to help training stability
        self.y = self.y / self.y.max()
        
        self.x = self.inputs # Includes x3, x4 noise columns
        
    def to_device(self, device):
        self.x = self.x.to(device)
        self.y = self.y.to(device)
        return self

# ==========================================
# 3. THE ULTIMATE KAN LAYER
# ==========================================
class KANLayer(nn.Module):
    def __init__(self, in_features, out_features, grid_size=3, spline_order=3, scale_noise=0.1, scale_base=1.0, scale_spline=1.0, grid_range=[-1, 1]):
        super(KANLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.grid_range = grid_range

        # Initialize Grid
        self.update_grid_buffer(grid_size, device='cpu') # Temp cpu

        # Weights
        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(torch.Tensor(out_features, in_features, grid_size + spline_order))
        self.spline_scaler = nn.Parameter(torch.ones(out_features, in_features)) 

        self.reset_parameters(scale_noise, scale_base, scale_spline)
        self.acts = None # For regularization tracking

    def update_grid_buffer(self, grid_size, device='cpu'):
        h = (self.grid_range[1] - self.grid_range[0]) / grid_size
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
        A = self.b_splines(x).transpose(0, 1)
        B = y.transpose(0, 1)
        solution = torch.linalg.lstsq(A, B).solution
        return solution.permute(2, 0, 1).contiguous()

    def forward(self, x):
        base_output = F.linear(F.silu(x), self.base_weight)
        bs = self.b_splines(x)
        
        # Calculate scaled weights. 
        # Crucial: Pruned connections have spline_scaler=0, effectively killing the spline.
        scaled_weight = self.spline_weight * self.spline_scaler.unsqueeze(-1)
        
        self.acts = torch.einsum('bij,oij->boi', bs, scaled_weight)
        return base_output + self.acts.sum(dim=2)

    # --- PRUNING FUNCTION ---
    def prune(self, threshold):
        # We prune based on the scaler magnitude.
        # If the scaler is 0, the weight update (during grid extension) won't matter 
        # because scaler=0 multiplies everything to 0.
        with torch.no_grad():
            mask = (self.spline_scaler.abs() > threshold).float()
            self.spline_scaler.mul_(mask)
            self.base_weight.mul_(mask)
            
            # Return number of pruned edges
            return (self.spline_scaler == 0).sum().item()

    # --- GRID EXTENSION FUNCTION ---
    def extend_grid(self, new_grid_size):
        current_device = self.grid.device
        
        # Sample the OLD curve
        num_samples = 1000 
        x_sample = torch.linspace(self.grid_range[0], self.grid_range[1], num_samples, device=current_device)
        x_sample = x_sample.unsqueeze(1).expand(-1, self.in_features)
        
        with torch.no_grad():
            old_bs = self.b_splines(x_sample)
            # Use raw weights here to capture shape. 
            # The scaler (mask) is preserved separately!
            old_spline_out = torch.einsum('bij,oij->boi', old_bs, self.spline_weight)
            y_target = old_spline_out.permute(0, 2, 1) 

        # Create new grid parameters
        self.grid_size = new_grid_size
        self.update_grid_buffer(new_grid_size, device=current_device)
        self.spline_weight = nn.Parameter(torch.Tensor(self.out_features, self.in_features, new_grid_size + self.spline_order).to(current_device))
        
        # Fit new weights
        with torch.no_grad():
            new_coeffs = self.curve2coeff(x_sample, y_target)
            self.spline_weight.data.copy_(new_coeffs)

class KAN(nn.Module):
    def __init__(self, layers_hidden, initial_grid=3):
        super(KAN, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers_hidden) - 1):
            self.layers.append(KANLayer(layers_hidden[i], layers_hidden[i+1], grid_size=initial_grid))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def extend_grid(self, new_grid_size):
        for layer in self.layers:
            layer.extend_grid(new_grid_size)
            
    def prune(self, threshold=1e-2):
        total_pruned = 0
        for layer in self.layers:
            total_pruned += layer.prune(threshold)
        return total_pruned

    def get_reg_loss(self):
        # Simple L1 on scalers for coarsening phase
        return sum(layer.spline_scaler.abs().mean() for layer in self.layers)

# ==========================================
# 4. THE FULL PIPELINE
# ==========================================
def run_pipeline(model, train_data, test_data):
    print(f"\n--- Starting Full KAN Pipeline (Prune + Extend) ---")
    
    # Optimizer (Adam is safer for Extension than L-BFGS)
    optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    
    def get_rmse(pred, target): return torch.sqrt(torch.mean((pred - target) ** 2))
    
    history = {'epoch': [], 'rmse': []}
    
    # SCHEDULE
    # 0-300:   Grid=3  (Coarse Training + L1 Reg)
    # 300:     PRUNE   (Cut Noise)
    # 300-600: Grid=10 (Medium Training + Pure RMSE)
    # 600:     EXTEND  (G=10 -> G=20)
    # 600-900: Grid=20 (Fine Tuning)
    
    total_epochs = 900
    
    for epoch in range(total_epochs):
        optimizer.zero_grad()
        pred = model(train_data.x)
        rmse = get_rmse(pred, train_data.y)
        
        # Phase 1: Regularization ON
        if epoch < 300:
            reg = model.get_reg_loss()
            loss = rmse + 0.05 * reg # Strong Reg to find structure
        else:
            loss = rmse # Pure fitting
            
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # --- EVENT: PRUNING (Epoch 300) ---
        if epoch == 300:
            print(f"\n✂️  Epoch {epoch}: Pruning & Extending to Grid=10...")
            n_pruned = model.prune(threshold=0.05)
            print(f"   Pruned {n_pruned} connections.")
            
            # First extension (Coarse -> Medium)
            model.extend_grid(10)
            
            # Reset Optimizer
            optimizer = optim.AdamW(model.parameters(), lr=0.01)
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

        # --- EVENT: FINE EXTENSION (Epoch 600) ---
        if epoch == 600:
            print(f"\n⚡ Epoch {epoch}: Extending to Grid=20 (High Res)...")
            model.extend_grid(20)
            
            # Reset Optimizer (Lower LR for fine-tuning)
            optimizer = optim.AdamW(model.parameters(), lr=0.005)
        
        # Logging
        if epoch % 20 == 0:
            with torch.no_grad():
                test_rmse = get_rmse(model(test_data.x), test_data.y)
            history['epoch'].append(epoch)
            history['rmse'].append(test_rmse.item())
            
            if epoch % 100 == 0:
                print(f"Ep {epoch} | RMSE: {test_rmse:.2e} | Grid: {model.layers[0].grid_size}")

    return history

def plot_final_arch(model, filename="kan_complex_arch.png"):
    # Visualize the pruned structure
    G = nx.Graph()
    pos = {}
    inputs = ['x1', 'x2', 'x3 (noise)', 'x4 (noise)']
    
    # Nodes
    for i, name in enumerate(inputs):
        G.add_node(name, layer=0); pos[name] = (0, -i)
    for i in range(5):
        G.add_node(f"h{i}", layer=1); pos[f"h{i}"] = (1, -i*0.8)
    G.add_node("f", layer=2); pos["f"] = (2, -1.5)
    
    # Edges
    colors, widths = [], []
    w_in = model.layers[0].spline_scaler.detach().cpu().numpy()
    
    for h in range(5):
        for i in range(4):
            if w_in[h,i] > 0.01:
                G.add_edge(inputs[i], f"h{h}")
                colors.append('blue'); widths.append(w_in[h,i]*2)
                
    w_out = model.layers[1].spline_scaler.detach().cpu().numpy()
    for h in range(5):
        if w_out[0,h] > 0.01:
            G.add_edge(f"h{h}", "f")
            colors.append('blue'); widths.append(w_out[0,h]*2)
            
    plt.figure(figsize=(8,6))
    nx.draw_networkx(G, pos, node_color='lightgray', edge_color=colors, width=widths)
    plt.title("Learned Architecture (Active Paths)")
    plt.axis('off')
    plt.savefig(filename)
    print("✅ Diagram saved.")

# ==========================================
# 5. RUN
# ==========================================
if __name__ == "__main__":
    train_data = ComplexDataset(5000, mode='train').to_device(device)
    test_data = ComplexDataset(1000, mode='test').to_device(device)
    
    # Input: 4 dims. Hidden: 5 dims. Output: 1 dim.
    # Start with VERY COARSE grid (3) to learn broad structure
    model = KAN([4, 5, 1], initial_grid=3).to(device)
    
    history = run_pipeline(model, train_data, test_data)
    
    # Plot Loss
    plt.figure(figsize=(10,6))
    plt.plot(history['epoch'], history['rmse'], linewidth=2)
    plt.yscale('log')
    plt.axvline(300, color='r', linestyle='--', label='Prune & Extend G=10')
    plt.axvline(600, color='g', linestyle='--', label='Extend G=20')
    plt.title(r"KAN Training: $f = \exp(\sin(\pi x_1) + x_2^2)$")
    plt.ylabel("RMSE")
    plt.xlabel("Epochs")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("kan_pipeline_loss.png")
    
    plot_final_arch(model)
