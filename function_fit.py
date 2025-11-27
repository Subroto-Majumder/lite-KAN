import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from kan import KAN

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
# 3. THE FULL PIPELINE
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
# 4. RUN
# ==========================================
if __name__ == "__main__":
    train_data = ComplexDataset(5000, mode='train').to_device(device)
    test_data = ComplexDataset(1000, mode='test').to_device(device)
    
    # Input: 4 dims. Hidden: 5 dims. Output: 1 dim.
    # Start with VERY COARSE grid (3) to learn broad structure
    model = KAN([4, 5, 1], grid_size=3).to(device)
    
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
