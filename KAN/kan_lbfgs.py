import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

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
        # Note: L-BFGS usually works best with FULL BATCH (all data at once)
        # So we reduce num_samples slightly to fit in VRAM easily if needed, 
        # but 2000-5000 is fine for this size model.
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
# 3. EXACT KAN LAYER
# ==========================================
class KANLinearExact(nn.Module):
    def __init__(self, in_features, out_features, grid_size=10, spline_order=3, scale_noise=0.1, scale_base=1.0, scale_spline=1.0, grid_range=[-1, 1]):
        super(KANLinearExact, self).__init__()
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
        self.acts = None

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
        scaled_weight = self.spline_weight * self.spline_scaler.unsqueeze(-1)
        self.acts = torch.einsum('bij,oij->boi', bs, scaled_weight)
        spline_output = self.acts.sum(dim=2) 
        return base_output + spline_output

class KANPaper(nn.Module):
    def __init__(self, layers_hidden):
        super(KANPaper, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers_hidden) - 1):
            self.layers.append(KANLinearExact(layers_hidden[i], layers_hidden[i+1], grid_size=10))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def get_paper_reg_loss(self, lamb_l1=1.0, lamb_entropy=1.0):
        total_l1 = 0.0
        total_entropy = 0.0
        for layer in self.layers:
            if layer.acts is None: continue
            edge_norms = layer.acts.abs().mean(dim=0) 
            layer_l1 = edge_norms.sum()
            total_l1 += layer_l1
            if layer_l1 > 1e-6:
                probs = edge_norms / layer_l1
                entropy = - torch.sum(probs * torch.log(probs + 1e-8))
                total_entropy += entropy
        return lamb_l1 * total_l1 + lamb_entropy * total_entropy
    
    def prune_nodes(self, threshold=1e-2):
        print("✂️  Applying Node Pruning...")
        for l in range(len(self.layers) - 1):
            incoming_layer = self.layers[l]
            outgoing_layer = self.layers[l+1]
            in_norms = incoming_layer.acts.abs().mean(dim=0)
            out_norms = outgoing_layer.acts.abs().mean(dim=0)
            score_in = in_norms.max(dim=1).values
            score_out = out_norms.max(dim=0).values
            keep_mask = (score_in > threshold) & (score_out > threshold)
            
            num_pruned = (~keep_mask).sum().item()
            if num_pruned > 0:
                print(f"   Hidden Layer {l}: Pruning {num_pruned} nodes.")
                with torch.no_grad():
                    incoming_layer.spline_scaler.data[~keep_mask, :] = 0.
                    incoming_layer.base_weight.data[~keep_mask, :] = 0.
                    outgoing_layer.spline_scaler.data[:, ~keep_mask] = 0.
                    outgoing_layer.base_weight.data[:, ~keep_mask] = 0.

# ==========================================
# 4. TRAINING ENGINE (L-BFGS + RMSE)
# ==========================================
def train_experiment(model, train_data, test_data, name, do_pruning=False):
    print(f"\n--- Experiment: {name} (L-BFGS + RMSE) ---")
    
    # L-BFGS Config:
    # lr=1.0 is standard for L-BFGS (it uses line search to find real step size)
    # history_size=10 is typical memory limit
    optimizer = optim.LBFGS(model.parameters(), lr=1.0, history_size=10, line_search_fn="strong_wolfe", tolerance_grad=1e-9, tolerance_change=1e-9)
    
    history = {'epoch': [], 'rmse': [], 'reg_loss': []}
    
    # L-BFGS converges FAST. We don't need 1500 epochs. 
    # We will do "steps". One L-BFGS step can involve multiple evaluations.
    total_steps = 100 
    prune_step = 50 # Prune halfway
    
    def get_rmse(pred, target):
        return torch.sqrt(torch.mean((pred - target) ** 2))

    for step in range(total_steps):
        
        # --- L-BFGS CLOSURE ---
        # L-BFGS requires a function that:
        # 1. Clears gradients
        # 2. Computes Loss
        # 3. Backprops
        # 4. Returns Loss
        def closure():
            optimizer.zero_grad()
            pred = model(train_data.x)
            
            # RMSE Loss
            rmse = get_rmse(pred, train_data.y)
            
            # Regularization (Only before pruning)
            if step < prune_step:
                reg_loss = model.get_paper_reg_loss()
                loss = rmse + 0.01 * reg_loss
            else:
                loss = rmse # Pure RMSE for fine-tuning
                
            loss.backward()
            return loss

        # Perform optimization step
        optimizer.step(closure)
        
        # --- LOGGING (Once per step) ---
        with torch.no_grad():
            pred_train = model(train_data.x)
            train_rmse = get_rmse(pred_train, train_data.y)
            reg_val = model.get_paper_reg_loss()
            
            pred_test = model(test_data.x)
            test_rmse = get_rmse(pred_test, test_data.y)
            
        history['epoch'].append(step)
        history['rmse'].append(test_rmse.item())
        history['reg_loss'].append(reg_val.item())
        
        if step % 5 == 0:
            print(f"Step {step} | Train RMSE: {train_rmse:.2e} | Test RMSE: {test_rmse:.2e}")

        # --- PRUNING EVENT ---
        if do_pruning and step == prune_step:
            print("\n🛑 Pausing for Pruning...")
            model.prune_nodes(threshold=1e-2)
            
            # Re-initialize Optimizer? 
            # For L-BFGS, it's safer to keep it or create a new one to reset history
            # But since L-BFGS adapts well, we can just continue, 
            # though resetting cleans the 'memory' of the bad gradients from before pruning.
            print("🔄 Resetting L-BFGS for Fine-tuning...")
            optimizer = optim.LBFGS(model.parameters(), lr=1.0, history_size=10, line_search_fn="strong_wolfe")

    return history

def plot_pruned_architecture(model, filename="kan_pruned_lbfgs.png"):
    print(f"\n--- Generating Diagram: {filename} ---")
    layer0 = model.layers[0]
    layer1 = model.layers[1]
    
    # Identify active nodes
    in_norms = layer0.acts.abs().mean(dim=0)
    out_norms = layer1.acts.abs().mean(dim=0)
    active_indices = torch.nonzero((in_norms.max(dim=1).values > 0.01) & (out_norms.max(dim=0).values > 0.01)).flatten().cpu().numpy()
    
    print(f"Active Hidden Nodes: {active_indices}")

    G = nx.Graph()
    pos = {}
    inputs = ['u', 'v', 'z']
    
    for i, name in enumerate(inputs):
        G.add_node(name, layer=0); pos[name] = (0, -i)
    for i in range(5):
        G.add_node(f"h{i}", layer=1); pos[f"h{i}"] = (1, -i * 0.6)
    G.add_node("f", layer=2); pos["f"] = (2, -1)
    
    colors, widths = [], []
    
    # Edges
    w_in = layer0.spline_scaler.detach().cpu().numpy()
    w_out = layer1.spline_scaler.detach().cpu().numpy()
    
    # Input->Hidden
    for h in range(5):
        for i in range(3):
            strength = w_in[h, i] if h in active_indices else 0
            if strength > 0.01:
                G.add_edge(inputs[i], f"h{h}")
                colors.append('blue'); widths.append(strength * 3)

    # Hidden->Output
    for h in range(5):
        strength = w_out[0, h] if h in active_indices else 0
        if strength > 0.01:
            G.add_edge(f"h{h}", "f")
            colors.append('blue'); widths.append(strength * 3)

    plt.figure(figsize=(8, 6))
    nx.draw_networkx_nodes(G, pos, node_color='lightgray', node_size=500)
    nx.draw_networkx_nodes(G, pos, nodelist=[f"h{i}" for i in active_indices], node_color='lightgreen', node_size=700)
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos, edge_color=colors, width=widths)
    plt.axis('off')
    plt.title("L-BFGS Pruned KAN")
    plt.savefig(filename)
    print("✅ Diagram saved.")

# ==========================================
# 5. EXECUTION
# ==========================================
if __name__ == "__main__":
    train_data = FeynmanDataset(2000, mode='train').to_device(device)
    test_data = FeynmanDataset(1000, mode='test').to_device(device)
    
    # 1. Unpruned
    model_unpruned = KANPaper([3, 5, 1]).to(device)
    hist_unpruned = train_experiment(model_unpruned, train_data, test_data, "Unpruned", do_pruning=False)
    
    # 2. Pruned
    model_pruned = KANPaper([3, 5, 1]).to(device)
    hist_pruned = train_experiment(model_pruned, train_data, test_data, "Pruned", do_pruning=True)
    
    # 3. Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(hist_unpruned['epoch'], hist_unpruned['rmse'], label="Unpruned", color='orange', linestyle='--')
    plt.plot(hist_pruned['epoch'], hist_pruned['rmse'], label="Pruned", color='blue', linewidth=2)
    plt.axvline(x=50, color='red', linestyle=':', label='Pruning')
    plt.yscale('log')
    plt.ylabel("Test RMSE")
    plt.xlabel("L-BFGS Steps")
    plt.legend()
    plt.title("L-BFGS Training: RMSE Loss")
    plt.grid(True, alpha=0.3)
    plt.savefig("kan_lbfgs_results.png")
    
    plot_pruned_architecture(model_pruned)
