import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# ==========================================
# 1. SETUP & GPU CONFIG
# ==========================================
plt.switch_backend('Agg') 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.3, device=0)
    print(f"✅ Running on {torch.cuda.get_device_name(0)}")

# ==========================================
# 2. DATASET (Complex Function)
# ==========================================
class FeynmanDataset:
    def __init__(self, num_samples=10000, mode='train'):
        if mode == 'test': torch.manual_seed(42)
        
        # 4 Inputs
        self.x1 = torch.rand(num_samples, 1) * 2 - 1
        self.x2 = torch.rand(num_samples, 1) * 2 - 1
        self.x3 = torch.rand(num_samples, 1) * 2 - 1
        self.x4 = torch.rand(num_samples, 1) * 2 - 1
        
        # Target: f = exp(sin(pi*x1) + x2^2) + sin(x3*x4)
        self.y = torch.exp(torch.sin(torch.pi * self.x1) + self.x2**2) + torch.sin(self.x3 * self.x4)
        self.x = torch.cat([self.x1, self.x2, self.x3, self.x4], dim=1)
        
    def to_device(self, device):
        self.x = self.x.to(device)
        self.y = self.y.to(device)
        return self

# ==========================================
# 3. EXACT KAN LAYER (Paper Compliant)
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
        
        # Cache to store the actual phi(x) outputs for regularization
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
        # Shape: (Batch, In)
        base_output = F.linear(F.silu(x), self.base_weight)
        bs = self.b_splines(x) # (Batch, In, Grid)
        
        # Calculate output of EACH edge separately (Batch, Out, In)
        scaled_weight = self.spline_weight * self.spline_scaler.unsqueeze(-1)
        self.acts = torch.einsum('bij,oij->boi', bs, scaled_weight)
        
        # Sum over inputs to get layer output
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
    
    # --- EXACT LOSS IMPLEMENTATION (Eq 2.17 - 2.20) ---
    def get_paper_reg_loss(self, lamb_l1=1.0, lamb_entropy=1.0):
        total_l1 = 0.0
        total_entropy = 0.0
        
        for layer in self.layers:
            if layer.acts is None: continue
            
            # Eq 2.17: L1 norm of each activation function (averaged over batch)
            # acts shape: (Batch, Out, In) -> mean(0) -> (Out, In)
            edge_norms = layer.acts.abs().mean(dim=0) 
            
            # Eq 2.18: Layer L1 is sum of all edge norms
            layer_l1 = edge_norms.sum()
            total_l1 += layer_l1
            
            # Eq 2.19: Entropy of the layer
            if layer_l1 > 1e-6:
                probs = edge_norms / layer_l1
                entropy = - torch.sum(probs * torch.log(probs + 1e-8))
                total_entropy += entropy
            
        return lamb_l1 * total_l1 + lamb_entropy * total_entropy
    
    # --- NODE PRUNING (Eq 2.21) ---
    def prune_nodes(self, threshold=1e-2):
        print("✂️  Applying Paper-Style Node Pruning...")
        
        # Loop through hidden layers (between Input and Output)
        for l in range(len(self.layers) - 1):
            incoming_layer = self.layers[l]
            outgoing_layer = self.layers[l+1]
            
            # Calculate Edge Norms from the cached activations
            in_norms = incoming_layer.acts.abs().mean(dim=0) # (Nodes, Inputs)
            out_norms = outgoing_layer.acts.abs().mean(dim=0) # (Outputs, Nodes)
            
            # Eq 2.21: Scores
            score_in = in_norms.max(dim=1).values # Max incoming signal
            score_out = out_norms.max(dim=0).values # Max outgoing signal
            
            # Identify important nodes
            keep_mask = (score_in > threshold) & (score_out > threshold)
            
            num_pruned = (~keep_mask).sum().item()
            if num_pruned > 0:
                print(f"   Hidden Layer {l}: Pruning {num_pruned} nodes.")
                
                # Zero out connections
                with torch.no_grad():
                    incoming_layer.spline_scaler.data[~keep_mask, :] = 0.
                    incoming_layer.base_weight.data[~keep_mask, :] = 0.
                    outgoing_layer.spline_scaler.data[:, ~keep_mask] = 0.
                    outgoing_layer.base_weight.data[:, ~keep_mask] = 0.

# ==========================================
# 4. TRAINING ENGINE
# ==========================================
def train_experiment(model, train_data, test_data, name, do_pruning=False):
    print(f"\n--- Experiment: {name} ---")
    optimizer = optim.AdamW(model.parameters(), lr=0.01)
    
    # Track metrics
    history = {'epoch': [], 'loss': [], 'reg_loss': []}
    
    # We use a 2-Phase schedule: 
    # Phase 1: 1000 epochs (Learn Structure with Reg)
    # Phase 2: 500 epochs (Fine-tune, optionally after pruning)
    total_epochs = 1500
    prune_epoch = 1000
    
    for epoch in range(total_epochs):
        optimizer.zero_grad()
        pred = model(train_data.x)
        mse = nn.MSELoss()(pred, train_data.y)
        
        # Calculate Exact Paper Regularization
        # Using lambda=1.0 for l1 and entropy, scaled by 0.01 overall
        reg_loss = model.get_paper_reg_loss(lamb_l1=1.0, lamb_entropy=1.0)
        
        # Apply Regularization ONLY in Phase 1
        if epoch < prune_epoch:
            loss = mse + 0.01 * reg_loss
        else:
            loss = mse # Fine-tuning (Pure MSE)
            
        loss.backward()
        optimizer.step()
        
        # --- PRUNING EVENT ---
        if do_pruning and epoch == prune_epoch:
            model.prune_nodes(threshold=1e-6)
            # Reset optimizer for clean fine-tuning
            optimizer = optim.AdamW(model.parameters(), lr=0.001)

        # Logging
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                test_mse = nn.MSELoss()(model(test_data.x), test_data.y).item()
            history['epoch'].append(epoch)
            history['loss'].append(test_mse)
            history['reg_loss'].append(reg_loss.item())
            
            if epoch % 100 == 0:
                print(f"Ep {epoch} | Test MSE: {test_mse:.2e} | Reg: {reg_loss.item():.2e}")
            model.train()

    return history

def plot_pruned_architecture(model, filename="kan_pruned_architecture.png"):
    print(f"\n--- Generating Architecture Diagram: {filename} ---")
    
    # We look at Layer 0 (Input -> Hidden) and Layer 1 (Hidden -> Output)
    layer0 = model.layers[0]
    layer1 = model.layers[1]
    
    # Get mask of active nodes in hidden layer
    # A node is active if it has significant incoming AND outgoing connections
    threshold = 0.01
    
    # Calculate norms exactly like the pruning function
    in_norms = layer0.acts.abs().mean(dim=0) # (Hidden, Inputs)
    out_norms = layer1.acts.abs().mean(dim=0) # (Output, Hidden)
    
    score_in = in_norms.max(dim=1).values
    score_out = out_norms.max(dim=0).values
    
    active_nodes = (score_in > threshold) & (score_out > threshold)
    active_indices = torch.nonzero(active_nodes).flatten().cpu().numpy()
    
    print(f"Active Hidden Nodes: {active_indices}")
    
    G = nx.Graph()
    pos = {}
    
    # 1. Inputs (x1, x2, x3, x4)
    inputs = ['x1', 'x2', 'x3', 'x4']
    for i, name in enumerate(inputs):
        G.add_node(name, layer=0)
        pos[name] = (0, -i)
        
    # 2. Hidden Nodes (0 to 9)
    num_hidden = layer0.out_features
    for i in range(num_hidden):
        node_name = f"h{i}"
        G.add_node(node_name, layer=1)
        pos[node_name] = (1, -i * (len(inputs)/num_hidden)) # Scale spacing
        
    # 3. Output
    G.add_node("f", layer=2)
    pos["f"] = (2, -1.5)
    
    # DRAW EDGES
    colors = []
    widths = []
    
    # Input -> Hidden
    w_in = layer0.spline_scaler.detach().cpu().numpy() # (Hidden, In)
    for h in range(num_hidden):
        for i in range(len(inputs)):
            # If hidden node is dead, make line invisible
            strength = w_in[h, i] if h in active_indices else 0
            if strength > 0.01:
                G.add_edge(inputs[i], f"h{h}")
                colors.append('blue')
                widths.append(strength * 2)

    # Hidden -> Output
    w_out = layer1.spline_scaler.detach().cpu().numpy() # (Out, Hidden)
    for h in range(num_hidden):
        strength = w_out[0, h] if h in active_indices else 0
        if strength > 0.01:
            G.add_edge(f"h{h}", "f")
            colors.append('blue')
            widths.append(strength * 2)

    plt.figure(figsize=(8, 6))
    nx.draw_networkx_nodes(G, pos, node_color='lightgray', node_size=500)
    
    # Highlight active hidden nodes
    nx.draw_networkx_nodes(G, pos, nodelist=[f"h{i}" for i in active_indices], node_color='lightgreen', node_size=700)
    
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos, edge_color=colors, width=widths)
    
    plt.title("Pruned KAN Architecture\n(Only Green Nodes are Active)")
    plt.axis('off')
    plt.savefig(filename)
    print("✅ Diagram saved.")


# ==========================================
# 5. EXECUTION & PLOTTING
# ==========================================
if __name__ == "__main__":
    # Data
    train_data = FeynmanDataset(10000, mode='train').to_device(device)
    test_data = FeynmanDataset(2000, mode='test').to_device(device)
    
    # 1. Unpruned Experiment
    model_unpruned = KANPaper([4, 10, 1]).to(device)
    hist_unpruned = train_experiment(model_unpruned, train_data, test_data, "Unpruned KAN", do_pruning=False)
    
    # 2. Pruned Experiment
    model_pruned = KANPaper([4, 10, 1]).to(device)
    hist_pruned = train_experiment(model_pruned, train_data, test_data, "Pruned KAN", do_pruning=True)
    
    # 3. Plotting
    plt.figure(figsize=(10, 6))
    
    # Plot Unpruned
    plt.plot(hist_unpruned['epoch'], hist_unpruned['loss'], 
             label=f"Unpruned (Final: {hist_unpruned['loss'][-1]:.2e})", 
             color='orange', alpha=0.7, linestyle='--')
    
    # Plot Pruned
    plt.plot(hist_pruned['epoch'], hist_pruned['loss'], 
             label=f"Pruned (Final: {hist_pruned['loss'][-1]:.2e})", 
             color='blue', linewidth=2)
    
    # Annotate Pruning Event
    plt.axvline(x=1000, color='red', linestyle=':', label='Pruning Event')
    plt.text(1010, 1e-1, "Node Pruning & \nReg OFF", color='red', fontsize=10)
    
    plt.yscale('log')
    plt.title("Exact Paper Implementation: Node Pruning vs Baseline")
    plt.xlabel("Epochs")
    plt.ylabel("Test MSE Loss (Log Scale)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig("kan_exact_results.png")
    print("\n✅ Comparison saved to 'kan_exact_results.png'")
    
    # 4. Architecture Diagram
    plot_pruned_architecture(model_pruned)
