import torch
import torch.nn as nn
from kan import KAN_Layer
import matplotlib.pyplot as plt

# Training data
xs = torch.linspace(-3, 3, 100)
targets = torch.sin(torch.exp(xs))
deg = 3  # deg of polynomial
knot_vec = torch.linspace(-15, 15, 100)
model = nn.Sequential(
    KAN_Layer(in_dim=1, out_dim=2, deg=deg, knot_vec=knot_vec),
    KAN_Layer(in_dim=2, out_dim=1, deg=deg, knot_vec=knot_vec)
)

# L-BFGS optimizer
optimizer = torch.optim.LBFGS(model.parameters(), lr=1, max_iter=20, history_size=10)

loss_plot_y = []
xs_batch = xs.unsqueeze(1)  # Prepare batch once

print("Training with L-BFGS optimizer...")

for step in range(50):  # Much fewer steps needed!
    def closure():
        optimizer.zero_grad()
        preds = model(xs_batch).squeeze()
        loss = 0.5 * ((preds - targets)**2).mean()
        loss.backward()
        return loss
    
    loss = optimizer.step(closure)
    loss_plot_y.append(loss.item())
    
    if step % 10 == 0 or step == 49:
        print(f"Step {step} | Loss = {loss.item():.6f}")

print()
print("=" * 60)
print("Training Complete!")
print("=" * 60)

# Make the loss plot against iterations
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
ind = [i for i in range(len(loss_plot_y))]
plt.plot(ind, loss_plot_y)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss over Iterations')
plt.grid(True)

# Plotting preds vs inputs and targets vs inputs
plt.subplot(1, 2, 2)
with torch.no_grad():
    xs_batch = xs.unsqueeze(1)
    preds = model(xs_batch).squeeze()

plt.plot(xs.numpy(), targets.numpy(), label='Target (sin(x))', linewidth=2)
plt.plot(xs.numpy(), preds.detach().numpy(), label='Predictions', linestyle='--', linewidth=2)
plt.xlabel('x')
plt.ylabel('y')
plt.title('KAN Approximation of sin(x)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('kan_training_results.png', dpi=150)
print(f"\nPlot saved as 'kan_training_results.png'")
print(f"Final Loss: {loss_plot_y[-1]:.6f}")
plt.show()
