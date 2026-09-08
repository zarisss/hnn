"""
Hamiltonian Neural Network (HNN) for the ideal mass-spring system.

The network predicts a scalar treated as the system's Hamiltonian H_theta.
H_theta is differentiated with respect to the state (q, p) to get a
predicted symplectic gradient (dq/dt, dp/dt), which is compared against
the true derivatives from the dataset. The network never sees H directly —
only its gradient — so it learns H indirectly, up to an additive constant.
"""

import torch
from torch import nn
import matplotlib.pyplot as plt

import mass_spring as ms


class HNN(nn.Module):
    """Maps a (q, p) state to a scalar Hamiltonian-like value."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(2, 10),
            nn.Tanh(),
            nn.Linear(10, 5),
            nn.Tanh(),
            nn.Linear(5, 1),
           # nn.Tanh(),not using the last layer activation fucntion as it will squash the scalar between(-1, 1).
        )

    def forward(self, state):
        return self.linear(state)


# --- Setup---
model = HNN()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

state_train, state_test, deriv_train, deriv_test = ms.build_data()

losses = []
NUM_ITERATIONS = 2000

print("Initial weights (layer 0):\n", model.linear[0].weight)

# --- Training loop ---
for i in range(NUM_ITERATIONS):
    optimizer.zero_grad()  # clear gradients before this iteration's backward pass

    # Forward pass: network predicts the scalar Hamiltonian
    H_pred = model(state_train)

    # Differentiate H_pred w.r.t. the input state (q, p), keeping the graph
    # alive (create_graph=True) so backprop can later reach the weights.
    grad_H = torch.autograd.grad(H_pred, state_train, grad_outputs=torch.ones_like(H_pred), create_graph=True,)[0]  # tuple -> tensor of shape (N, 2): [dH/dq, dH/dp]

    dH_dq = grad_H[:, 0:1]
    dH_dp = grad_H[:, 1:2]

    # Symplectic rotation: predicted (dq/dt, dp/dt) = (dH/dp, -dH/dq)
    deriv_pred = torch.cat([dH_dp, -dH_dq], dim=1)

    loss = loss_fn(deriv_pred, deriv_train)
    loss.backward()
    optimizer.step()

    losses.append(loss.item()) #item() method is used, to store only the final loss value instead f the entire graph.

print("Updated weights (layer 0):\n", model.linear[0].weight)

# --- Plot loss curve ---
plt.plot(losses)
plt.xlabel("Iteration")
plt.ylabel("MSE Loss")
plt.title("HNN Training Loss (Mass-Spring System)")
plt.savefig("loss_plot.jpg")

# Testin --------#

test_model = model(state_test)
grad_test_model = torch.autograd.grad(test_model, state_test, grad_outputs=torch.ones_like(test_model))[0]
dHt_dq = grad_test_model[:, 0:1]
dHt_dp = grad_test_model[:, 1:2]
deri_testt = torch.cat([dHt_dp, -dHt_dq], dim = 1)
test_loss = loss_fn(deri_testt, deriv_test)
# --- Detach test predictions from the graph, convert to numpy for plotting ---
deri_testt_np = deri_testt.detach().numpy()
deriv_test_np = deriv_test.numpy()  # already ungraphed, no .detach() needed, but harmless if included

# --- Scatter plot: true vs predicted, one subplot per component ---
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# dq/dt component
axes[0].scatter(deriv_test_np[:, 0], deri_testt_np[:, 0], alpha=0.5)
lims_q = [deriv_test_np[:, 0].min(), deriv_test_np[:, 0].max()]
axes[0].plot(lims_q, lims_q, 'r--')  # diagonal reference line
axes[0].set_xlabel("True dq/dt")
axes[0].set_ylabel("Predicted dq/dt")
axes[0].set_title("dq/dt: True vs Predicted")

# dp/dt component
axes[1].scatter(deriv_test_np[:, 1], deri_testt_np[:, 1], alpha=0.5)
lims_p = [deriv_test_np[:, 1].min(), deriv_test_np[:, 1].max()]
axes[1].plot(lims_p, lims_p, 'r--')
axes[1].set_xlabel("True dp/dt")
axes[1].set_ylabel("Predicted dp/dt")
axes[1].set_title("dp/dt: True vs Predicted")

plt.tight_layout()
plt.savefig("test_pred_vs_true.jpg")
