import torch
from torch import nn
import mass_spring as ms

class HNN(nn.Module): #creating a class HNN that inherits the class MODULE from nn module.
    def __init__(self): #__init__ is an constructor, that is used while initialising the object.
        super().__init__()
        self.linear = nn.Sequential( #nn.Sequential is a class from the nn module, nn.Sequential(), is calling that class to construct an instance/object of it and attaches to the linear(variable name)) attribute.
                nn.Linear(2, 10),
                nn.ReLU(),
                nn.Linear(10, 5),
                nn.ReLU(),
                nn.Linear(5, 1),
                )
    def forward(self, state):
        out = self.linear(state)
        return out

#flow: feed the states in the network -> network predicts the hamiltonian -> hamiltonian is differentiated wrt the states _> compared with the derivatives generated from the mass__spring module -> calculating the costfunction and then backprop.
model = HNN() #creating the instance of the class HNN and attaching it to the model variable
state_ten, test_ten, train_deri = ms.build_data() #using the build_data method from the mass_spring module.
Hamiltonian = model(state_ten)
#todo: differentiate the hamiltonian with q and p
gradie = torch.autograd.grad(Hamiltonian, state_ten, grad_outputs=torch.ones_like(Hamiltonian, dtype=torch.float32), create_graph=True) # [dH/dq, dH/dp], gradie is a tuple.
#todo2: rotating the tensor by 90 and multiplying a minus as to mathc with the data generateed
#rotated_tensor =
statey_ten = gradie[0]
dhdq = statey_ten[:, 0:1]
dhdp = statey_ten[:, 1:2]
sttate = torch.cat([dhdp, -dhdq], dim=1)
loss_funct = nn.MSELoss()
mse = loss_funct(sttate, train_deri)
_ = mse.backward()
#print("shape of dhdq, dhdp: ", sttate.shape)
#print("shape of derivative: ", train_deri.shape)
print("weight of the first layer: ", model.linear[0].weight.grad)

#todo: updating the weights, till now got the weights to be updated.
