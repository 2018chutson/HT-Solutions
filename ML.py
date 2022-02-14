#hi
import pandas as pd
import numpy as np
import torch as tr

class Net(tr.nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    # l1 is the layer from input to hidden
    # l2 is the layer from hidden to output
    self.l1 = tr.nn.Linear(4, 10) #raise(NotImplementedError)
    self.l2 = tr.nn.Linear(10,1) #raise(NotImplementedError)
  def forward(self, x):
    # Define the network's forward pass
    # For an input feature vector x, should return a scalar output y
    # y should be between 0 and 1, indicating the recidivism prediction
    # Use tanh for the hidden layer activation
    # Use sigmoid for the output node activation
    
    x = tr.tanh(self.l1(x))   
    x = tr.sigmoid(self.l2(x))
   #raise(NotImplementedError)
    return x

data = pd.read_csv("test data.csv", index_col=0)
labels = data.pop("incidents")
features = data.copy()

print("test data")
print(data)


def to_tensor(df): return tr.tensor(df.values.astype(np.float32))

inputs, targets = to_tensor(features), to_tensor(labels)

net = Net()
optimizer = tr.optim.SGD(net.parameters(), lr=0.01/inputs.shape[0])
num_epochs = 100
for epoch in range(num_epochs):
  # Start with zero gradient
  optimizer.zero_grad()

  # Calculate network output and sum of squared loss for each datapoint
  y = net(inputs)
  loss = ((y.squeeze() - targets.squeeze())**2).sum()

  # Calculate gradients and take a descent step
  loss.backward()
  optimizer.step()

  # Monitor optimization progress
  num_errors = (y.squeeze().round().detach().numpy() != targets.numpy()).sum()
  if epoch % (num_epochs/10) == 0: print(loss, num_errors)

# Print predicted and target outputs for the first 10 datapoints 
print(y.squeeze()[:2])
print(y.squeeze()[:2].round())
print(targets[:2].squeeze())
print((y.squeeze().detach().numpy() > .5).any())

