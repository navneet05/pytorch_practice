import torch
import numpy as np

# direct from data
data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)
print(x_data)
# from numpy array
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print(x_np)
# from another tensor
x_ones = torch.ones_like(x_data) # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")

#properties of tensor
print(f"Shape of tensor: {x_data.shape}")
print(f"Datatype of tensor: {x_data.dtype}")
print(f"Device tensor is stored on: {x_data.device}")
n = x_data.numpy()
print(type(x_data))
# moving tensor to gpu
# We move our tensor to the GPU if available
if torch.cuda.is_available():
	tensor = tensor.to('cuda')

# numpy like slicing
tensor = torch.ones(4, 4)
print('First row: ', tensor[0])
print('First column: ', tensor[:, 0])
print('Last column:', tensor[..., -1])
tensor[:,1] = 0
print(tensor)