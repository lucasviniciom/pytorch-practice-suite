## 00. PyTorch Fundamentals

import torch 
print(torch.__version__)
print(torch.cuda.is_available())

## Intro to Tensors

# scalar
scalar = torch.tensor(7)
print(scalar)
scalar.item()

# Vector
vector = torch.tensor([7,7]) 
print(vector)

# MATRIX
MATRIX = torch.tensor([[7,8],
                       [9,10]])

print(MATRIX)

#TENSOR
TENSOR = torch.tensor([[[1,2,3],
                        [3,6,9],
                        [2,4,5]]])

print(TENSOR)

## Random tensors
## Useful as the basic of NN generally is start tensors full of random numbers and adjust these to better represent the data

random_tensor = torch.rand(2,2,2,4)

#print(random_tensor)

# Create tensor of all zeros and ones
zeros = torch.zeros(3,4)
ones = torch.ones(3,4)

## Range of tensors
one_ten = torch.arange(0,10)
print(one_ten)

##tensor dtypes
##standard torch.float32, other options as .float16, .float64, .complex32 might be useful
## Is one of the most common errors in PyTorch and DeepLearning
## Other 2 are Tensor not in right shape
## And Tensor not on right device(cpu, gpu, etc)

float_32_tensor = torch.tensor([3.0,6.0,9.0],
                               dtype = None, #what dataype is the tensor
                               device = None, #What device tensor is on
                               requires_grad=False) #wheter or not track gradients in operations


