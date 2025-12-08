## 00. PyTorch Fundamentals

import torch 
import time
import numpy as np
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


###Getting information from tensors
# datatype - tensor.dtype
# shape - tensor.shape
# device - tensor.device

some_tensor = torch.rand(3,4, dtype=torch.float16, device='cpu')
print(some_tensor)
print(f"Datatype: {some_tensor.dtype}")
print(f"Shape: {some_tensor.shape}")
print(f"Device: {some_tensor.device}")


## gpu test

# CPU
#x_cpu = torch.randn(5000, 5000)
#start = time.time()
#y_cpu = x_cpu @ x_cpu
#print("CPU time:", time.time() - start)

# GPU
#print("-----GPU Test-----")
#print("CUDA available:", torch.cuda.is_available())
#print("Current device index:", torch.cuda.current_device())
#print("Device name:", torch.cuda.get_device_name(0))
#x_gpu = x_cpu.to("cuda")
#torch.cuda.synchronize()
#start = time.time()
#y_gpu = x_gpu @ x_gpu
#torch.cuda.synchronize()
#print("GPU time:", time.time() - start)

###
#torch.cuda.empty_cache()
#print("Allocated:", torch.cuda.memory_allocated() / 1024**2, "MB")
#print("Cached:", torch.cuda.memory_reserved() / 1024**2, "MB")

# allocate
#a = torch.randn(10000, 10000, device="cuda")
#
#print("Allocated after tensor:", torch.cuda.memory_allocated() / 1024**2, "MB")
#print("Cached:", torch.cuda.memory_reserved() / 1024**2, "MB")

### Manipulating Tensors
## Tensor operations:
# Adition, Subtraction, Multiplication(element-wise)
# Division, Matrix multiplication

tensor = torch.tensor([1,2,3])
tensor = tensor + 10
print(tensor)
tensor *= 10
print(tensor)
tensor -= 10
print(tensor)
#Pytorch in-build functions. In general, prefer the python for basic ones. For complex mult, use torch.
X = torch.mul(tensor,10)
print(X)

## Matrix multiplication
# Elementwise
tensor = torch.tensor([1,4,5])
print(tensor,"*",tensor)
print(f"Equals:{tensor * tensor}")

#Matrix multiplication
tensor_mult = torch.matmul(tensor,tensor)
print(tensor_mult)

#Transpose matrix - useful if mismatch

tensor = torch.tensor([[1,4,5],[5,6,8]])

tensor_T = tensor.T
print(tensor_T)

## Min, Max, Mean, Sum - Aggregations

X = torch.arange(0, 100, 10)
print(X)
print(torch.min(X), X.min())
print(torch.max(X), X.max())
#print(torch.mean(X)) doesnt work with long, requires float32
print(torch.mean(X.type(torch.float32))) 
#Find positional min and max
a = X.argmin()
print(a)
a = X.argmax()
print(a)

# Reshaping, stacking, squeezing and unsqueezing tensors
# View = Return a view of an input tensor, keeping the same memory as original
# Stacking - Combining multiple tensors on top(vstack) or side by side(hstack)
# Squeeze - removels all 1 dimensions from a tensor
# Unsqueeze - add a 1 dimensions to a target tensro
# Permute - Return a view of the input with dimensions permuted/swapped in a certain way

x = torch.arange(1, 10)
print(x, x.shape)
x_reshaped = x.reshape(1,9,1)
print(x_reshaped,x_reshaped.shape)

#Change view
Z = X.view(2,5)
print(Z, Z.shape)
#Changing Z changes X - because a view of a tensor shares the same memory as the original
Z[:,0]=5
print(Z, X)

#Stack tensors on top each other

x_stacked = torch.stack([x,x,x,x])
print(x_stacked)

#Squeeze / Unsqueeze
print(x_reshaped,x_reshaped.shape)
x_squeezed= torch.squeeze(x_reshaped)
print(x_reshaped,x_squeezed)

print(f'Previous tensor: {x_squeezed}')
print(f'Previous shape: {x_squeezed.shape}')
x_unsqueezed =  x_squeezed.unsqueeze(dim=0)
#The dimension here is where is going to add a new one, in which position of the dimensions [0,1,2..etc]
print(f'New tensor: {x_unsqueezed}')
print(f'New shape: {x_unsqueezed.shape}')

# Permute - rearragen the dimensions in another order
x_original = torch.rand(size=(224,224,3)) #common in images, height/widht/colour
x_permuted = x_original.permute(2,0,1)

print(f'Previous shape: {x_original.shape}')
print(f'New shape: {x_permuted.shape}')

print('----------')
### Indexing - Selecting data from tensors
X = torch.arange(1,19).reshape(2,3,3)
print(X, X.shape)
#Index examples
print(X[:])
print(X[:][1])
print(X[:][1][2])

## Pytorch tensors + Numpy

array = np.arange(1.0,8.0)
tensor = torch.from_numpy(array)
#Reflects numpy default float64 unless specified otherwise
print(array,tensor)
#tensor to numpy
tensor = torch.ones(7)
numpy_tensor = tensor.numpy()
print(tensor,numpy_tensor)

#Reproducibility
random_seed = 42
torch.manual_seed(random_seed)
random_tensor_A = torch.rand(3,4)
random_seed = 42
torch.manual_seed(random_seed)
random_tensor_B = torch.rand(3,4)
print(random_tensor_A)
print(random_tensor_B)
print(random_tensor_A == random_tensor_B)

## Running tensors and Pytorch objects on GPUs
## Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

#Putting tensors (and models) on the GPU
tensor = torch.tensor([1,2,3])
#tensor not on GPU
print(tensor, tensor.device)

#Move to tensor
tensor_on_gpu = tensor.to(device)
print(tensor_on_gpu, tensor_on_gpu.device)

## Move tensors back to CPU
#If tensor is on GPU, can't transfor it to Numpy
#This doesn't work
#tensor_on_gpu.numpy()
tensor_on_cpu = tensor.cpu()
print(tensor_on_cpu, tensor_on_cpu.device)
array = tensor_on_cpu.numpy()
print(array)

##