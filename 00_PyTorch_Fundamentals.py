## 00. PyTorch Fundamentals

import torch 
import time
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

