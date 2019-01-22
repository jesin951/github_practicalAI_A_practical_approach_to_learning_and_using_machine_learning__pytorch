

#######################################################################

# coding: utf-8
# # PyTorch
# <img src="https://raw.githubusercontent.com/GokuMohandas/practicalAI/master/images/logo.png" width=150>
# 
# In this lesson we'll learn about PyTorch which is a machine learning library used to build dynamic neural networks. We'll learn about the basics, like creating and using Tensors, in this lesson but we'll be making models with it in the next lesson.
# 
# <img src="https://raw.githubusercontent.com/GokuMohandas/practicalAI/master/images/pytorch.png" width=300>
# # Tensor basics
# In[2]:
# Load PyTorch library
import numpy as np
import torch
# In[4]:
# Creating a zero tensor
x = torch.Tensor(3, 4)
print("Type: {}".format(x.type()))
print("Size: {}".format(x.shape))
print("Values: \n{}".format(x))
# In[5]:
# Creating a random tensor
x = torch.randn(2, 3) # normal distribution (rand(2,3) -> uniform distribution)
print (x)
# In[6]:
# Zero and Ones tensor
x = torch.zeros(2, 3)
print (x)
x = torch.ones(2, 3)
print (x)
# In[7]:
# List → Tensor
x = torch.Tensor([[1, 2, 3],[4, 5, 6]])
print("Size: {}".format(x.shape)) 
print("Values: \n{}".format(x))
# In[8]:
# NumPy array → Tensor
x = torch.from_numpy(np.random.rand(2, 3))
print("Size: {}".format(x.shape)) 
print("Values: \n{}".format(x))
# In[9]:
# Changing tensor type
x = torch.Tensor(3, 4)
print("Type: {}".format(x.type()))
x = x.long()
print("Type: {}".format(x.type()))
# # Tensor operations
# In[10]:
# Addition
x = torch.randn(2, 3)
y = torch.randn(2, 3)
z = x + y
print("Size: {}".format(z.shape)) 
print("Values: \n{}".format(z))
# In[11]:
# Dot product
x = torch.randn(2, 3)
y = torch.randn(3, 2)
z = torch.mm(x, y)
print("Size: {}".format(z.shape)) 
print("Values: \n{}".format(z))
# In[12]:
# Transpose
x = torch.randn(2, 3)
print("Size: {}".format(x.shape)) 
print("Values: \n{}".format(x))
y = torch.t(x)
print("Size: {}".format(y.shape)) 
print("Values: \n{}".format(y))
# In[13]:
# Reshape
z = x.view(3, 2)
print("Size: {}".format(z.shape)) 
print("Values: \n{}".format(z))
# In[14]:
# Dangers of reshaping (unintended consequences)
x = torch.tensor([
    [[1,1,1,1], [2,2,2,2], [3,3,3,3]],
    [[10,10,10,10], [20,20,20,20], [30,30,30,30]]
])
print("Size: {}".format(x.shape)) 
print("Values: \n{}\n".format(x))
a = x.view(x.size(1), -1)
print("Size: {}".format(a.shape)) 
print("Values: \n{}\n".format(a))
b = x.transpose(0,1).contiguous()
print("Size: {}".format(b.shape)) 
print("Values: \n{}\n".format(b))
c = b.view(b.size(0), -1)
print("Size: {}".format(c.shape)) 
print("Values: \n{}".format(c))
# In[15]:
# Dimensional operations
x = torch.randn(2, 3)
print("Values: \n{}".format(x))
y = torch.sum(x, dim=0) # add each row's value for every column
print("Values: \n{}".format(y))
z = torch.sum(x, dim=1) # add each columns's value for every row
print("Values: \n{}".format(z))
# # Indexing, Splicing and Joining
# In[16]:
x = torch.randn(3, 4)
print("x: \n{}".format(x))
print ("x[:1]: \n{}".format(x[:1]))
print ("x[:1, 1:3]: \n{}".format(x[:1, 1:3]))
# In[17]:
# Select with dimensional indicies
x = torch.randn(2, 3)
print("Values: \n{}".format(x))
col_indices = torch.LongTensor([0, 2])
chosen = torch.index_select(x, dim=1, index=col_indices) # values from column 0 & 2
print("Values: \n{}".format(chosen)) 
row_indices = torch.LongTensor([0, 1])
chosen = x[row_indices, col_indices] # values from (0, 0) & (2, 1)
print("Values: \n{}".format(chosen)) 
# In[18]:
# Concatenation
x = torch.randn(2, 3)
print("Values: \n{}".format(x))
y = torch.cat([x, x], dim=0) # stack by rows (dim=1 to stack by columns)
print("Values: \n{}".format(y))
# # Gradients
# In[19]:
# Tensors with gradient bookkeeping
x = torch.rand(3, 4, requires_grad=True)
y = 3*x + 2
z = y.mean()
z.backward() # z has to be scalar
print("Values: \n{}".format(x))
print("x.grad: \n", x.grad)
# * $ y = 3x + 2 $
# * $ z = \sum{y}/N $
# * $ \frac{\partial(z)}{\partial(x)} = \frac{\partial(z)}{\partial(y)} \frac{\partial(y)}{\partial(x)} = \frac{1}{N} * 3 = \frac{1}{12} * 3 = 0.25 $
# # CUDA tensors
# In[20]:
# Is CUDA available?
print (torch.cuda.is_available())
# If the code above return False, then go to `Runtime` → `Change runtime type` and select `GPU` under `Hardware accelerator`. 
# In[23]:
# Creating a zero tensor
x = torch.Tensor(3, 4).to("cpu")
print("Type: {}".format(x.type()))
# In[24]:
# Creating a zero tensor
x = torch.Tensor(3, 4).to("cuda")
print("Type: {}".format(x.type()))
