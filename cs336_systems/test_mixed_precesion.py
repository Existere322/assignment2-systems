import torch.nn as nn
import torch
from cs336_basics.optimizer import AdamW

"""
s = torch.tensor(0,dtype=torch.float32) 
for i in range(1000):  
    s += torch.tensor(0.01,dtype=torch.float32) 
print(s)  
s = torch.tensor(0,dtype=torch.float16) 

for i in range(1000):  
    s += torch.tensor(0.01,dtype=torch.float16) 
print(s)  

s = torch.tensor(0,dtype=torch.float32) 
for i in range(1000): 
    s += torch.tensor(0.01,dtype=torch.float16) 
print(s)  

s = torch.tensor(0,dtype=torch.float32) 
for i in range(1000):  
    x = torch.tensor(0.01,dtype=torch.float16) 
    s += x.type(torch.float32) 
print(s)
"""


class ToyModel(nn.Module):  

    def __init__(self, in_features: int, out_features: int, device):  
        super().__init__() 
        self.fc1 = nn.Linear(in_features, 10, bias=False, device=device) 
        self.ln = nn.LayerNorm(10, device=device) 
        self.fc2 = nn.Linear(10, out_features, bias=False, device=device) 
        self.relu = nn.ReLU() 
        
    def forward(self, x):  
        x = self.fc1(x)
        print(f"fc1 output dtype: {x.dtype}")
        x = self.relu(x) 
        print(f"relu output dtype: {x.dtype}")
        x = self.ln(x) 
        print(f"ln output dtype: {x.dtype}")
        x = self.fc2(x)    
        print(f"fc2 output dtype: {x.dtype}")
        return x
    

def print_param_and_grad_dtype(model):
    print("\n=== Parameter and gradient dtype ===")
    for name, param in model.named_parameters():
        grad_dtype = None if param.grad is None else param.grad.dtype
        print(
            f"{name:20s} | "
            f"param dtype: {str(param.dtype):15s} | "
            f"grad dtype: {grad_dtype}"
        )

    
model = ToyModel(10, 10, device="cuda")
x: torch.Tensor = torch.rand([10], device="cuda", dtype=torch.float32)
target: torch.Tensor = torch.rand([10], device="cuda", dtype=torch.float32)
optimizer = AdamW(model.parameters())
criterion = nn.MSELoss()


optimizer.zero_grad()
print("\n=== Forward without autocast ===")
y = model(x)
print(f"final y dtype: {y.dtype}")
loss = criterion(y, target)
print(f"final loss dtype: {loss.dtype}")
loss.backward()
optimizer.step()
print_param_and_grad_dtype(model)
optimizer.zero_grad()


model1 = ToyModel(10, 10, device="cuda")
optimizer = AdamW(model1.parameters())
optimizer.zero_grad()
print("\n=== Forward with autocast bf16 ===")
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    y = model1(x)
    print(f"final y dtype: {y.dtype}")
    loss = criterion(y, target)
    print(f"final loss dtype: {loss.dtype}")
    loss.backward()
    optimizer.step()
    print_param_and_grad_dtype(model1)
    optimizer.zero_grad()

print(f"final y dtype: {y.dtype}")
