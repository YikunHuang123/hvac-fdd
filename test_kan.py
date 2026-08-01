import torch
from kan import KAN

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

model = KAN(width=[5, 3, 5], grid=5, k=3, seed=42)
x = torch.randn(10, 5)
y = model(x)
print("KAN output shape:", y.shape)
print("Success!")
