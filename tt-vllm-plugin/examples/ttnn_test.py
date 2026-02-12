import torch
import ttnn

# Open device
device = ttnn.open_device(device_id=0)

# Create two simple tensors
torch_a = torch.ones((32, 32), dtype=torch.bfloat16)
torch_b = torch.ones((32, 32), dtype=torch.bfloat16)

# Convert to ttnn tensors and move to device
a = ttnn.from_torch(torch_a, device=device, layout=ttnn.TILE_LAYOUT)
b = ttnn.from_torch(torch_b, device=device, layout=ttnn.TILE_LAYOUT)

# Add tensors
output = ttnn.add(a, b)

# Convert back to torch and verify
result = ttnn.to_torch(output)
print(f"Result shape: {result.shape}")
print(f"Expected: 2.0, Got: {result[0, 0].item()}")

# Close device
ttnn.close_device(device)
