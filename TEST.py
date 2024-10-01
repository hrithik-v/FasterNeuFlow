# import torch
# from NeuFlow import neuflow

# model = neuflow.NeuFlow()
# # print(model)

# image0 = torch.rand(1, 3, 224, 224)
# image1 = torch.rand(1, 3, 224, 224)

# output = model(image0, image1)
# print(f"output image: {output.shape}")
# # summary(model.cuda(), (3, 384, 520))
import torch
import torch.nn as nn

# Create the convolution layers
conv_x8 = nn.Conv2d(in_channels=128, out_channels=192, kernel_size=2)
conv_x16 = nn.Conv2d(in_channels=256, out_channels=192, kernel_size=2)

# Input tensors
x8 = torch.randn(1, 128, 28, 28)
x16 = torch.randn(1, 256, 14, 14)

# Forward pass through the convolution layers
x8_transformed = conv_x8(x8)  # Shape will be [1, 192, 28, 28]
x16_transformed = conv_x16(x16)  # Shape will be [1, 192, 14, 14]

# Print shapes
print(x8_transformed.shape)  # torch.Size([1, 192, 28, 28])
print(x16_transformed.shape)  # torch.Size([1, 192, 14, 14])
