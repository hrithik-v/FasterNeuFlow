import torch
import torch.nn as nn
import torch.nn.functional as F
from NeuFlow.debug import fprint

class SAConvMixer(nn.Module):
    def __init__(self, dim, depth, kernel_size=9):
        super(SAConvMixer, self).__init__()
        self.layers = nn.Sequential(
            *[nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size, groups=dim, padding=kernel_size // 2),
                nn.GELU(),
                nn.Conv2d(dim, dim, 1),
                nn.GELU()
            ) for _ in range(depth)]
        )
        self.residual_scale = 0.5  # Scale for residual connection

    def forward(self, x):
        residual = x
        x = self.layers(x)
        return x + self.residual_scale * residual  # Add residual connection

class UpSample(nn.Module):
    def __init__(self, feature_dim, upsample_factor, sa_conv_depth=4):
        super(UpSample, self).__init__()

        self.upsample_factor = upsample_factor

        # Convolution layers for concatenated flow and feature tensors
        self.conv1 = nn.Conv2d(2 + feature_dim, 256, 3, 1, 1)
        self.sa_conv_mixer = SAConvMixer(256, sa_conv_depth)
        self.conv2 = nn.Conv2d(256, upsample_factor ** 2 * 9, 1, 1, 0)
        self.relu = nn.ReLU()

        # Additional layer for convex weights
        self.convex_weights = nn.Conv2d(256, upsample_factor ** 2 * 9, 1, 1, 0)

        # Initialize weights
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        nn.init.constant_(self.conv1.bias, 0)
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
        nn.init.constant_(self.conv2.bias, 0)
        nn.init.kaiming_normal_(self.convex_weights.weight, nonlinearity='relu')
        nn.init.constant_(self.convex_weights.bias, 0)

    def normalize_flow(self, flow, min_value=-10, max_value=10):
        """
        Normalize the flow tensor to keep the values within a specific range.
        """
        # Clip the flow values to stay within the specified range
        return torch.clamp(flow, min=min_value, max=max_value)

    def forward(self, feature, flow):
        # Normalize the flow tensor to ensure the values stay within a stable range
        flow = self.normalize_flow(flow)

        # Concatenate flow and feature tensors
        concat = torch.cat((flow, feature), dim=1)

        # Apply convolution and SAConvMixer to the concatenated tensor
        x = self.relu(self.conv1(concat))
        x = self.sa_conv_mixer(x)

        # Generate convex weights for upsampling
        convex_weights = self.convex_weights(x)
        b, _, h, w = flow.shape

        # Reshape and normalize weights for convex combination
        convex_weights = convex_weights.view(b, 1, 9, self.upsample_factor, self.upsample_factor, h, w)
        convex_weights = torch.softmax(convex_weights, dim=2) + 1e-8  # Softmax for stability

        # Unfold the flow tensor for the convolution operation
        unfolded_flow = F.unfold(flow, [3, 3], padding=1)
        unfolded_flow = unfolded_flow.view(b, 2, 9, 1, 1, h, w)

        # Apply convex combination to the unfolded flow
        up_flow = torch.sum(convex_weights * unfolded_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        up_flow = up_flow.reshape(b, 2, self.upsample_factor * h, self.upsample_factor * w)

        fprint("Upsampled Flow: ", up_flow.shape, f"Range: {up_flow.min()} - {up_flow.max()}")
        return up_flow
