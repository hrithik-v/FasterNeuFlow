import torch
import torch.nn.functional as F

class SAConvMixer(torch.nn.Module):
    def __init__(self, dim, depth, kernel_size=9):
        super(SAConvMixer, self).__init__()
        self.layers = torch.nn.Sequential(
            *[torch.nn.Sequential(
                torch.nn.Conv2d(dim, dim, kernel_size, groups=dim, padding=kernel_size // 2),
                torch.nn.GELU(),
                torch.nn.Conv2d(dim, dim, 1),
                torch.nn.GELU()
            ) for _ in range(depth)]
        )

    def forward(self, x):
        return self.layers(x)

class UpSample(torch.nn.Module):
    def __init__(self, feature_dim, upsample_factor, sa_conv_depth=4):
        super(UpSample, self).__init__()

        self.upsample_factor = upsample_factor

        self.conv1 = torch.nn.Conv2d(2 + feature_dim, 256, 3, 1, 1)
        self.sa_conv_mixer = SAConvMixer(256, sa_conv_depth)
        self.conv2 = torch.nn.Conv2d(256, upsample_factor ** 2 * 9, 1, 1, 0)
        self.relu = torch.nn.ReLU(inplace=True)

        # Additional layer for convex weights
        self.convex_weights = torch.nn.Conv2d(256, upsample_factor ** 2 * 9, 1, 1, 0)

    def forward(self, feature, flow):
        concat = torch.cat((flow, feature), dim=1)

        x = self.relu(self.conv1(concat))
        x = self.sa_conv_mixer(x)

        # Convex weights generation
        convex_weights = self.convex_weights(x)
        b, _, h, w = flow.shape

        # Reshape and normalize weights for convex combination
        convex_weights = convex_weights.view(b, 1, 9, self.upsample_factor, self.upsample_factor, h, w)
        convex_weights = torch.softmax(convex_weights, dim=2)

        # Unfold flow and apply convex upsampling
        unfolded_flow = F.unfold(flow, [3, 3], padding=1)
        unfolded_flow = unfolded_flow.view(b, 2, 9, 1, 1, h, w)

        up_flow = torch.sum(convex_weights * unfolded_flow, dim=2)  # Convex combination
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        up_flow = up_flow.reshape(b, 2, self.upsample_factor * h, self.upsample_factor * w)

        return up_flow
