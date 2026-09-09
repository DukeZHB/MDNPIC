"""
Diffusion Classification (DC) network of MDNPIC.

The denoising model eps_theta takes the image, the timestep, the noisy label
y_t and the three-dimensional guidance priors (y_global, y_local, y_texture)
predicted by the pre-trained MGCG network, and predicts the added noise
(Eq. 5 in the paper).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18, resnet50
from torchvision.models.densenet import densenet121


class ConditionalLinear(nn.Module):
    def __init__(self, num_in, num_out, n_steps):
        super(ConditionalLinear, self).__init__()
        self.num_out = num_out
        self.lin = nn.Linear(num_in, num_out)
        self.embed = nn.Embedding(n_steps, num_out)
        self.embed.weight.data.uniform_()

    def forward(self, x, t):
        out = self.lin(x)
        gamma = self.embed(t)
        out = gamma.view(-1, self.num_out) * out
        return out


class ConditionalModel(nn.Module):
    def __init__(self, config, guidance=False):
        super(ConditionalModel, self).__init__()
        n_steps = config.diffusion.timesteps + 1
        y_dim = config.data.num_classes
        arch = config.model.arch
        feature_dim = config.model.feature_dim
        in_channels = getattr(config.data, "num_channels", 3)
        self.guidance = guidance
        # encoder for x
        self.encoder_x = ResNetEncoder(arch=arch, feature_dim=feature_dim,
                                       in_channels=in_channels)
        # batch norm layer
        self.norm = nn.BatchNorm1d(feature_dim)

        # conditional guidance MLP; conditioning input = [y_t, y_g, y_l, y_tx]
        if self.guidance:
            self.lin1 = ConditionalLinear(y_dim * 4, feature_dim, n_steps)
        else:
            self.lin1 = ConditionalLinear(y_dim, feature_dim, n_steps)
        self.unetnorm1 = nn.BatchNorm1d(feature_dim)
        self.lin2 = ConditionalLinear(feature_dim, feature_dim, n_steps)
        self.unetnorm2 = nn.BatchNorm1d(feature_dim)
        self.lin3 = ConditionalLinear(feature_dim, feature_dim, n_steps)
        self.unetnorm3 = nn.BatchNorm1d(feature_dim)
        self.lin4 = nn.Linear(feature_dim, y_dim)

    def forward(self, x, y, t, y_global=None, y_local=None, y_texture=None):
        x = self.encoder_x(x)
        x = self.norm(x)
        if self.guidance:
            if y_global is None:
                y_global = torch.zeros_like(y)
            if y_local is None:
                y_local = torch.zeros_like(y)
            if y_texture is None:
                y_texture = torch.zeros_like(y)
            y = torch.cat([y, y_global, y_local, y_texture], dim=-1)
        y = self.lin1(y, t)
        y = self.unetnorm1(y)
        y = F.softplus(y)
        y = x * y
        y = self.lin2(y, t)
        y = self.unetnorm2(y)
        y = F.softplus(y)
        y = self.lin3(y, t)
        y = self.unetnorm3(y)
        y = F.softplus(y)
        y = self.lin4(y)
        return y


# ResNet 18 or 50 as image encoder
class ResNetEncoder(nn.Module):
    def __init__(self, arch='resnet18', feature_dim=128, in_channels=1):
        super(ResNetEncoder, self).__init__()

        self.f = []
        if arch == 'resnet50':
            backbone = resnet50()
            self.featdim = backbone.fc.weight.shape[1]
        elif arch == 'resnet18':
            backbone = resnet18()
            self.featdim = backbone.fc.weight.shape[1]
        elif arch == 'densenet121':
            backbone = densenet121(pretrained=True)
            self.featdim = backbone.classifier.weight.shape[1]
        elif arch == 'vit':
            from timm.models import create_model
            backbone = create_model('pvt_v2_b2',
                                    pretrained=True,
                                    num_classes=4,
                                    drop_rate=0,
                                    drop_path_rate=0.1,
                                    drop_block_rate=None,
                                    )
            backbone.head = nn.Sequential()
            self.featdim = 512
        else:
            raise NotImplementedError(f"Unknown encoder architecture: {arch}")

        # adapt the stem convolution to the actual number of input channels
        if in_channels != 3 and hasattr(backbone, "conv1"):
            backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7,
                                       stride=2, padding=3, bias=False)

        for name, module in backbone.named_children():
            if name != 'fc':
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)

        self.g = nn.Linear(self.featdim, feature_dim)

    def forward_feature(self, x):
        feature = self.f(x)
        feature = torch.flatten(feature, start_dim=1)
        feature = self.g(feature)
        return feature

    def forward(self, x):
        feature = self.forward_feature(x)
        return feature
