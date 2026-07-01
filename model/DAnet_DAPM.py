import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
import numbers


class Net(nn.Module):
    def __init__(self, factor, angRes):
        super(Net, self).__init__()
        channels = 64
        n_group = 4
        n_block = 4
        self.factor = factor
        self.gen_code = Gen_Code(15)
        self.initial_conv = nn.Conv2d(3, channels, kernel_size=3, stride=1, padding=1, bias=False) 
        self.deep_conv = CascadeGroups(n_group, n_block, angRes, channels)
        self.up_sample = nn.Sequential(
            nn.Conv2d(channels, channels * factor ** 2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.PixelShuffle(factor),
            nn.Conv2d(channels, 3, kernel_size=1, stride=1, padding=0, bias=False))
        # compress
        self.compress = nn.Sequential(
            nn.Linear(256, 64, bias=False),
            nn.LeakyReLU(0.1, True)
        )

    def forward(self, data):
        (lf, code) = data
        b, u, v, c, h, w = lf.shape
        x = rearrange(lf, 'b u v c h w -> (b u v) c h w')
        buffer = self.initial_conv(x)
        buffer = rearrange(buffer, '(b u v) c h w -> b u v c h w', b=b, u=u, v=v)
        buffer = self.deep_conv(buffer, code)
        buffer = rearrange(buffer, 'b u v c h w -> (b u v) c h w')
        out = self.up_sample(buffer)
        out = rearrange(out, '(b u v) c h w -> b u v c h w', b=b, u=u, v=v)

        return out


class CascadeGroups(nn.Module): 
    def __init__(self, n_group, n_block, angRes, channels):
        super(CascadeGroups, self).__init__()
        self.n_group = n_group
        Groups = []
        prompt_scale_factors = [1, 2, 2, 2]
        prompt_num_heads = [1, 2, 4, 8]
        for i in range(n_group):
            Groups.append(BasicGroup(n_block, angRes, channels, prompt_scale_factors[i], prompt_num_heads[i]))
        self.Group = nn.Sequential(*Groups)
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x, code):
        b, u, v, c, h, w = x.shape
        buffer = x
        for i in range(self.n_group):
            buffer = self.Group[i](buffer, code)

        buffer = rearrange(buffer, 'b u v c h w -> (b u v) c h w')
        out = self.conv(buffer)
        out = rearrange(out, '(b u v) c h w -> b u v c h w', b=b, u=u, v=v)

        return out + x


class BasicGroup(nn.Module):
    def __init__(self, n_block, angRes, channels,prompt_scale_factors, prompt_num_heads):
        super(BasicGroup, self).__init__()
        self.DAB = DAPMBlock(channels,prompt_scale_factors, prompt_num_heads)
        self.n_block = n_block
        Blocks = []
        for i in range(n_block):
            Blocks.append(DistgBlock(angRes, channels))
        self.block = nn.Sequential(*Blocks)
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x, code):
        b, u, v, c, h, w = x.shape
        buffer = self.DAB(x, code)
        for i in range(self.n_block):
            buffer = self.block[i](buffer)
        buffer = rearrange(buffer, 'b u v c h w -> (b u v) c h w')
        out = self.conv(buffer)
        out = rearrange(out, '(b u v) c h w -> b u v c h w', b=b, u=u, v=v)

        return out + x


class DistgBlock(nn.Module):
    def __init__(self, angRes, channels):
        super(DistgBlock, self).__init__()
        self.spa_conv = SpaConv(channels, channels)
        self.ang_conv = AngConv(angRes, channels, channels // 4)
        self.epi_conv = EpiConv(angRes, channels, channels // 2)
        self.fuse = nn.Sequential(
            nn.Conv2d(2 * channels + channels // 4, channels, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
        )

    def forward(self, x):
        b, u, v, c, h, w = x.shape
        fea_spa = self.spa_conv(x)
        fea_ang = self.ang_conv(x)
        fea_epih = self.epi_conv(x)
        xT = rearrange(x, 'b u v c h w -> b v u c w h')
        fea_epiv = rearrange(self.epi_conv(xT), 'b v u c w h -> b u v c h w')
        fea = torch.cat((fea_spa, fea_ang, fea_epih, fea_epiv), dim=3)
        fea = rearrange(fea, 'b u v c h w -> (b u v) c h w')
        out = self.fuse(fea)
        out = rearrange(out, '(b u v) c h w -> b u v c h w', b=b, u=u, v=v)

        return out + x


class SpaConv(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(SpaConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channel_in, channel_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(channel_in, channel_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.1, True))

    def forward(self, x):
        b, u, v, c, h, w = x.shape
        input = rearrange(x, 'b u v c h w -> (b u v) c h w')
        out = self.conv(input)
        out = rearrange(out, '(b u v) c h w -> b u v c h w', b=b, u=u, v=v)

        return out


class AngConv(nn.Module):
    def __init__(self, angRes, channel_in, channel_out):
        super(AngConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channel_in, channel_out, kernel_size=angRes, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(channel_out, angRes * angRes * channel_out, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.PixelShuffle(angRes))

    def forward(self, x):
        b, u, v, c, h, w = x.shape
        input_ang = rearrange(x, 'b u v c h w -> (b h w) c u v')
        out = self.conv(input_ang)
        out = rearrange(out, '(b h w) c u v -> b u v c h w', b=b, h=h, w=w)

        return out


class EpiConv(nn.Module):
    def __init__(self, angRes, channel_in, channel_out):
        super(EpiConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channel_in, channel_out, kernel_size=angRes, stride=1, padding=(0, angRes//2), bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(channel_out, angRes * channel_out, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1, True),
            PixelShuffle1D(angRes))

    def forward(self, x):
        b, u, v, c, h, w = x.shape
        input_epi = rearrange(x, 'b u v c h w -> (b u h) c v w')
        out = self.conv(input_epi)
        out = rearrange(out, '(b u h) c v w -> b u v c h w', b=b, u=u, h=h)

        return out


class PixelShuffle1D(nn.Module):
    def __init__(self, factor):
        super(PixelShuffle1D, self).__init__()
        self.factor = factor

    def forward(self, x):
        b, fc, h, w = x.shape
        c = fc // self.factor

        return x.view(b, c, h * self.factor, w)



class DAPMBlock(nn.Module):
    def __init__(self, channels,ffn_expansion_factor=1.0,
                 bias=False, LayerNorm_type='WithBias',prompt_scale_factor: int = 1,
                 prompt_num_heads: int = 1):
        super(DAPMBlock, self).__init__()

        self.pgm = PromptGenBlock(prompt_dim=32 * prompt_scale_factor,
                                  prompt_len=256,
                                  lin_dim=channels) # * prompt_scale_factor


        self.pim = PromptInteractionBlock(dim=channels + (32 * prompt_scale_factor), num_heads=prompt_num_heads,#dim=dim+32, num_heads=prompt_num_heads,
                                          ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                          LayerNorm_type=LayerNorm_type)

        self.conv = nn.Conv2d(channels * 2 + (32 * prompt_scale_factor), channels, 3, 1, 1)

    def forward(self, x, code_array):
        b, u, v, c, h, w = x.shape
        x1 = rearrange(x, 'b u v c h w -> (b u v) c h w')

        prompt = self.pgm(x1, code_array)
        f_hat = self.pim(x1, prompt) 
        out = torch.cat([f_hat, x1], 1)
        out = self.conv(out)
        out = rearrange(out, '(b u v) c h w -> b u v c h w', b=b, u=u, v=v)



        return out


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)



class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        x = (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias





##########################################################################
## Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, dim=192, num_heads=1, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias'):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        # x = self.norm1(x)
        # x = self.attn(x)
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


##########################################################################
##---------- Prompt Gen Module (PGM)-----------------------
class PromptGenBlock(nn.Module):
    def __init__(self, prompt_dim=96, prompt_len=256, lin_dim=96):
        super(PromptGenBlock, self).__init__()
        # self.prompt_param = nn.Parameter(torch.rand(1, prompt_len, prompt_dim, prompt_size, prompt_size))
        self.prompt_dim = prompt_dim
        self.linear_layer = nn.Linear(lin_dim, prompt_len)
        self.conv3x3 = nn.Conv2d(prompt_dim, prompt_dim, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x,k_embd):
        B, C, H, W = x.shape  # 1st it = _x96x128x128
        emb = x.mean(dim=(-2, -1))  # 4,96
        prompt_weights = F.softmax(self.linear_layer(emb), dim=1)  # 4,256
        # prompt = prompt_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * k_embd #4,256,1,4,256
        prompt = prompt_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * k_embd.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.prompt_dim, 1, 1)  # 4,256,32,1,1
        prompt = torch.sum(prompt, dim=1)  # 4,32,1,1
        # prompt = prompt.expand(-1, self.prompt_dim ,-1,-1)
        prompt = F.interpolate(prompt, (H, W), mode="bilinear")  # 4,32,128,128
        prompt = self.conv3x3(prompt)

        return prompt


##---------- Prompt Interaction Block (PIM) -----------------------

class PromptInteractionBlock(nn.Module):
    def __init__(self, dim=192, num_heads=1, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias'):
        super(PromptInteractionBlock, self).__init__()
        self.transformerblock = TransformerBlock(dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type)


    def forward(self, x, prompt):
        out = torch.cat([x, prompt], 1) 
        out = self.transformerblock(out)
        return out


class CA_Layer(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(CA_Layer, self).__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(channel_in, 16, 1, 1, 0),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(16, channel_out, 1, 1, 0),
            nn.Sigmoid())
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x, code):
        b, u, v, c, h, w = x.shape
        fea = rearrange(x, 'b u v c h w -> (b u v) c h w')
        code_fea = self.avg_pool(fea)
        code_deg = rearrange(code, 'b c u v -> (b u v) c 1 1')
        code = torch.cat((code_fea, code_deg), dim=1)
        att = self.mlp(code)
        att = att.repeat(1, 1, h, w)
        out = fea * att
        out = rearrange(out, '(b u v) c h w -> b u v c h w', b=b, u=u, v=v)

        return out


class Gen_Code(nn.Module):
    def __init__(self, channel_out):
        super(Gen_Code, self).__init__()
        kernel_size = 21
        ax = torch.arange(kernel_size).float() - kernel_size // 2
        xx = ax.repeat(kernel_size).view(1, kernel_size, kernel_size)
        yy = ax.repeat_interleave(kernel_size).view(1, kernel_size, kernel_size)
        self.xx_yy = -(xx ** 2 + yy ** 2)
        self.gen_code = nn.Sequential(
            nn.Conv2d(kernel_size ** 2, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(64, channel_out, kernel_size=1, stride=1, padding=0, bias=False))

    def forward(self, sigma):
        b, c, u, v = sigma.shape

        kernel = torch.exp(self.xx_yy.to(sigma.device) / (2. * sigma.view(-1, 1, 1) ** 2))
        kernel = kernel / kernel.sum([1, 2], keepdim=True)
        kernel = rearrange(kernel, '(b u v) h w -> b (h w) u v', b=b, u=u, v=v)
        code = self.gen_code(kernel)

        return code




