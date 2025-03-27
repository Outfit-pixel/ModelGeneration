import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import lpips

class Normalize(nn.Module):
    def __init__(self, in_channels, num_groups=32):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)
    
    def forward(self, x):
        return self.norm(x)

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, dropout=0.0):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)
        self.norm1 = Normalize(in_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = nn.Dropout(dropout) if dropout > 1e-6 else nn.Identity()
        self.nin_shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return self.nin_shortcut(x) + h

class Encoder(nn.Module):
    def __init__(self, ch=128, ch_mult=(1, 2, 4, 8), num_res_blocks=2, in_channels=3, z_channels=32):
        super().__init__()
        self.ch = ch
        self.in_channels = in_channels
        self.conv_in = nn.Conv2d(in_channels, ch, 3, stride=1, padding=1)
        self.down = nn.ModuleList()

        in_ch_mult = (1,) + tuple(ch_mult)
        for i_level in range(len(ch_mult)):
            block = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(num_res_blocks):
                block.append(ResnetBlock(block_in, block_out))
                block_in = block_out
            downsample = nn.Conv2d(block_in, block_in, 3, stride=2, padding=1) if i_level < len(ch_mult) - 1 else None
            self.down.append(nn.Sequential(*block, downsample if downsample else nn.Identity()))

        self.mid = nn.Sequential(
            ResnetBlock(block_out, block_out),
            ResnetBlock(block_out, block_out)
        )
        self.norm_out = Normalize(block_out)
        self.conv_out = nn.Conv2d(block_out, z_channels, 3, stride=1, padding=1)

    def forward(self, x):
        h = self.conv_in(x)
        for down in self.down:
            h = down(h)
        h = self.mid(h)
        h = self.conv_out(F.silu(self.norm_out(h)))
        return h

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=1024, embedding_dim=32, beta=0.25, decay=0.99):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta
        self.decay = decay

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_w', self.embedding.weight.data.clone())

    def forward(self, x):
        flat_x = x.view(-1, self.embedding_dim)
        distances = torch.cdist(flat_x, self.embedding.weight)
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()
        quantized = torch.matmul(encodings, self.embedding.weight).view(x.shape)

        commitment_loss = F.mse_loss(quantized.detach(), x)
        embedding_loss = F.mse_loss(quantized, x.detach())
        loss = self.beta * commitment_loss + embedding_loss

        quantized = x + (quantized - x).detach()

        if self.training:
            self._ema_update(encodings, flat_x)

        return quantized, loss, encoding_indices

    def _ema_update(self, encodings, x):
        with torch.no_grad():
            encodings_sum = encodings.sum(0)
            self.ema_cluster_size.mul_(self.decay).add_(encodings_sum, alpha=1 - self.decay)
            n = torch.sum(self.ema_cluster_size)
            self.ema_cluster_size = ((self.ema_cluster_size + 1e-5) / (n + self.num_embeddings * 1e-5) * n)
            dw = torch.matmul(encodings.t(), x)
            self.ema_w.mul_(self.decay).add_(dw, alpha=1 - self.decay)
            self.embedding.weight.data.copy_(self.ema_w / self.ema_cluster_size.unsqueeze(1))

class Decoder(nn.Module):
    def __init__(self, ch=128, ch_mult=(8, 4, 2, 1), num_res_blocks=2, in_channels=32, out_channels=3):
        super().__init__()
        self.ch = ch
        self.in_channels = in_channels
        self.conv_in = nn.Conv2d(in_channels, ch * ch_mult[0], 3, stride=1, padding=1)
        self.mid = nn.Sequential(
            ResnetBlock(ch * ch_mult[0], ch * ch_mult[0]),
            ResnetBlock(ch * ch_mult[0], ch * ch_mult[0])
        )
        self.up = nn.ModuleList()
        for i_level in range(len(ch_mult)):
            block = nn.ModuleList()
            block_in = ch * ch_mult[i_level]
            block_out = ch * ch_mult[min(i_level + 1, len(ch_mult) - 1)]
            for _ in range(num_res_blocks):
                block.append(ResnetBlock(block_in, block_in))
            if i_level < len(ch_mult) - 1:
                upsample = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv2d(block_in, block_out, 3, stride=1, padding=1)
                )
                self.up.append(nn.Sequential(*block, upsample))
            else:
                self.up.append(nn.Sequential(*block))
        self.norm_out = Normalize(block_out)
        self.conv_out = nn.Conv2d(block_out, out_channels, 3, stride=1, padding=1)

    def forward(self, x):
        h = self.conv_in(x)
        h = self.mid(h)
        for up in self.up:
            h = up(h)
        h = F.silu(self.norm_out(h))
        h = torch.sigmoid(self.conv_out(h))
        return h