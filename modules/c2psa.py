from torch import nn
import torch
from .conv import Conv

class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, attn_ratio: float = 0.5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x


class PSABlock(nn.Module):
    def __init__(self, c: int, attn_ratio: float = 0.5, num_heads: int = 4, shortcut: bool = True):
        super().__init__()
        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn  = nn.Sequential(Conv(c, 2*c, 1), Conv(2*c, c, 1, act=False))
        self.add  = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x)  if self.add else self.ffn(x)
        return x

class PSA(nn.Module):
    def __init__(self, c1: int, c2: int, e: float = 0.5,
                 attn_ratio: float = 0.5, num_heads: int | None = None):
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        assert self.c > 0

        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        # 头数的安全回退：官方给的是 self.c // 64
        h = (self.c // 64) if num_heads is None else num_heads
        h = max(h, 1)
        if self.c % h != 0:  # 就近选可整除的头数
            for t in range(h, 0, -1):
                if self.c % t == 0:
                    h = t
                    break

        self.attn = Attention(self.c, attn_ratio=attn_ratio, num_heads=h)
        self.ffn  = nn.Sequential(Conv(self.c, 2*self.c, 1), Conv(2*self.c, self.c, 1, act=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), dim=1))


class C2PSA(nn.Module):
    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5,
                 attn_ratio: float = 0.5, num_heads: int | None = None):
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        h = (self.c // 64) if num_heads is None else num_heads
        h = max(h, 1)
        if self.c % h != 0:
            for t in range(h, 0, -1):
                if self.c % t == 0:
                    h = t
                    break

        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=attn_ratio, num_heads=h) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), dim=1))
