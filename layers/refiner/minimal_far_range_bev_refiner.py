"""Minimal, bounded, far-range-only BEV residual refiner for Step8B."""
import torch
from torch import nn

class MinimalFarRangeBEVRefiner(nn.Module):
    def __init__(self, channels=128, x_min=0.0, x_max=192.0, x_threshold=100.0, alpha_cap=0.1):
        super().__init__(); self.x_min=float(x_min); self.x_max=float(x_max); self.x_threshold=float(x_threshold); self.alpha_cap=float(alpha_cap)
        self.conv1=nn.Conv2d(channels,channels,3,padding=1,bias=False); self.norm=nn.GroupNorm(8,channels); self.act=nn.ReLU(inplace=True); self.conv2=nn.Conv2d(channels,channels,3,padding=1,bias=False); self.alpha_raw=nn.Parameter(torch.zeros(()))
        # A strictly zero conv2 together with alpha=0 would create a dead
        # product with zero gradient.  Tiny initialization keeps the baseline
        # perturbation negligible while allowing alpha to learn.
        nn.init.normal_(self.conv2.weight, mean=0.0, std=1e-3)
    def forward(self,x):
        h,w=x.shape[-2:]; xs=torch.linspace(self.x_min,self.x_max,w,device=x.device,dtype=x.dtype); mask=(xs>=self.x_threshold).view(1,1,1,w)
        residual=self.conv2(self.act(self.norm(self.conv1(x)))); alpha=self.alpha_cap*torch.tanh(self.alpha_raw)
        scaled = alpha * mask * residual
        if getattr(self, 'trace_enabled', False):
            self.last_raw_residual = residual.detach()
            self.last_scaled_residual = scaled.detach()
            self.last_mask = mask.detach()
        return x+scaled
