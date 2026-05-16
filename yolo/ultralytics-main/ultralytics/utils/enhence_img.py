import torch
import torch.fft
import torch.nn.functional as F
import random

def frequency_mixup(img,beta=1.0):
    B, C, H, W = img.shape

    fft = torch.fft.rfftn(img, dim=(-2, -1))
    amp = torch.abs(fft)
    pha = torch.angle(fft)
    
    h_freq = fft.shape[-2]
    w_freq = fft.shape[-1]
    

    ratio = 0.45 
    h_idx = int(h_freq * ratio)
    w_idx = int(w_freq * ratio)
    
    mask = torch.zeros_like(amp)
    mask[:, :, 0:h_idx, 0:w_idx] = 1.0
    
    idx = torch.randperm(B, device=img.device)
    amp_shuffle = amp[idx]
    
    lam = torch.distributions.Beta(beta, beta).sample((B, 1, 1, 1)).to(img.device)

    amp_mixed = lam * amp + (1 - lam) * amp_shuffle
    
    amp_aug = mask * amp_mixed + (1 - mask) * amp
    
    fft_aug = torch.polar(amp_aug, pha)
    img_aug = torch.fft.irfftn(fft_aug, s=(H, W), dim=(-2, -1))
    
    img_aug = torch.clamp(img_aug, min=img.min(), max=img.max())
    
    return img_aug
