import torch
import numpy as np
import torch.nn.functional as F

class DiffusionBlend:

    def __init__(self):
        pass
    
    @staticmethod
    def BlendDiffusionOutput(diffusion_output: torch.tensor, rendered_imgs: torch.tensor): 
        # downsample rendered_imgs first
        downsampled_imgs = F.interpolate(rendered_imgs, (256, 256), mode="bilinear", align_corners=False)
        blended_imgs = diffusion_output + downsampled_imgs
        blended_imgs = downsampled_imgs
        return blended_imgs