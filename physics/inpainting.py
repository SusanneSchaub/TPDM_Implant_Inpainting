import torch
import logging
import nibabel as nib
import numpy as np
import pdb


mask=nib.load("/home/s.schaub/tpdm-main/x800_sf/2implants.nii.gz")
mask=np.asarray(mask.dataobj)
#mask = mask.astype(np.float32)
mask=torch.from_numpy(mask)

class Inpainting_mask():
    def __init__(self, factor: int):
        self.factor = factor
    
    def A(self, x: torch.Tensor) -> torch.Tensor:
        N, C, Y, Z = x.shape

        assert C == 1
        msk = mask[...,Z]

        result=x*msk


        return result
    
    # def A_T(self, x: torch.Tensor) -> torch.Tensor:
    #     return x / self.factor
    #
    # def A_dagger(self, x:torch.Tensor):
    #     N, C, Y, Z = x.shape
    #     assert C == 1
    #
    #     x = x.clone().detach()
    #     result = x.repeat_interleave(self.factor, dim=3)
    #
    #     return result