"""BF16 interpolation bridge for PyTorch 2.0 CUDA execution."""
import torch
import torch.nn.functional as F

def install_cuda_bf16_interpolate():
    if getattr(F.interpolate,'_study_cuda_bf16',False):return
    original=F.interpolate
    def interpolate(input,*args,**kwargs):
        if input.is_cuda and input.dtype==torch.bfloat16:
            with torch.autocast('cuda',enabled=False):
                return original(input.float(),*args,**kwargs).to(input.dtype)
        return original(input,*args,**kwargs)
    interpolate._study_cuda_bf16=True;F.interpolate=interpolate
