"""Image decoding on host; grayscale arithmetic and spatial flips on CUDA."""
import numpy as np
import torch
from PIL import Image

def native_gray(path,device,flip=False,direction='horizontal'):
    if torch.device(device).type!='cuda':raise RuntimeError('CUDA native image processing required')
    image=Image.open(path)
    array=np.array(image,copy=True)
    tensor=torch.as_tensor(array,device=device)
    if image.mode=='P':
        palette=torch.tensor(image.getpalette(),device=device,dtype=torch.long).reshape(-1,3)
        tensor=palette[tensor.long()]
    if tensor.ndim==3:
        rgb=tensor[...,:3].long()
        tensor=((rgb[...,0]*19595+rgb[...,1]*38470+rgb[...,2]*7471+32768)>>16).clamp(0,255).byte()
    else:tensor=tensor.clamp(0,255).byte()
    if tuple(tensor.shape)!=(500,500):raise RuntimeError('Native500 required')
    if flip:
        axes=[]
        if direction in ('horizontal','diagonal'):axes.append(1)
        if direction in ('vertical','diagonal'):axes.append(0)
        tensor=tensor.flip(axes)
    return tensor
