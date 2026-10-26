"""CUDA PyTorch affine-prefix implementation of Mamba selective scan.

No model architecture/parameters change. FP32 recurrence, logarithmic-depth
parallel prefix, analytical first-order gradients; real A and variable B/C
are the exact MambaVision call contract. Unsupported contracts fail loudly.
"""
import sys
import types
import torch
import torch.nn.functional as F

def prefix(a,b):
    # Compose (a,b): x -> a*x+b. No division/cumprod underflow instability.
    step=1
    while step<b.shape[-1]:
        b=torch.cat((b[...,:step],b[...,step:]+a[...,step:]*b[...,:-step]),-1)
        a=torch.cat((a[...,:step],a[...,step:]*a[...,:-step]),-1)
        step*=2
    return b

class Scan(torch.autograd.Function):
    @staticmethod
    def forward(ctx,u,delta,A,B,C,D,bias,softplus):
        uf=u.float(); raw=delta.float()+(bias.float()[None,:,None] if bias is not None else 0)
        dt=F.softplus(raw) if softplus else raw
        a=torch.exp(dt[:,:,None,:]*A.float()[None,:,:,None])
        state=prefix(a,dt[:,:,None,:]*B.float()[:,None,:,:]*uf[:,:,None,:])
        out=(state*C.float()[:,None,:,:]).sum(2)
        if D is not None:out=out+D.float()[None,:,None]*uf
        ctx.save_for_backward(u,delta,A,B,C,state,raw,dt)
        ctx.D=D;ctx.bias=bias;ctx.softplus=softplus
        return out.to(u.dtype)
    @staticmethod
    def backward(ctx,grad):
        u,delta,A,B,C,state,raw,dt=ctx.saved_tensors
        uf=u.float();g=grad.float()
        a=torch.exp(dt[:,:,None,:]*A.float()[None,:,:,None])
        next_a=torch.cat((a[...,1:],torch.zeros_like(a[...,:1])),-1)
        gs=prefix(next_a.flip(-1),(g[:,:,None,:]*C.float()[:,None,:,:]).flip(-1)).flip(-1)
        previous=torch.cat((torch.zeros_like(state[...,:1]),state[...,:-1]),-1)
        ga=gs*previous*a
        gu=(gs*dt[:,:,None,:]*B.float()[:,None,:,:]).sum(2)
        if ctx.D is not None:gu=gu+g*ctx.D.float()[None,:,None]
        gd=(ga*A.float()[None,:,:,None]+gs*B.float()[:,None,:,:]*uf[:,:,None,:]).sum(2)
        if ctx.softplus:gd=gd*raw.sigmoid()
        gA=(ga*dt[:,:,None,:]).sum((0,3))
        gB=(gs*dt[:,:,None,:]*uf[:,:,None,:]).sum(1)
        gC=(g[:,:,None,:]*state).sum(1)
        gD=(g*uf).sum((0,2)).to(ctx.D.dtype) if ctx.D is not None else None
        gb=gd.sum((0,2)).to(ctx.bias.dtype) if ctx.bias is not None else None
        return gu.to(u.dtype),gd.to(delta.dtype),gA.to(A.dtype),gB.to(B.dtype),gC.to(C.dtype),gD,gb,None

def selective_scan_fn(u,delta,A,B,C,D=None,z=None,delta_bias=None,delta_softplus=False,return_last_state=False):
    tensors=[u,delta,A,B,C]+[v for v in (D,z,delta_bias) if v is not None]
    if not all(t.is_cuda for t in tensors):raise RuntimeError('CUDA selective scan required; no CPU fallback')
    if A.is_complex() or B.ndim!=3 or C.ndim!=3 or return_last_state:
        raise NotImplementedError('Only registered MambaVision real-A variable3D-B/C no-last-state contract supported')
    with torch.autocast('cuda',enabled=False):
        output=Scan.apply(u,delta,A,B,C,D,delta_bias,delta_softplus)
        return output if z is None else output*F.silu(z)

def install():
    try:
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as existing
        return 'installed mamba_ssm selective_scan_fn'
    except ImportError:
        root=types.ModuleType('mamba_ssm');root.__path__=[]
        ops=types.ModuleType('mamba_ssm.ops');ops.__path__=[]
        api=types.ModuleType('mamba_ssm.ops.selective_scan_interface');api.selective_scan_fn=selective_scan_fn
        api.__doc__=__doc__;root.ops=ops;ops.selective_scan_interface=api
        sys.modules[root.__name__]=root;sys.modules[ops.__name__]=ops;sys.modules[api.__name__]=api
        return 'project CUDA PyTorch affine-prefix selective scan; FP32 analytical gradients'
