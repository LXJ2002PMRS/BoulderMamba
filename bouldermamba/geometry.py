"""Same native geometry for prevalidated GPU banks, without host synchronizations.

Caller must verify finite positive boxes before entering this fast path. All
arithmetic/pixel-center/zero-padding conventions match native_mask_geometry.
"""
import torch
import torch.nn.functional as F


def context_boxes(boxes,context=1.8,minimum=32,maximum=192):
    boxes=boxes.float();side=((boxes[:,2:]-boxes[:,:2]).amax(1)*context).clamp(minimum,maximum).ceil()
    origin=((boxes[:,:2]+boxes[:,2:])/2-side[:,None]/2).floor()
    return torch.cat((origin,origin+side[:,None]),1)


def physical_grid(boxes,shape):
    h,w=shape;ys=(torch.arange(h,device=boxes.device,dtype=torch.float32)+.5)/h
    xs=(torch.arange(w,device=boxes.device,dtype=torch.float32)+.5)/w;extent=boxes[:,2:]-boxes[:,:2]
    gx=boxes[:,0,None,None]+xs[None,None,:]*extent[:,0,None,None]
    gy=boxes[:,1,None,None]+ys[None,:,None]*extent[:,1,None,None]
    return torch.stack((gx.expand(-1,h,-1),gy.expand(-1,-1,w)),-1)


def sample_roi(maps,boxes,points):
    grid=(points-boxes[:,None,None,:2])/(boxes[:,None,None,2:]-boxes[:,None,None,:2])*2-1
    return F.grid_sample(maps.float(),grid,mode='bilinear',padding_mode='zeros',align_corners=False)


def probability_logit(p):return torch.logit(p.float().clamp(1e-4,1-1e-4))


def paste_roi(maps,boxes,shape):
    h,w=shape;x0,y0,x1,y1=torch.split(boxes.float(),1,1)
    ys=(torch.arange(h,device=boxes.device).float()+.5-y0)/(y1-y0)*2-1
    xs=(torch.arange(w,device=boxes.device).float()+.5-x0)/(x1-x0)*2-1
    grid=torch.stack((xs[:,None,:].expand(-1,h,-1),ys[:,:,None].expand(-1,-1,w)),3)
    return F.grid_sample(maps.float(),grid,mode='bilinear',padding_mode='zeros',align_corners=False)
