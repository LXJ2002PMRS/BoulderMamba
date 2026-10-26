"""Small, bounded native-pixel refiner. Labels never enter forward.

One output cell is exactly one original pixel. Everything outside the integer
patch retains the source mask. No confidence gate restricts correction/loss.
Includes the native-pixel objective and the checkpoint-compatible module.
"""
import torch
from torch import nn
import torch.nn.functional as F
from .geometry import sample_roi,probability_logit


class NativePrecisionRefiner(nn.Module):
    def __init__(self,width=32,max_delta=2.,patch_size=96):
        super().__init__();self.max_delta=max_delta;self.patch_size=patch_size
        self.semantic=nn.Sequential(nn.Conv2d(256,16,1),nn.GroupNorm(4,16),nn.SiLU())
        self.input=nn.Sequential(nn.Conv2d(24,width,3,padding=1),nn.GroupNorm(8,width),nn.SiLU())
        self.blocks=nn.ModuleList([nn.Sequential(nn.Conv2d(width,width,3,padding=d,dilation=d),nn.GroupNorm(8,width),nn.SiLU(),nn.Conv2d(width,width,3,padding=1),nn.GroupNorm(8,width)) for d in (1,2,3)])
        self.output=nn.Conv2d(width,1,1);nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
        self.register_buffer('offsets',torch.arange(patch_size,dtype=torch.float32)+.5,persistent=False)

    def forward(self,batch,photometric=False):
        p=batch['native_probabilities'].detach().float();gray=batch['gray'].detach().float()/255
        boxes=batch['boxes'].detach().float();origins=batch['origins'].detach().float();n=len(boxes);s=self.patch_size
        x=origins[:,0,None,None]+self.offsets[None,None,:];y=origins[:,1,None,None]+self.offsets[None,:,None]
        points=torch.stack((x.expand(n,s,s),y.expand(n,s,s)),-1)
        semantic=sample_roi(self.semantic(batch['decoder_features'].detach().float()),boxes,points)
        wh=(boxes[:,2:]-boxes[:,:2]).clamp_min(1);center=(boxes[:,:2]+boxes[:,2:])/2
        relative=((points-center[:,None,None])/wh[:,None,None]).clamp(-3,3).permute(0,3,1,2)
        valid=((points[...,0]>=0)&(points[...,0]<500)&(points[...,1]>=0)&(points[...,1]<500))[:,None].float()
        prior=probability_logit(p)
        if photometric:
            gain=.85+.3*torch.rand(n,1,1,1,device=p.device);offset=(torch.rand(n,1,1,1,device=p.device)-.5)*.1
            gray=(gray*gain+offset).clamp(0,1)
        scale=wh.prod(1).sqrt().log()[:,None,None,None].expand(-1,1,s,s)/5
        # Confidence descriptor only: retain the 24-channel checkpoint contract.
        # It never multiplies the residual or selects supervised pixels.
        confidence_feature=(prior.abs()<4).float()
        features=torch.cat((gray,p,prior/4,relative,valid,scale,confidence_feature,semantic),1)
        hidden=self.input(features.to(memory_format=torch.channels_last))
        for block in self.blocks:hidden=F.silu(hidden+block(hidden))
        delta=self.max_delta*torch.tanh(self.output(hidden).float())*valid
        return dict(delta=delta,prior=prior,valid=valid,confidence_feature=confidence_feature)


def precision_loss(result,target,gt_area,outside_area,outside_intersection,weights,valid_examples,strength=1.):
    target=target.detach().float();logits=result['prior']+strength*result['delta'];valid=result['valid'];band=valid
    # Equal per-pixel BCE: foreground does not receive the old 8x multiplier.
    bce=(F.binary_cross_entropy_with_logits(logits,target,reduction='none')*band).sum((1,2,3))/band.sum((1,2,3)).clamp_min(1)
    probability=logits.sigmoid()*valid;intersection=(probability*target).sum((1,2,3))+outside_intersection
    prediction=probability.sum((1,2,3))+outside_area
    iou=1-(intersection+1)/(prediction+gt_area-intersection).clamp_min(1).add(1)
    # Explicit signed-area control discourages both erosion and expansion.
    area=((prediction-gt_area)/gt_area.clamp_min(10)).square().clamp_max(4)
    regularizer=(result['delta'].square()*band).sum((1,2,3))/band.sum((1,2,3)).clamp_min(1)
    per=bce+iou+.1*area+.03*regularizer
    w=weights*valid_examples
    return (per*w).sum()/w.sum().clamp_min(1),dict(bce=bce.mean().detach(),iou=iou.mean().detach(),area=area.mean().detach())
