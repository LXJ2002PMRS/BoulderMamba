"""Joint all-parameter Cascade/NPR training; native prediction interface.

Proposal coordinates and GT targets are not differentiable; image/ROI
features and mask probabilities are. Native forward never receives targets.
Uncertainty gating is removed; confidence input remains a descriptor only.
"""
import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
from mmdet.registry import MODELS
from mmdet.models.detectors import CascadeRCNN
from mmdet.models.roi_heads import CascadeRoIHead
from mmdet.structures.bbox import bbox2roi
from .quality_head import QualityCascadeRoIHead,whole_instance_iou_targets
from .native_refiner import NativePrecisionRefiner,precision_loss
from .geometry import sample_roi,paste_roi,probability_logit
from .image_io import native_gray


class JointNativePrecisionRefiner(NativePrecisionRefiner):
    def forward(self,batch,photometric=False):
        p=batch['native_probabilities'].float();gray=batch['gray'].float()/255
        boxes=batch['boxes'].detach().float();origins=batch['origins'].detach().float();n=len(boxes);s=self.patch_size
        x=origins[:,0,None,None]+self.offsets[None,None,:];y=origins[:,1,None,None]+self.offsets[None,:,None]
        points=torch.stack((x.expand(n,s,s),y.expand(n,s,s)),-1)
        semantic=sample_roi(self.semantic(batch['decoder_features'].float()),boxes,points)
        wh=(boxes[:,2:]-boxes[:,:2]).clamp_min(1);center=(boxes[:,:2]+boxes[:,2:])/2
        relative=((points-center[:,None,None])/wh[:,None,None]).clamp(-3,3).permute(0,3,1,2)
        valid=((points[...,0]>=0)&(points[...,0]<500)&(points[...,1]>=0)&(points[...,1]<500))[:,None].float()
        prior=probability_logit(p)
        if photometric:
            gray=(gray*(.85+.3*torch.rand(n,1,1,1,device=p.device))+(torch.rand(n,1,1,1,device=p.device)-.5)*.1).clamp(0,1)
        scale=wh.prod(1).sqrt().log()[:,None,None,None].expand(-1,1,s,s)/5;confidence_feature=(prior.abs()<4).float()
        hidden=self.input(torch.cat((gray,p,prior/4,relative,valid,scale,confidence_feature,semantic),1).to(memory_format=torch.channels_last))
        for block in self.blocks:hidden=F.silu(hidden+block(hidden))
        # The confidence channel is a descriptor, never a correction gate.
        delta=self.max_delta*torch.tanh(self.output(hidden).float())*valid
        return dict(delta=delta,prior=prior,valid=valid,confidence_feature=confidence_feature)

def native_batch(probabilities,features,boxes,gray):
    n=len(boxes);s=96;origins=((boxes[:,:2]+boxes[:,2:])/2-48).floor().to(torch.int64)
    offsets=torch.arange(s,device=boxes.device,dtype=torch.float32)+.5
    x=origins[:,0,None,None]+offsets[None,None,:];y=origins[:,1,None,None]+offsets[None,:,None]
    points=torch.stack((x.expand(n,s,s),y.expand(n,s,s)),-1)
    valid=((points[...,0]>=0)&(points[...,0]<500)&(points[...,1]>=0)&(points[...,1]<500))[:,None]
    prob=sample_roi(probabilities,boxes,points)*valid
    xx=(origins[:,0,None,None]+torch.arange(s,device=boxes.device)[None,None,:]).expand(n,s,s)
    yy=(origins[:,1,None,None]+torch.arange(s,device=boxes.device)[None,:,None]).expand(n,s,s)
    graypatch=gray[yy.clamp(0,499),xx.clamp(0,499)][:,None]*valid
    return dict(native_probabilities=prob,gray=graypatch,boxes=boxes,origins=origins,decoder_features=features),xx,yy,valid

@MODELS.register_module()
class BoulderMambaRoIHead(QualityCascadeRoIHead):
    def __init__(self,native_loss_weight=.25,native_training_samples=64,**kwargs):
        super().__init__(**kwargs);self.native_refiner=JointNativePrecisionRefiner();self.native_loss_weight=native_loss_weight;self.native_training_samples=native_training_samples
        self._last_decoder=None;self.mask_head[-1].convs[-1].register_forward_hook(self._capture_decoder)
        self.native_gray=None;self.native_metas=None;self.native_last_training={}
    def _capture_decoder(self,module,args,result):self._last_decoder=result
    def _mask_forward(self,stage,x,rois):
        result=super()._mask_forward(stage,x,rois)
        if stage==self.num_stages-1:
            result['decoder_features']=F.adaptive_avg_pool2d(self._last_decoder,14);self._last_decoder=None
        return result
    def mask_loss(self,stage,x,sampling_results,batch_gt_instances):
        # The original three stage mask losses still train all mask heads.
        result=CascadeRoIHead.mask_loss(self,stage,x,sampling_results,batch_gt_instances)
        if stage!=self.num_stages-1:return result
        rois=bbox2roi([s.pos_priors for s in sampling_results]);losses=result['loss_mask']
        if not len(rois):
            losses['loss_mask_quality']=sum(p.sum()*0 for p in self.quality_head.parameters())
            losses['loss_native_precision']=sum(p.sum()*0 for p in self.native_refiner.parameters());return result
        others=[]
        with torch.no_grad():
            for index in range(self.num_stages-1):others.append(self._mask_forward(index,x,rois)['mask_preds'].float().sigmoid())
        ensemble=torch.stack(others+[result['mask_preds'].float().sigmoid()])
        quality_targets=whole_instance_iou_targets(ensemble.detach().mean(0),result['mask_targets'],sampling_results,batch_gt_instances)
        quality_logits=self.quality_head(result['mask_feats'],ensemble,rois)
        losses['loss_mask_quality']=self.quality_loss_weight*F.binary_cross_entropy_with_logits(quality_logits,quality_targets)
        offset=0;native_losses=[];eligible_total=0;sampled_total=0
        for image_index,(sampled,gt) in enumerate(zip(sampling_results,batch_gt_instances)):
            count=len(sampled.pos_priors)
            if not count:continue
            selected=torch.randperm(count,device=rois.device)[:self.native_training_samples]+offset;offset+=count
            meta=self.native_metas[image_index];scale=rois.new_tensor(meta['scale_factor']).repeat(2);boxes=rois[selected,1:]/scale
            local=selected-(offset-count);gt_ids=sampled.pos_assigned_gt_inds[local]
            # Native labels follow the same image flip. Nearest 3x resize is
            # verified below; this never changes the inference interface.
            masks=gt.masks.masks;assert tuple(meta['img_shape'][:2])==(1500,1500) and min(masks.shape[-2:])>=1500
            # DetDataPreprocessor pads masks to 1504; remove only that padding.
            truth=torch.as_tensor(np.ascontiguousarray(masks[:,:1500,:1500][:,::3,::3]),device=rois.device,dtype=torch.uint8)
            areas=truth.sum((1,2)).float()[gt_ids]
            features=result['decoder_features'][selected].to(torch.float16).float()
            probability=ensemble.mean(0)[selected]
            batch,xx,yy,valid=native_batch(probability,features,boxes,self.native_gray[image_index])
            target=truth[gt_ids[:,None,None],yy.clamp(0,499),xx.clamp(0,499)][:,None]*valid
            with torch.no_grad():
                binary=batch['native_probabilities']>=.5;outside_area=[];outside_intersection=[]
                for start in range(0,len(selected),8):
                    end=min(start+8,len(selected));full=paste_roi(probability[start:end].detach(),boxes[start:end],(500,500))[:,0]>=.5
                    outside_area.append(full.sum((1,2)).float()-binary[start:end].sum((1,2,3)))
                    outside_intersection.append((full*truth[gt_ids[start:end]]).sum((1,2)).float()-(binary[start:end]*target[start:end]).sum((1,2,3)))
                oa=torch.cat(outside_area);oi=torch.cat(outside_intersection)
                inter=(binary*target).sum((1,2,3))+oi;prediction=binary.sum((1,2,3))+oa;eligible=inter/(prediction+areas-inter).clamp_min(1)>=.5
                counts=torch.bincount(gt_ids,minlength=len(truth));weights=1/counts[gt_ids].float().clamp_min(1)
            refined=self.native_refiner(batch,photometric=True)
            loss,_=precision_loss(refined,target,areas,oa,oi,weights,eligible.float());native_losses.append(loss)
            eligible_total+=int(eligible.sum());sampled_total+=len(selected)
        losses['loss_native_precision']=self.native_loss_weight*torch.stack(native_losses).mean() if native_losses else sum(p.sum()*0 for p in self.native_refiner.parameters())
        self.native_last_training=dict(sampled=sampled_total,eligible=eligible_total);return result

    def predict_mask(self,x,batch_img_metas,results_list,rescale=False):
        rois=bbox2roi([r.bboxes for r in results_list])
        if not len(rois):return super().predict_mask(x,batch_img_metas,results_list,rescale)
        assert len(results_list)==1 and rescale and tuple(batch_img_metas[0]['ori_shape'][:2])==(500,500)
        stages=[]
        for stage in range(self.num_stages):
            result=self._mask_forward(stage,x,rois);stages.append(result['mask_preds'].float().sigmoid())
        ensemble=torch.stack(stages);quality=self.quality_head(result['mask_feats'],ensemble,rois).sigmoid();merged=ensemble.mean(0)
        scale=rois.new_tensor(batch_img_metas[0]['scale_factor']).repeat(2);boxes=rois[:,1:]/scale
        features=result['decoder_features'].to(torch.float16).float()
        results_list=self.mask_head[-1].predict_by_feat(mask_preds=(merged,),results_list=results_list,batch_img_metas=batch_img_metas,rcnn_test_cfg=self.test_cfg,rescale=True,activate_map=True)
        prediction=results_list[0];prediction.class_scores=prediction.scores.clone();prediction.mask_qualities=quality.to(prediction.scores.dtype)
        if self.apply_quality:prediction.scores=prediction.scores*prediction.mask_qualities
        assert torch.equal(prediction.bboxes,boxes)
        for start in range(0,len(boxes),64):
            end=min(start+64,len(boxes));length=end-start;ids=torch.arange(start,end,device=boxes.device)
            if length<64:ids=torch.cat((ids,ids[:1].expand(64-length)))
            batch,_,_,_=native_batch(merged[ids],features[ids],boxes[ids],self.native_gray[0])
            with torch.autocast('cuda',dtype=torch.bfloat16):r=self.native_refiner(batch)
            masks=(r['prior']+r['delta']>=0)[:length,0];origins=batch['origins'][:length].cpu().tolist()
            for i,(px,py) in enumerate(origins,start):
                x0,x1=max(0,px),min(500,px+96);y0,y1=max(0,py),min(500,py+96)
                if x1>x0 and y1>y0:prediction.masks[i,y0:y1,x0:x1]=masks[i-start,y0-py:y1-py,x0-px:x1-px]
        return results_list

@MODELS.register_module()
class BoulderMamba(CascadeRCNN):
    def _attach_native_images(self,samples):
        gray=[]
        for sample in samples:
            meta=sample.metainfo
            gray.append(native_gray(meta['img_path'],next(self.parameters()).device,meta.get('flip',False),meta.get('flip_direction','horizontal')))
        self.roi_head.native_gray=gray;self.roi_head.native_metas=[s.metainfo for s in samples]
    def loss(self,batch_inputs,batch_data_samples):
        self._attach_native_images(batch_data_samples);return super().loss(batch_inputs,batch_data_samples)
    def predict(self,batch_inputs,batch_data_samples,rescale=True):
        self._attach_native_images(batch_data_samples);return super().predict(batch_inputs,batch_data_samples,rescale)
