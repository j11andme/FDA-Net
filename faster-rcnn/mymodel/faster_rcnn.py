import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from mmcv.runner import BaseModule
from mmdet.models.builder import DETECTORS
from mmdet.models.detectors import TwoStageDetector

@DETECTORS.register_module()
class MyFasterRCNN(TwoStageDetector):
    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 init_cfg=None,    
                 consistency_weight=3.0,  
                 style_margin=0.6,        
                 align_metric='cosine',
                 bg_proto_type='high',
                 ):     
        super(MyFasterRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
        
        self.beta = 1.0
        self.consistency_weight = consistency_weight

        self.style_margin = style_margin
        self.lp_kernel = 5
        self.norm_eps = 1e-6
        self.align_metric = align_metric
        self.bg_proto_type = bg_proto_type
    def frequency_mixup(self, img):
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
        
        lam = torch.distributions.Beta(self.beta, self.beta).sample((B, 1, 1, 1)).to(img.device)
        
        amp_mixed = lam * amp + (1 - lam) * amp_shuffle
        
        amp_aug = mask * amp_mixed + (1 - mask) * amp
        
        fft_aug = torch.polar(amp_aug, pha)
        img_aug = torch.fft.irfftn(fft_aug, s=(H, W), dim=(-2, -1))
        
        img_aug = torch.clamp(img_aug, min=img.min(), max=img.max())
        
        return img_aug

    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        
        losses = dict()

        x_orig = self.extract_feat(img)

        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x_orig, img_metas, gt_bboxes, gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore, proposal_cfg=proposal_cfg, **kwargs)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(
            x_orig, img_metas, proposal_list, gt_bboxes, gt_labels,
            gt_bboxes_ignore, gt_masks, **kwargs)
        losses.update(roi_losses)

        with torch.no_grad():
            img_aug = self.frequency_mixup(img)
        
        x_aug = self.extract_feat(img_aug)
        
        loss_contrast = 0.0
        valid_layers = 0
        img_h, img_w = img.shape[2], img.shape[3]

        for i in range(len(x_orig)):
            feat_h, feat_w = x_orig[i].shape[2], x_orig[i].shape[3]
            stride_h = img_h / feat_h
            stride_w = img_w / feat_w

            fg_mask = torch.zeros((x_orig[i].size(0), 1, feat_h, feat_w),
                                    device=x_orig[i].device)

            with torch.no_grad():
                for b in range(x_orig[i].size(0)):
                    if len(gt_bboxes[b]) > 0:
                        bboxes_scaled = gt_bboxes[b].clone()
                        bboxes_scaled[:, 0::2] /= stride_w
                        bboxes_scaled[:, 1::2] /= stride_h

                        for bbox in bboxes_scaled:
                            if not torch.isfinite(bbox).all():
                                continue

                            x1, y1, x2, y2 = bbox
                            x1 = torch.floor(x1).long().clamp(0, feat_w - 1)
                            y1 = torch.floor(y1).long().clamp(0, feat_h - 1)
                            x2 = torch.ceil(x2).long().clamp(0, feat_w)
                            y2 = torch.ceil(y2).long().clamp(0, feat_h)

                            if x2 > x1 and y2 > y1:
                                fg_mask[b, :, y1:y2, x1:x2] = 1.0

            if fg_mask.sum() <= 0:
                continue

            feat_t = x_orig[i].detach()
            feat_s = x_aug[i]

            k = self.lp_kernel
            pad = k // 2
            feat_t_l = F.avg_pool2d(feat_t, kernel_size=k, stride=1, padding=pad)
            feat_s_l = F.avg_pool2d(feat_s, kernel_size=k, stride=1, padding=pad)
            feat_t_h = feat_t - feat_t_l
            feat_s_h = feat_s - feat_s_l

            tH = F.normalize(feat_t_h, dim=1, eps=self.norm_eps)
            sH = F.normalize(feat_s_h, dim=1, eps=self.norm_eps)
            tL = F.normalize(feat_t_l, dim=1, eps=self.norm_eps)
            sL = F.normalize(feat_s_l, dim=1, eps=self.norm_eps)

            # HF           
            if self.align_metric == 'cosine':
                    pos_sim_h = (tH * sH).sum(dim=1, keepdim=True)
                    loss_pos_h = (1.0 - pos_sim_h) * fg_mask
            elif self.align_metric == 'l1':
                loss_pos_h = F.l1_loss(feat_s_h, feat_t_h, reduction='none').mean(dim=1, keepdim=True) * fg_mask
            elif self.align_metric == 'mse':
                loss_pos_h = F.mse_loss(feat_s_h, feat_t_h, reduction='none').mean(dim=1, keepdim=True) * fg_mask
            
            term_hf = loss_pos_h.sum() / (fg_mask.sum() + 1e-6)
            # LF
            if self.align_metric == 'cosine':
                    pos_sim_l = (tL * sL).sum(dim=1, keepdim=True)
                    loss_style = F.relu(pos_sim_l - self.style_margin) * fg_mask
            else:
                if self.align_metric == 'l1':
                    dist_l = F.l1_loss(feat_s_l, feat_t_l, reduction='none').mean(dim=1, keepdim=True)
                else:
                    dist_l = F.mse_loss(feat_s_l, feat_t_l, reduction='none').mean(dim=1, keepdim=True)
                loss_style = F.relu(self.style_margin - dist_l) * fg_mask
            
            term_lf = loss_style.sum() / (fg_mask.sum() + 1e-6)
            # Neg
            bg_mask = 1.0 - fg_mask
            term_neg = 0.0
            if bg_mask.sum() > 0:
                if self.bg_proto_type == 'high':
                    bg_feat = tH if self.align_metric == 'cosine' else feat_t_h
                elif self.bg_proto_type == 'low':
                    bg_feat = tL if self.align_metric == 'cosine' else feat_t_l
                elif self.bg_proto_type == 'mixed':
                    bg_feat = F.normalize(feat_t, dim=1, eps=self.norm_eps) if self.align_metric == 'cosine' else feat_t

                bg_proto = (bg_feat * bg_mask).sum(dim=(2, 3), keepdim=True)
                bg_area = bg_mask.sum(dim=(2, 3), keepdim=True).clamp_min(1e-6)
                bg_proto = bg_proto / bg_area
                
                if self.align_metric == 'cosine':
                    bg_proto = F.normalize(bg_proto, dim=1, eps=self.norm_eps)
                    neg_sim = (sH * bg_proto).sum(dim=1, keepdim=True)
                    loss_neg = F.relu(neg_sim - self.neg_margin) * fg_mask
                else:
                    if self.align_metric == 'l1':
                        neg_dist = F.l1_loss(feat_s_h, bg_proto.expand_as(feat_s_h), reduction='none').mean(dim=1, keepdim=True)
                    else:
                        neg_dist = F.mse_loss(feat_s_h, bg_proto.expand_as(feat_s_h), reduction='none').mean(dim=1, keepdim=True)
                    loss_neg = F.relu(self.neg_margin - neg_dist) * fg_mask

                term_neg = loss_neg.sum() / (fg_mask.sum() + 1e-6)

            loss_layer = term_hf +  term_lf +  term_neg

            loss_contrast += loss_layer
            valid_layers += 1

        if valid_layers > 0:
            loss_contrast = loss_contrast / valid_layers
            losses['loss_dual_stream_contrast'] = loss_contrast * self.consistency_weight


  
        if self.with_rpn:
            rpn_losses_aug, proposal_list_aug = self.rpn_head.forward_train(
                x_aug, img_metas, gt_bboxes, gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore, proposal_cfg=proposal_cfg, **kwargs)
            
            for key, val in rpn_losses_aug.items():
                if 'loss' in key:
                    if isinstance(val, list):
                        losses[key + '_aug'] = [v  for v in val]
                    else:
                        losses[key + '_aug'] = val 
        
        roi_losses_aug = self.roi_head.forward_train(
            x_aug, img_metas, proposal_list_aug, gt_bboxes, gt_labels,
            gt_bboxes_ignore, gt_masks, **kwargs)
        
        for key, val in roi_losses_aug.items():
            if 'loss' in key:
                if isinstance(val, list):
                    losses[key + '_aug'] = [v  for v in val]
                else:
                    losses[key + '_aug'] = val 

        return losses


