import torch
import torch.nn as nn
import torch.nn.functional as F


class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()

    def _gather_feat(self, feat, ind, mask=None):
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat

    def _tranpose_and_gather_feat(self, feat, ind):
        feat = feat.permute(0, 2, 3, 1).contiguous()
        feat = feat.view(feat.size(0), -1, feat.size(3))
        feat = self._gather_feat(feat, ind)
        return feat

    def forward(self, output, mask, ind, target):
        # torch.Size([1, 1, 152, 152])
        # torch.Size([1, 500])
        # torch.Size([1, 500])
        # torch.Size([1, 500, 1])
        pred = self._tranpose_and_gather_feat(output, ind)  # torch.Size([1, 500, 1])
        if mask.sum():
            mask = mask.unsqueeze(2).expand_as(pred).bool()
            loss = F.binary_cross_entropy(pred.masked_select(mask),
                                          target.masked_select(mask),
                                          reduction='mean')
            return loss
        else:
            return 0.

class OffSmoothL1Loss(nn.Module):
    def __init__(self):
        super(OffSmoothL1Loss, self).__init__()

    def _gather_feat(self, feat, ind, mask=None):
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat

    def _tranpose_and_gather_feat(self, feat, ind):
        feat = feat.permute(0, 2, 3, 1).contiguous()
        feat = feat.view(feat.size(0), -1, feat.size(3))
        feat = self._gather_feat(feat, ind)
        return feat

    def forward(self, output, mask, ind, target):
        # torch.Size([1, 2, 152, 152])
        # torch.Size([1, 500])
        # torch.Size([1, 500])
        # torch.Size([1, 500, 2])
        pred = self._tranpose_and_gather_feat(output, ind)  # torch.Size([1, 500, 2])
        if mask.sum():
            mask = mask.unsqueeze(2).expand_as(pred).bool()
            loss = F.smooth_l1_loss(pred.masked_select(mask),
                                    target.masked_select(mask),
                                    reduction='mean')
            return loss
        else:
            return 0.

class FocalLoss(nn.Module):
    def __init__(self):
        super(FocalLoss, self).__init__()

    # MAE * grid to get pixel
    def forward(self, pred, gt, metrics=False):
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        neg_weights = torch.pow(1 - gt, 4)

        loss = 0

        pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
        neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

        num_pos  = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos

        if metrics:
            kpts_metrics = self.compute_metrics(pred, gt)
            return loss, kpts_metrics
        
        return loss
        
    def _nms(self, heat, kernel=3):
        hmax = F.max_pool2d(heat, (kernel, kernel), stride=1, padding=(kernel - 1) // 2)
        keep = (hmax == heat).float()
        return heat * keep
    
    def compute_metrics(self, pred, gt, threshold=[0.3, 0.5, 0.75]):
        kpts_metrics = {}

        for thrs in threshold:
            pred_nms = self._nms(pred, kernel=3)
            gt_nms = self._nms(gt, kernel=3)

            pred_bin = (pred_nms >= thrs).float()
            gt_bin = (gt_nms==1).float()

            tp = ((pred_bin == 1) & (gt_bin == 1)).sum().item()
            fp = ((pred_bin == 1) & (gt_bin == 0)).sum().item()
            fn = ((pred_bin == 0) & (gt_bin == 1)).sum().item()

            precision = tp / (tp + fp + 1e-6)
            recall = tp / (tp + fn + 1e-6)
            f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)
            
            kpts_metrics[f'Precision_corners_{thrs}'] = precision
            kpts_metrics[f'Recall_corners_{thrs}'] = recall
            kpts_metrics[f'F1_corners_{thrs}'] = f1_score

        return kpts_metrics


def isnan(x):
    return x != x

  
class LossAll(torch.nn.Module):
    def __init__(self):
        super(LossAll, self).__init__()
        self.L_hm = FocalLoss()
        self.L_wh = OffSmoothL1Loss()
        self.L_off = OffSmoothL1Loss()
        self.L_cls_theta = BCELoss()
        self.L_hm_kpts = FocalLoss()
        self.L_off_kpts = OffSmoothL1Loss

    def forward(self, pr_decs, gt_batch):
        hm_loss  = self.L_hm(pr_decs['hm'], gt_batch['hm'])
        wh_loss  = self.L_wh(pr_decs['wh'], gt_batch['reg_mask'], gt_batch['ind'], gt_batch['wh'])
        off_loss = self.L_off(pr_decs['reg'], gt_batch['reg_mask'], gt_batch['ind'], gt_batch['reg'])
        cls_theta_loss = self.L_cls_theta(pr_decs['cls_theta'], gt_batch['reg_mask'], gt_batch['ind'], gt_batch['cls_theta'])
        hm_kpts_loss, kpts_metrics = self.L_hm(pr_decs['hm_kpts'], gt_batch['hm_kpts'], metrics=True)
        off_kpts_loss = self.L_off(pr_decs['reg_kpts'], gt_batch['reg_mask_kpts'], gt_batch['ind_kpts'], gt_batch['reg_kpts'])

        if isnan(hm_loss) or isnan(wh_loss) or isnan(off_loss):
            print('hm loss is {}'.format(hm_loss))
            print('wh loss is {}'.format(wh_loss))
            print('off loss is {}'.format(off_loss))

        # print(hm_loss)
        # print(wh_loss)
        # print(off_loss)
        # print(cls_theta_loss)
        # print('-----------------')

        mae = 0
        loss =  hm_loss + wh_loss + off_loss + cls_theta_loss + hm_kpts_loss + off_kpts_loss
        return loss, hm_loss, wh_loss, off_loss, cls_theta_loss, hm_kpts_loss, off_kpts_loss, kpts_metrics, mae
