import torch.nn.functional as F
import torch

class DecDecoder(object):
    def __init__(self, K, conf_thresh, num_classes):
        self.K = K
        self.conf_thresh = conf_thresh
        self.num_classes = num_classes

    def _topk(self, scores):
        batch, cat, height, width = scores.size()

        topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), self.K)
        
        topk_inds = topk_inds % (height * width)
        topk_ys = (topk_inds // width).int().float()
        topk_xs = (topk_inds % width).int().float()

        ys_indices = topk_ys.squeeze(1)
        xs_indices = topk_xs.squeeze(1)

        topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), self.K)
        topk_clses = (topk_ind // self.K).int()
        topk_inds = self._gather_feat( topk_inds.view(batch, -1, 1), topk_ind).view(batch, self.K)
        topk_ys = self._gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, self.K)
        topk_xs = self._gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, self.K)

        return topk_score, topk_inds, topk_clses, topk_ys, topk_xs, ys_indices, xs_indices


    def _nms(self, heat, kernel=3):
        hmax = F.max_pool2d(heat, (kernel, kernel), stride=1, padding=(kernel - 1) // 2)
        keep = (hmax == heat).float()
        return heat * keep

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
    
    def ctdet_decode(self, pr_decs):
        heat = pr_decs['hm']
        wh = pr_decs['wh']
        reg = pr_decs['reg']
        cls_theta = pr_decs['cls_theta']
        heat_kpts = pr_decs['hm_kpts']
        reg_kpts = pr_decs['reg_kpts']
        edges = pr_decs['edges']

        batch, c, height, width = heat.size()
        heat = self._nms(heat)

        scores, inds, clses, ys, xs, ys_indices, xs_indices = self._topk(heat)
        reg = self._tranpose_and_gather_feat(reg, inds)
        reg = reg.view(batch, self.K, 2)
        xs = xs.view(batch, self.K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, self.K, 1) + reg[:, :, 1:2]
        clses = clses.view(batch, self.K, 1).float()
        scores = scores.view(batch, self.K, 1)
        wh = self._tranpose_and_gather_feat(wh, inds)
        wh = wh.view(batch, self.K, 10)
        # add
        cls_theta = self._tranpose_and_gather_feat(cls_theta, inds)
        cls_theta = cls_theta.view(batch, self.K, 1)
        mask = (cls_theta>0.8).float().view(batch, self.K, 1)
        
        #
        tt_x = (xs+wh[..., 0:1])*mask + (xs)*(1.-mask)
        tt_y = (ys+wh[..., 1:2])*mask + (ys-wh[..., 9:10]/2)*(1.-mask)
        rr_x = (xs+wh[..., 2:3])*mask + (xs+wh[..., 8:9]/2)*(1.-mask)
        rr_y = (ys+wh[..., 3:4])*mask + (ys)*(1.-mask)
        bb_x = (xs+wh[..., 4:5])*mask + (xs)*(1.-mask)
        bb_y = (ys+wh[..., 5:6])*mask + (ys+wh[..., 9:10]/2)*(1.-mask)
        ll_x = (xs+wh[..., 6:7])*mask + (xs-wh[..., 8:9]/2)*(1.-mask)
        ll_y = (ys+wh[..., 7:8])*mask + (ys)*(1.-mask)
        #

        batch_kpts, c_kpts, height_kpts, width_kpts = heat_kpts.size()
        heat_kpts = self._nms(heat_kpts)
        
        self.visualize_heatmap_and_corners(heat, heat_kpts, edges)
        
        scores_kpts, inds_kpts, clses_kpts, ys_kpts, xs_kpts, ys_kpts_indices, xs_kpts_indices = self._topk(heat_kpts)
        reg_kpts = self._tranpose_and_gather_feat(reg_kpts, inds_kpts)
        reg_kpts = reg_kpts.view(batch_kpts, self.K, 2)
        xs_kpts = xs_kpts.view(batch_kpts, self.K, 1) + reg_kpts[:, :, 0:1]
        ys_kpts = ys_kpts.view(batch_kpts, self.K, 1) + reg_kpts[:, :, 1:2]
        scores_kpts = scores_kpts.view(batch_kpts, self.K, 1)
        
        detections = torch.cat([xs,                      # cen_x
                                ys,                      # cen_y
                                tt_x,
                                tt_y,
                                rr_x,
                                rr_y,
                                bb_x,
                                bb_y,
                                ll_x,
                                ll_y,
                                scores,
                                clses],
                               dim=2)

        detections_kpts = torch.cat([xs_kpts, ys_kpts, scores_kpts], dim=2)

        index_kpts = (scores_kpts>self.conf_thresh).squeeze(0).squeeze(1)
        detections_kpts = detections_kpts[:,index_kpts,:]

        index = (scores>self.conf_thresh).squeeze(0).squeeze(1)
        detections = detections[:,index,:]

        center_indices_hm = torch.cat([xs_indices, ys_indices], dim=0)
        kpts_indices_hm = torch.cat([xs_kpts_indices, ys_kpts_indices], dim=0)
        
        # hm indices from points
        center_indices_hm = center_indices_hm[:, index].T
        kpts_indices_hm = kpts_indices_hm[:, index_kpts].T

        # coordinates
        centers_edge_points = detections[:, :, :2].squeeze(0)
        kpts_edge_points = detections_kpts[:, :, :2].squeeze(0)

        num_centers = centers_edge_points.shape[0]
        num_kpts = kpts_edge_points.shape[0]

        if num_centers > 0 and num_kpts > 0:
            # hm indices from heatmaps
            centers = torch.nonzero(heat[0][0]>self.conf_thresh).flip(-1).float()
            keypoints = torch.nonzero(heat_kpts[0][0]>self.conf_thresh).flip(-1).float()

            comparison_centers = (centers.unsqueeze(1) == center_indices_hm.unsqueeze(0))
            match_matrix_centers = torch.all(comparison_centers, dim=2)
            indices_to_reorder_centers = torch.argmax(match_matrix_centers.int(), dim=1)
            # reordered_centers_indices = center_indices_hm[indices_to_reorder_centers]
            reordered_centers_coords = centers_edge_points[indices_to_reorder_centers]

            comparison_keypoints = (keypoints.unsqueeze(1) == kpts_indices_hm.unsqueeze(0))
            match_matrix_keypoints = torch.all(comparison_keypoints, dim=2)
            indices_to_reorder_keypoints = torch.argmax(match_matrix_keypoints.int(), dim=1)
            # reordered_keypoints_indices = kpts_indices_hm[indices_to_reorder_keypoints]
            reordered_keypoints_coords = kpts_edge_points[indices_to_reorder_keypoints]

            center_repeated = reordered_centers_coords.repeat_interleave(num_kpts, dim=0)
            kpts_tiled = reordered_keypoints_coords.repeat(num_centers, 1)
            all_pairs = torch.stack((center_repeated, kpts_tiled), dim=1)
            
            valid_edges_mask = edges > self.conf_thresh
            valid_edges = all_pairs[valid_edges_mask]
            valid_edges = valid_edges.view(valid_edges.shape[0], 4)
            valid_scores = edges[valid_edges_mask].unsqueeze(-1)
            valid_edges_with_scores = torch.cat((valid_edges, valid_scores), dim=1)
        else:
            valid_edges_with_scores = torch.zeros((0, 5), device=edges.device)

        return detections.data.cpu().numpy(), detections_kpts.data.cpu().numpy(), valid_edges_with_scores.cpu().numpy()


    def visualize_heatmap_and_corners(self, hm, hm_kpts, edges):
        import cv2
        import os
        import numpy as np
        import matplotlib.pyplot as plt

        grid_h, grid_w = hm.shape[2], hm_kpts.shape[3]
        batch_size = 1  # Assuming batch size is 1
        edges = (edges > self.conf_thresh)

        for batch_idx in range(batch_size):

            black_image = np.zeros((grid_h * 4, grid_w * 4, 3), dtype=np.uint8)

            centers = np.argwhere(hm[batch_idx][0].cpu().numpy() > self.conf_thresh)
            keypoints = np.argwhere(hm_kpts[batch_idx][0].cpu().numpy() > self.conf_thresh)

            scale_x, scale_y = 4, 4

            centers_scaled = [(x * scale_x, y * scale_y) for y, x in centers]
            keypoints_scaled = [(x * scale_x, y * scale_y) for y, x in keypoints]

            for x, y in centers_scaled:  # Plot center points (red)
                cv2.circle(black_image, (x, y), 3, (255, 255, 255), -1)

            for x, y in keypoints_scaled:  # Plot keypoints (blue)
                cv2.circle(black_image, (x, y), 3, (0, 255, 0), -1)

            all_pairs = [(c, k) for c in centers_scaled for k in keypoints_scaled]

            for idx, (center, keypoint) in enumerate(all_pairs):
                if edges.view(-1)[idx]:  # Check if this edge exists in edges tensor
                    cv2.line(black_image, center, keypoint, (255, 0, 0), 1)

            output_path = os.path.join("/mnt/c/src/BBAVectors-Oriented-Object-Detection/temp_images", f"heatmap_test_{batch_idx}.png")
            cv2.imwrite(output_path, black_image)
