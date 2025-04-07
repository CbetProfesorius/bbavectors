import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from .model_parts import CombinationModule
from . import resnet


class CTRBOX(nn.Module):
    def __init__(self, heads, pretrained, down_ratio, down_ratio_kpts, final_kernel, head_conv):
        super(CTRBOX, self).__init__()
        self.channels = [3, 64, 256, 512, 1024, 2048]
        assert down_ratio in [2, 4, 8, 16]
        assert down_ratio_kpts in [2, 4, 8, 16]
        self.l1 = int(np.log2(down_ratio))
        self.base_network = resnet.resnet101(pretrained=pretrained)
        self.dec_c2 = CombinationModule(512, 256, batch_norm=True)
        self.dec_c3 = CombinationModule(1024, 512, batch_norm=True)
        self.dec_c4 = CombinationModule(2048, 1024, batch_norm=True)
        self.heads = heads

        self.head_modules = nn.ModuleDict()


        for head in self.heads:
            classes = self.heads[head]
            
            if head == 'wh':
                fc = nn.Sequential(nn.Conv2d(self.channels[self.l1], head_conv, kernel_size=3, padding=1, bias=True),
                                   nn.BatchNorm2d(head_conv),   # BN not used in the paper, but would help stable training
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(head_conv, classes, kernel_size=3, padding=1, bias=True))
            else:
                if 'hm_kpts' in head or 'reg_kpts' in head:
                    fc = nn.Sequential(nn.Conv2d(self.channels[self.l1], head_conv, kernel_size=3, padding=1, bias=True),
                                    nn.BatchNorm2d(head_conv),   # BN not used in the paper, but would help stable training
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(head_conv, classes, kernel_size=final_kernel, stride=1, padding=final_kernel // 2, bias=True))
                else:
                    fc = nn.Sequential(nn.Conv2d(self.channels[self.l1], head_conv, kernel_size=3, padding=1, bias=True),
                                    nn.BatchNorm2d(head_conv),   # BN not used in the paper, but would help stable training
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(head_conv, classes, kernel_size=final_kernel, stride=1, padding=final_kernel // 2, bias=True))
                    
            if 'hm' in head or 'hm_kpts' in head:
                fc[-1].bias.data.fill_(-2.19)
            else:
                self.fill_fc_weights(fc)

            self.__setattr__(head, fc)

        feat_dim = (self.channels[self.l1] + 2) * 2 + 2 + 1
        
        self.classifier = nn.Sequential(
                        nn.Linear(feat_dim, int(feat_dim//2)),
                        nn.BatchNorm1d(int(feat_dim//2)),
                        nn.ReLU(),
                        nn.Dropout(0.5),
                        nn.Linear(int(feat_dim//2), int(feat_dim//4)),
                        nn.BatchNorm1d(int(feat_dim//4)),
                        nn.ReLU(),
                        nn.Dropout(0.5),
                        nn.Linear(int(feat_dim//4), 1)
                    )

        # self.classifier[-1].bias.data.fill_(-2.19)

    def fill_fc_weights(self, m):
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _nms(self, heat, kernel=3):
        hmax = F.max_pool2d(heat, (kernel, kernel), stride=1, padding=(kernel - 1) // 2)
        keep = (hmax == heat).float()
        return heat * keep
    
    def reshape_with_indices(self, tensor):
        batch, features, width, height = tensor.shape
        
        # Create row and column index tensors
        cols = torch.arange(width, device=tensor.device).view(1, 1, width, 1).expand(batch, 1, width, height)
        rows = torch.arange(height, device=tensor.device).view(1, 1, 1, height).expand(batch, 1, width, height)

        norm_cols = cols.float() / (width - 1)
        norm_rows = rows.float() / (height - 1)
        
        # Concatenate indices to the original tensor along the feature dimension
        tensor_with_indices = torch.cat([tensor, norm_cols, norm_rows], dim=1)
        
        # Reshape to (Batch, Features, Width * Height)
        tensor_reshaped = tensor_with_indices.view(batch, features + 2, width * height)

        return tensor_reshaped

    def get_edges(self, gt_batch):
        center_corners_indices = gt_batch['center_corners_indices']
        batch_size = center_corners_indices.shape[0] 
        
        
        indices_bbox = []
        indices_kpts = []
        edge_labels = []
        
        for i in range(batch_size):
            centers = torch.nonzero(gt_batch['hm'][i].flatten() == 1).flatten()
            keypoints = torch.nonzero(gt_batch['hm_kpts'][i].flatten() == 1).flatten()
            
            hm_height, hm_width = gt_batch['hm'][i][0].shape
            hm_kpts_height, hm_kpts_width = gt_batch['hm_kpts'][i][0].shape

            indices_bbox.append(centers)
            indices_kpts.append(keypoints)

            bbox_exp = centers.unsqueeze(1).expand(-1, keypoints.shape[0])  # [num_centers, num_keypoints]
            kpts_exp = keypoints.unsqueeze(0).expand(centers.shape[0], -1)  # [num_centers, num_keypoints]

            valid_pairs = torch.stack((bbox_exp, kpts_exp), dim=-1).reshape(-1, 2)  # [num_pairs, 2]
            
            num_pairs = valid_pairs.shape[0]
            batch_edges = torch.zeros((num_pairs,), dtype=torch.float32, device=gt_batch['hm'].device)

            coords = center_corners_indices[i][:centers.shape[0], :, :]  # (num_centers, 5, 2)

            for row_idx in range(coords.shape[0]):
                center_x, center_y = int(coords[row_idx, 0, 0]), int(coords[row_idx, 0, 1])

                for kpt_idx in range(1, 5):  # keypoints are indexed from 1 to 4
                    kpt_x, kpt_y = int(coords[row_idx, kpt_idx, 0]), int(coords[row_idx, kpt_idx, 1])

                    # Check if this pair exists in our valid combinations
                    for pair_idx, (c_idx, k_idx) in enumerate(valid_pairs):
                        center_comb_y, center_comb_x = torch.div(c_idx, hm_width, rounding_mode='floor'), c_idx % hm_width
                        center_comb_y, center_comb_x = int(center_comb_y), int(center_comb_x)
                        kpt_comb_y, kpt_comb_x = torch.div(k_idx, hm_kpts_width, rounding_mode='floor'), k_idx % hm_kpts_width
                        kpt_comb_y, kpt_comb_x = int(kpt_comb_y), int(kpt_comb_x)

                        if ((int(center_comb_x) == int(center_x) and int(center_comb_y) == int(center_y) and
                            int(kpt_comb_x) == int(kpt_x) and int(kpt_comb_y) == int(kpt_y)) or 
                            (int(center_comb_x) == int(kpt_x) and int(center_comb_y) == int(kpt_y) and
                            int(kpt_comb_x) == int(center_x) and int(kpt_comb_y) == int(center_y)) or
                            (int(center_comb_x) == int(center_y) and int(center_comb_y) == int(center_x) and
                            int(kpt_comb_x) == int(kpt_y) and int(kpt_comb_y) == int(kpt_x)) or
                            (int(center_comb_x) == int(kpt_y) and int(center_comb_y) == int(kpt_x) and
                            int(kpt_comb_x) == int(center_y) and int(kpt_comb_y) == int(center_x))):
                            
                            batch_edges[pair_idx] = 1.0
                            break

            edge_labels.append(batch_edges)

        edge_labels = torch.cat(edge_labels, dim=0)  # [total_valid_pairs]

        return edge_labels
    
    def forward(self, x, gt_batch=None, phase=None):
        x = self.base_network(x)

        c4_combine = self.dec_c4(x[-1], x[-2])
        c3_combine = self.dec_c3(c4_combine, x[-3])
        # c2_combine = self.dec_c2(c3_combine, x[-4])

        feature_map = c3_combine

        dec_dict = {}
        for head in self.heads:
            dec_dict[head] = self.__getattr__(head)(feature_map)
            if 'hm' in head or 'hm_kpts' in head or 'cls' in head:
                dec_dict[head] = torch.sigmoid(dec_dict[head])

        flat_features = self.reshape_with_indices(feature_map) # [2, 512, w*h]

        if phase in ["train", "val"]:
            target_bbox_hm = (gt_batch['hm']==1).float()
            target_kpts_hm = (gt_batch['hm_kpts']==1).float()
        else:
            target_bbox_hm = self._nms(dec_dict['hm'], kernel=3)
            target_bbox_hm = (target_bbox_hm > 0.5)
            target_kpts_hm = self._nms(dec_dict['hm_kpts'], kernel=3)
            target_kpts_hm = (target_kpts_hm > 0.5)
        
        batch = flat_features.shape[0]

        def get_features(flat_features, target_hm, max_point_count=40):        
            indices = torch.zeros((batch, max_point_count), dtype=torch.long, device=target_hm.device)
            valid_counts = torch.zeros(batch, dtype=torch.long, device=target_hm.device)
            for i, target in enumerate(target_hm):
                target_indices = torch.nonzero(target.flatten() == 1).flatten()
                # print(len(target_indices))
                # self.visualize_heatmap_and_corners(gt_batch)
                valid_counts[i] = len(target_indices)
                assert len(target_indices) < max_point_count
                indices[i, :len(target_indices)] = target_indices
            indices = indices.unsqueeze(1)
            features = torch.gather(flat_features, dim=2, index=indices.expand(-1, flat_features.shape[1], -1))
            mask = (indices == 0).expand(-1, flat_features.shape[1], -1)
            return features.masked_fill(mask, 0.0), valid_counts
            # return features
        
        bbox_center_features, valid_bbox_counts = get_features(flat_features, target_bbox_hm, max_point_count=20) # shape [B, 514, 20]  514 = 512 (features) + 2 (global coords)
        keypoints_features, valid_kpts_counts = get_features(flat_features, target_kpts_hm, max_point_count=40)
        
        feat_dim = bbox_center_features.shape[1] # 514

        # bbox_keypoint_combinations = torch.zeros((batch, 20 * 40, feat_dim), device=flat_features.device)
        bbox_keypoint_combinations = []

        for i in range(batch):
            # bbox_feat = bbox_center_features[i].permute(1, 0)    # [20, 514]
            # kpts_feat = keypoints_features[i].permute(1, 0)      # [50, 514]

            valid_bbox_feat = bbox_center_features[i, :, :valid_bbox_counts[i]].permute(1, 0)  # [valid_bboxes, 514]
            valid_kpts_feat = keypoints_features[i, :, :valid_kpts_counts[i]].permute(1, 0)   # [valid_kpts, 514]

            bbox_feat_exp = valid_bbox_feat.unsqueeze(1).expand(-1, valid_kpts_counts[i], -1)  # [valid_bboxes, valid_kpts, 514]
            kpts_feat_exp = valid_kpts_feat.unsqueeze(0).expand(valid_bbox_counts[i], -1, -1)  # [valid_bboxes, valid_kpts, 514]

            center_coords = valid_bbox_feat[:, -2:]
            kpts_coords = valid_kpts_feat[:, -2:]
            center_coords_exp = center_coords.unsqueeze(1).expand(-1, valid_kpts_counts[i], -1)
            kpts_coords_exp = kpts_coords.unsqueeze(0).expand(valid_bbox_counts[i], -1, -1)
            relative_coords = kpts_coords_exp - center_coords_exp # Shape [valid_bboxes, valid_kpts, 2]
            distance = torch.norm(relative_coords, p=2, dim=-1, keepdim=True)
            combined_features = torch.cat([bbox_feat_exp, kpts_feat_exp, relative_coords, distance], dim=-1)
            bbox_keypoint_combinations.append(combined_features.reshape(-1, feat_dim*2+2+1))

            # Create cartesian combination: [20, 50, 514] -> flatten to [20*50, 514]
            # bbox_feat_exp = bbox_feat.unsqueeze(1).expand(-1, 40, -1)  # [20, 50, 514]
            # kpts_feat_exp = kpts_feat.unsqueeze(0).expand(20, -1, -1)  # [20, 50, 514]

            # hadamard_product = bbox_feat_exp * kpts_feat_exp  # [valid_bboxes, valid_kpts, 514]
            # bbox_keypoint_combinations.append(hadamard_product.reshape(-1, feat_dim))

            # hadamard_product = bbox_feat_exp * kpts_feat_exp  # [20, 40, 514]
            # bbox_keypoint_combinations[i] = hadamard_product.reshape(-1, feat_dim)

            # combined = torch.cat([bbox_feat_exp, kpts_feat_exp], dim=-1)  # [20, 50, 1028]
            # bbox_keypoint_combinations.append(combined.reshape(-1, feat_dim*2))

            # bbox_keypoint_combinations[i] = combined.reshape(-1, feat_dim * 2)  # [1000, 1028]

        bbox_keypoint_combinations = torch.cat(bbox_keypoint_combinations, dim=0)

        # shape = [B, 1000, 1028]
        # Flatten to [B*1000, 1028] to pass through MLP
        # bbox_keypoint_combinations = bbox_keypoint_combinations.view(-1, feat_dim)  # [B*1000, 1028]

        # bbox_feat_exp = bbox_center_features.permute(0, 2, 1).unsqueeze(2).expand(-1, -1, 40, -1)
        # kpts_feat_exp = keypoints_features.permute(0, 2, 1).unsqueeze(1).expand(-1, 20, -1, -1)

        # bbox_feat_exp = bbox_feat_exp.contiguous().view(-1, feat_dim)
        # kpts_feat_exp = kpts_feat_exp.contiguous().view(-1, feat_dim)

        # pred_logits = self.link_predictor(bbox_feat_exp, kpts_feat_exp).squeeze(-1).view(-1)

        pred_logits = self.classifier(bbox_keypoint_combinations).squeeze(-1)  # [B*1000]
        pred_probs = torch.sigmoid(pred_logits)

        dec_dict['edges'] = pred_probs

        if phase in ["train", "val"]:
            edge_labels = self.get_edges(gt_batch)
            num_pos = edge_labels.sum()
            num_neg = edge_labels.numel() - num_pos
            pos_weight = num_neg / (num_pos + 1e-6)
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=pred_logits.device))
            loss = loss_fn(pred_logits, edge_labels)

            acc = {}
            
            for thrs in [0.3, 0.5, 0.75]:
                pred_labels = (pred_probs > thrs).float() 

                num_correct = (pred_labels == edge_labels).sum() 

                accuracy = num_correct / edge_labels.numel()

                acc[f'edges_acc_{thrs}'] = accuracy

            return dec_dict, loss, acc
        else:
            return dec_dict


    def forward_full_onnx(self, x):
        x = self.base_network(x)

        c4_combine = self.dec_c4(x[-1], x[-2])
        c3_combine = self.dec_c3(c4_combine, x[-3])
        # c2_combine = self.dec_c2(c3_combine, x[-4])

        feature_map = c3_combine

        output_tensors = []

        dec_dict = {}
        for head in self.heads:
            dec_dict[head] = self.__getattr__(head)(feature_map)
            if 'hm' in head or 'hm_kpts' in head or 'cls' in head:
                dec_dict[head] = torch.sigmoid(dec_dict[head])

            output_tensors.append(dec_dict[head])

        B, C_feat, H, W = feature_map.shape

        # Note: reshape_with_indices expects [B, C, H, W]
        flat_features = self.reshape_with_indices(feature_map)
        # flat_features shape: [B, C_feat + 2, H * W]
        C_feat_coord = flat_features.shape[1] # Should be self.feature_channels + 2

        # 4. Candidate Selection using TopK
        hm = dec_dict['hm']       # Shape [B, num_classes_hm, H, W]
        hm_kpts = dec_dict['hm_kpts'] # Shape [B, num_classes_kpts, H, W]

        # Assuming single class for simplicity of topk selection
        # If multi-class, need max across class dim or handle differently
        hm_flat = hm.view(B, -1, H * W).max(dim=1)[0] # Max score per location [B, H*W]
        hm_kpts_flat = hm_kpts.view(B, -1, H * W).max(dim=1)[0] # Max score per location [B, H*W]

        # Get top k indices (flat indices in the H*W grid)
        # Using scores from sigmoid'd heatmaps
        _, bbox_topk_indices = torch.topk(hm_flat, k=20, dim=1) # Shape [B, 20]
        _, kpts_topk_indices = torch.topk(hm_kpts_flat, k=40, dim=1) # Shape [B, k_kpts]

        # 5. Fixed-Size Feature Gathering
        # Expand indices to match feature dimension
        bbox_indices_expanded = bbox_topk_indices.unsqueeze(1).expand(-1, C_feat_coord, -1) # Shape [B, C_feat_coord, k_bbox]
        kpts_indices_expanded = kpts_topk_indices.unsqueeze(1).expand(-1, C_feat_coord, -1) # Shape [B, C_feat_coord, k_kpts]

        # Gather features corresponding to topk indices
        # flat_features: [B, C_feat_coord, H * W]
        bbox_center_features = torch.gather(flat_features, dim=2, index=bbox_indices_expanded) # Shape [B, C_feat_coord, k_bbox]
        keypoints_features = torch.gather(flat_features, dim=2, index=kpts_indices_expanded) # Shape [B, C_feat_coord, k_kpts]

        # 6. Fixed-Size Feature Combination (Cartesian Product)
        # Permute to [B, k, C_feat_coord] for easier processing
        bbox_feat_p = bbox_center_features.permute(0, 2, 1) # Shape [B, k_bbox, C_feat_coord]
        kpts_feat_p = keypoints_features.permute(0, 2, 1) # Shape [B, k_kpts, C_feat_coord]

        # Expand features for all pairs [B, k_bbox, k_kpts, C_feat_coord]
        bbox_feat_exp = bbox_feat_p.unsqueeze(2).expand(-1, -1, 40, -1)
        kpts_feat_exp = kpts_feat_p.unsqueeze(1).expand(-1, 20, -1, -1)

        # Extract coordinates (last 2 channels from gathered features)
        center_coords = bbox_feat_p[..., -2:] # Shape [B, 20, 2]
        kpts_coords = kpts_feat_p[..., -2:] # Shape [B, k_kpts, 2]

        # Expand coordinates for all pairs [B, 20, k_kpts, 2]
        center_coords_exp = center_coords.unsqueeze(2).expand(-1, -1, 40, -1)
        kpts_coords_exp = kpts_coords.unsqueeze(1).expand(-1, 20, -1, -1)

        # Calculate relative coordinates and distance
        relative_coords = kpts_coords_exp - center_coords_exp # Shape [B, k_bbox, k_kpts, 2]
        distance = torch.norm(relative_coords, p=2, dim=-1, keepdim=True) # Shape [B, k_bbox, k_kpts, 1]

        # Concatenate all features for the classifier
        # [bbox_feat, kpt_feat, rel_coords, distance]
        combined_features = torch.cat([bbox_feat_exp, kpts_feat_exp, relative_coords, distance], dim=-1)
        # Expected shape: [B, k_bbox, 40, C_feat_coord*2 + 2 + 1]
        B_exp, K_bbox_exp, K_kpts_exp, C_combined = combined_features.shape

        # 7. Classifier Application
        # Reshape for MLP: [B * k_bbox * k_kpts, C_combined]
        combined_features_flat = combined_features.view(-1, C_combined)

        # Apply classifier
        pred_logits_flat = self.classifier(combined_features_flat) # Shape [B * k_bbox * k_kpts, 1]

        # Reshape output logits/probabilities
        pred_logits = pred_logits_flat.view(B, 20, 40) # Shape [B, k_bbox, k_kpts]
        pred_probs = torch.sigmoid(pred_logits) # Final edge probabilities

        output_tensors.append(pred_probs)

        return tuple(output_tensors)
        

    def visualize_feature_combinations_graph(self, bbox_center_features, keypoints_features, filename="feature_combinations_graph.png"):
        import cv2
        import os
        import random
        import networkx as nx
        import matplotlib.pyplot as plt

        batch = bbox_center_features.shape[0]

        for i in range(batch):
            bbox_feat = bbox_center_features[i].permute(1, 0).detach().cpu().numpy()  # [20, 514]
            kpts_feat = keypoints_features[i].permute(1, 0).detach().cpu().numpy()  # [40, 514]

            G = nx.Graph()
            nodes = []
            edges = []

            # Add nodes for bounding box centers
            for j in range(bbox_feat.shape[0]):
                node_id = f"b{j}"
                G.add_node(node_id, type="bbox")
                nodes.append(node_id)

            # Add nodes for keypoints
            for j in range(kpts_feat.shape[0]):
                node_id = f"k{j}"
                G.add_node(node_id, type="kpts")
                nodes.append(node_id)

            # Add edges for all combinations
            for bbox_idx in range(bbox_feat.shape[0]):
                for kpts_idx in range(kpts_feat.shape[0]):
                    G.add_edge(f"b{bbox_idx}", f"k{kpts_idx}")
                    edges.append((f"b{bbox_idx}", f"k{kpts_idx}"))

            # Visualize the graph
            pos = nx.spring_layout(G)  # Layout the nodes
            node_colors = ["red" if G.nodes[node]["type"] == "bbox" else "blue" for node in G.nodes()]

            plt.figure(figsize=(12, 12))
            nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=500, font_size=8, edge_color="gray")
            plt.title(f"Feature Combinations Graph - Batch {i}")
            plt.savefig(f"{filename.replace('.png', f'_batch_{i}.png')}")
            plt.close()


    def visualize_feature_combinations(self, bbox_center_features, keypoints_features, filename="feature_combinations.png"):
        import matplotlib.pyplot as plt

        batch = bbox_center_features.shape[0]
        feat_dim = bbox_center_features.shape[1]

        for i in range(batch):
            bbox_feat = bbox_center_features[i].permute(1, 0).detach().cpu().numpy()  # [20, 514]
            kpts_feat = keypoints_features[i].permute(1, 0).detach().cpu().numpy()  # [40, 514]

            # Create cartesian combination: [20, 50, 514] -> flatten to [20*50, 514]
            bbox_feat_exp = np.expand_dims(bbox_feat, axis=1)  # [20, 1, 514]
            bbox_feat_exp = np.tile(bbox_feat_exp, (1, 40, 1))  # [20, 40, 514]

            kpts_feat_exp = np.expand_dims(kpts_feat, axis=0)  # [1, 40, 514]
            kpts_feat_exp = np.tile(kpts_feat_exp, (20, 1, 1))  # [20, 40, 514]

            combined = np.concatenate([bbox_feat_exp, kpts_feat_exp], axis=-1)  # [20, 40, 1028]
            combined_reshaped = combined.reshape(-1, feat_dim * 2)  # [800, 1028]

            # Visualize the combined features (e.g., as a heatmap)
            plt.figure(figsize=(10, 10))
            plt.imshow(combined_reshaped, aspect='auto', cmap='viridis')
            plt.title(f"Combined Features - Batch {i}")
            plt.colorbar()
            plt.savefig(f"{filename.replace('.png', f'_batch_{i}.png')}")
            plt.close()



    def visualize_heatmap_and_corners(self, gt_batch):
        import cv2
        import os
        import random
        import networkx as nx
        import matplotlib.pyplot as plt

        grid_h, grid_w = gt_batch['hm'].shape[2], gt_batch['hm'].shape[3]
        center_corners_indices = gt_batch['center_corners_indices']
        batch_size = center_corners_indices.shape[0]

        for batch_idx in range(batch_size):
            
            num_rows = int((gt_batch['hm'][batch_idx]==1).float().sum())
            coords = center_corners_indices[batch_idx][:num_rows, :, :]

            G = nx.Graph()
            edge_colors = [] 
            node_colors = {}
            edges = []

            grid_image = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

            for row_idx in range(coords.shape[0]):
                center_x, center_y = int(coords[row_idx, 0, 0]), int(coords[row_idx, 0, 1])
                cv2.circle(grid_image, (center_x, center_y), 1, (0, 0, 255), -1)

                G.add_node((center_x, center_y))
                node_colors[(center_x, center_y)] = 'red'

                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                edge_color = (color[0]/255.0, color[1]/255.0, color[2]/255.0)

                for kpt_idx in range(1, 5):
                    kpt_x, kpt_y = int(coords[row_idx, kpt_idx, 0]), int(coords[row_idx, kpt_idx, 1])
                    cv2.circle(grid_image, (kpt_x, kpt_y), 1, (255, 0, 0), -1)
                    cv2.line(grid_image, (center_x, center_y), (kpt_x, kpt_y), color, 1)
                    G.add_node((kpt_x, kpt_y))  # Add corner nodes
                    G.add_edge((center_x, center_y), (kpt_x, kpt_y))
                    edges.append(((center_x, center_y), (kpt_x, kpt_y)))
                    edge_colors.append(edge_color)  # Assign color to the edge
                    node_colors[(kpt_x, kpt_y)] = (59/255.0, 15/255.0, 36/255.0)

            grid_image = cv2.flip(grid_image, 0)
            output_path = os.path.join("/home/ecognition/git/bbavectors/temp_images", f"grid_points_batch_{batch_idx}.png")
            cv2.imwrite(output_path, cv2.resize(grid_image, (grid_w * 4, grid_h * 4), interpolation=cv2.INTER_NEAREST))

            node_color_list = [node_colors[node] for node in G.nodes()]
            plt.figure(figsize=(8, 8))
            pos = {node: node for node in G.nodes()}  # Position nodes based on their coordinates
            nx.draw(G, pos, with_labels=False, node_size=50, node_color=node_color_list, edge_color=edge_colors, width=2, alpha=0.8)

            output_path = os.path.join("/home/ecognition/git/bbavectors/temp_images", f"graph_batch_{batch_idx}.jpg")
            plt.savefig(output_path, format="jpg")
            plt.close()