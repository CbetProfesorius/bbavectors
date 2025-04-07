# <<< NumPy ONLY Version - Single Image Inference - Dynamic Outputs >>>
# Required imports
import numpy as np
import cv2
import onnxruntime as ort
import os
import time
from scipy.ndimage import maximum_filter
# from datasets.DOTA_devkit.ResultMerge_multi_process import py_cpu_nms_poly_fast # Assumed to exist
import argparse

# --- Mock NMS function (Replace with your actual implementation) ---
# This is a placeholder. You NEED your actual 'py_cpu_nms_poly_fast'
def py_cpu_nms_poly_fast(dets, thresh):
    """
    Placeholder for the actual Cython/Python polygon NMS function.
    Input:
        dets: np.array [N, 9] (poly_coords[8], score) - MUST BE FLOAT64 for typical Cython implementations
        thresh: float, IoU threshold
    Output:
        keep_indices: list or np.array of indices to keep
    """
    # print("WARNING: Using MOCK NMS function. Replace with your actual 'py_cpu_nms_poly_fast'.")
    if dets.shape[0] == 0:
        return []
    # Extremely simple NMS based on score for demonstration ONLY
    scores = dets[:, 8]
    keep_indices = np.argsort(scores)[::-1] # Sort by score desc

    # --- A more realistic mock NMS step (still simplified) ---
    final_keep = []
    suppressed = np.zeros(len(keep_indices), dtype=bool)
    for i in range(len(keep_indices)):
        if suppressed[i]:
            continue
        final_keep.append(keep_indices[i])
        current_box_poly = dets[keep_indices[i], :8].reshape(4, 2)
        current_center = np.mean(current_box_poly, axis=0)
        for j in range(i + 1, len(keep_indices)):
            if suppressed[j]:
                continue
            other_box_poly = dets[keep_indices[j], :8].reshape(4, 2)
            other_center = np.mean(other_box_poly, axis=0)
            dist_sq = np.sum((current_center - other_center)**2)
            closeness_threshold_sq = (50*50) # Example threshold
            if dist_sq < closeness_threshold_sq:
                suppressed[j] = True
    return final_keep

# --- Helper Functions (Unchanged) ---
def _np_nms(heat, kernel=3):
    pad = (kernel - 1) // 2
    hmax = maximum_filter(heat, size=kernel, mode='constant', cval=-np.inf)
    keep = (heat == hmax).astype(np.float32)
    return heat * keep

def _np_topk(scores, K):
    batch, height, width = scores.shape
    scores_flat = scores.reshape(batch, -1)
    num_elements = height * width
    actual_K = min(K, num_elements)
    if actual_K == 0:
        return (np.zeros((batch, 0), dtype=scores.dtype),
                np.zeros((batch, 0), dtype=np.int64),
                np.zeros((batch, 0), dtype=np.float32),
                np.zeros((batch, 0), dtype=np.float32))
    topk_inds_flat = np.argpartition(scores_flat, -actual_K, axis=1)[:, -actual_K:]
    topk_scores = np.take_along_axis(scores_flat, topk_inds_flat, axis=1)
    sorted_idx_within_topk = np.argsort(topk_scores, axis=1)[:, ::-1]
    topk_inds_flat = np.take_along_axis(topk_inds_flat, sorted_idx_within_topk, axis=1)
    topk_scores = np.take_along_axis(topk_scores, sorted_idx_within_topk, axis=1)
    topk_ys = (topk_inds_flat // width).astype(np.float32)
    topk_xs = (topk_inds_flat % width).astype(np.float32)
    return topk_scores, topk_inds_flat, topk_ys, topk_xs


def _np_gather_feat(feat, ind):
    batch, num_elements, channels = feat.shape
    K = ind.shape[1]
    if K == 0:
        return np.zeros((batch, 0, channels), dtype=feat.dtype)
    ind_expanded = np.expand_dims(ind, axis=2).repeat(channels, axis=2)
    gathered_feat = np.take_along_axis(feat, ind_expanded.astype(np.int64), axis=1)
    return gathered_feat

def _np_tranpose_and_gather_feat(feat, ind):
    batch, channels, height, width = feat.shape
    feat = feat.transpose(0, 2, 3, 1) # B, H, W, C
    feat = feat.reshape(batch, -1, channels) # B, H*W, C
    feat = _np_gather_feat(feat, ind) # B, K, C
    return feat

# --- Decoding Functions (Modified Signature) ---
# Accepts booleans directly now instead of args object
def numpy_decode_boxes_kpts(pr_decs_np, K_decode, conf_thresh, has_cls_theta, has_keypoints):
    """
    Decodes boxes and keypoints from ONNX outputs using NumPy BEFORE NMS.
    Returns results on the FEATURE MAP scale.
    Requires 'hm', 'wh', 'reg' in pr_decs_np.
    Conditionally uses 'cls_theta', 'hm_kpts', 'reg_kpts'.
    """
    # Check minimum required inputs
    required_base = ['hm', 'wh', 'reg']
    if not all(key in pr_decs_np for key in required_base):
        missing = [key for key in required_base if key not in pr_decs_np]
        raise ValueError(f"Cannot decode boxes. Missing required base ONNX outputs: {missing}")

    heat = pr_decs_np['hm'][0]
    wh = pr_decs_np['wh']
    reg = pr_decs_np['reg']

    cls_theta = None
    if has_cls_theta:
        if 'cls_theta' not in pr_decs_np:
             # This case should ideally be caught earlier by checking session outputs
             print("Warning: 'has_cls_theta' is true, but 'cls_theta' not in ONNX results dictionary.")
             has_cls_theta = False # Disable theta decoding path
        else:
             cls_theta = pr_decs_np['cls_theta']

    heat_kpts = None
    reg_kpts = None
    if has_keypoints:
        if 'hm_kpts' not in pr_decs_np or 'reg_kpts' not in pr_decs_np:
             print("Warning: 'has_keypoints' is true, but 'hm_kpts' or 'reg_kpts' not in ONNX results dictionary.")
             has_keypoints = False # Disable keypoint decoding path
        else:
            heat_kpts = pr_decs_np['hm_kpts'][0]
            reg_kpts = pr_decs_np['reg_kpts']

    if heat.shape[0] > 1:
        print(f"Warning: 'hm' has {heat.shape[0]} classes, using class 0 for box detection.")
    heat = heat[0]

    # --- Boxes ---
    heat_nms = _np_nms(heat)
    scores, inds, ys, xs = _np_topk(heat_nms[np.newaxis,...], K_decode)
    scores, inds, ys, xs = scores[0], inds[0], ys[0], xs[0]

    keep = np.where(scores >= conf_thresh)[0]
    scores, inds, ys, xs = scores[keep], inds[keep], ys[keep], xs[keep]

    detections_fm = np.zeros((0, 11), dtype=np.float32)
    if len(scores) > 0:
        inds_batch = inds[np.newaxis, ...]
        reg_g = _np_tranpose_and_gather_feat(reg, inds_batch)[0]
        wh_g = _np_tranpose_and_gather_feat(wh, inds_batch)[0]
        xs_c = xs[:, np.newaxis] + reg_g[:, 0:1]
        ys_c = ys[:, np.newaxis] + reg_g[:, 1:2]

        if has_cls_theta:
            cls_theta_g = _np_tranpose_and_gather_feat(cls_theta, inds_batch)[0]
            mask = (cls_theta_g > 0.5).astype(np.float32)
            if wh_g.shape[1] != 10:
                 raise ValueError(f"Theta decoding active, but expected 'wh' to have 10 channels, got {wh_g.shape[1]}")
            tt_x = (xs_c + wh_g[..., 0:1]) * mask + (xs_c) * (1. - mask)
            tt_y = (ys_c + wh_g[..., 1:2]) * mask + (ys_c - wh_g[..., 9:10] / 2) * (1. - mask)
            rr_x = (xs_c + wh_g[..., 2:3]) * mask + (xs_c + wh_g[..., 8:9] / 2) * (1. - mask)
            rr_y = (ys_c + wh_g[..., 3:4]) * mask + (ys_c) * (1. - mask)
            bb_x = (xs_c + wh_g[..., 4:5]) * mask + (xs_c) * (1. - mask)
            bb_y = (ys_c + wh_g[..., 5:6]) * mask + (ys_c + wh_g[..., 9:10] / 2) * (1. - mask)
            ll_x = (xs_c + wh_g[..., 6:7]) * mask + (xs_c - wh_g[..., 8:9] / 2) * (1. - mask)
            ll_y = (ys_c + wh_g[..., 7:8]) * mask + (ys_c) * (1. - mask)
            tl_x = tt_x + ll_x - xs_c; tl_y = tt_y + ll_y - ys_c
            bl_x = bb_x + ll_x - xs_c; bl_y = bb_y + ll_y - ys_c
            tr_x = tt_x + rr_x - xs_c; tr_y = tt_y + rr_y - ys_c
            br_x = bb_x + rr_x - xs_c; br_y = bb_y + rr_y - ys_c
        else: # Standard axis-aligned box decoding
             if wh_g.shape[1] != 2:
                 raise ValueError(f"Theta decoding inactive, but expected 'wh' to have 2 channels, got {wh_g.shape[1]}")
             w = wh_g[:, 0:1]; h = wh_g[:, 1:2]
             tl_x = xs_c - w / 2; tl_y = ys_c - h / 2
             tr_x = xs_c + w / 2; tr_y = ys_c - h / 2
             br_x = xs_c + w / 2; br_y = ys_c + h / 2
             bl_x = xs_c - w / 2; bl_y = ys_c + h / 2

        detections_fm = np.concatenate([
            tr_x, tr_y, br_x, br_y, bl_x, bl_y, tl_x, tl_y,
            scores[:, np.newaxis], xs_c, ys_c
        ], axis=1)

    # --- Keypoints ---
    detections_kpts_fm = np.zeros((0, 3), dtype=np.float32)
    if has_keypoints: # Check the boolean passed to the function
        if heat_kpts.shape[0] > 1:
             print(f"Warning: 'hm_kpts' has {heat_kpts.shape[0]} classes, using class 0.")
        heat_kpts = heat_kpts[0]
        heat_kpts_nms = _np_nms(heat_kpts)
        scores_kpts, inds_kpts, ys_kpts, xs_kpts = _np_topk(heat_kpts_nms[np.newaxis,...], K_decode)
        scores_kpts, inds_kpts, ys_kpts, xs_kpts = scores_kpts[0], inds_kpts[0], ys_kpts[0], xs_kpts[0]
        keep_kpts = np.where(scores_kpts >= conf_thresh)[0]
        scores_kpts, inds_kpts, ys_kpts, xs_kpts = scores_kpts[keep_kpts], inds_kpts[keep_kpts], ys_kpts[keep_kpts], xs_kpts[keep_kpts]
        if len(scores_kpts) > 0:
            inds_kpts_batch = inds_kpts[np.newaxis, ...]
            reg_kpts_g = _np_tranpose_and_gather_feat(reg_kpts, inds_kpts_batch)[0]
            xs_kpts_abs = xs_kpts[:, np.newaxis] + reg_kpts_g[:, 0:1]
            ys_kpts_abs = ys_kpts[:, np.newaxis] + reg_kpts_g[:, 1:2]
            detections_kpts_fm = np.concatenate([xs_kpts_abs, ys_kpts_abs, scores_kpts[:, np.newaxis]], axis=1)

    return detections_fm, detections_kpts_fm


# Requires pr_decs_np to contain all necessary keys if called
def numpy_decode_onnx_edges(pr_decs_np, k_bbox_edges, k_kpts_edges, edge_conf_thresh):
    """
    Decodes edge probabilities between top-k bbox and top-k kpts candidates.
    Returns edge coordinates and scores on the FEATURE MAP scale.
    Requires 'hm', 'reg', 'hm_kpts', 'reg_kpts', 'edge_probs' in pr_decs_np.
    """
    required_keys = ['hm', 'reg', 'hm_kpts', 'reg_kpts', 'edge_probs']
    if not all(key in pr_decs_np for key in required_keys):
        missing = [key for key in required_keys if key not in pr_decs_np]
        # This function should only be called if edge_probs was detected,
        # so missing other keys indicates an inconsistency.
        print(f"Error: Cannot decode edges. Missing required ONNX outputs: {missing}")
        return np.zeros((0, 5), dtype=np.float32)

    heat = pr_decs_np['hm'][0, 0]
    reg = pr_decs_np['reg']
    heat_kpts = pr_decs_np['hm_kpts'][0, 0]
    reg_kpts = pr_decs_np['reg_kpts']
    edge_probs = pr_decs_np['edge_probs'][0]

    expected_shape = (k_bbox_edges, k_kpts_edges)
    if edge_probs.shape != expected_shape:
        print(f"Warning: 'edge_probs' shape mismatch. Expected {expected_shape}, got {edge_probs.shape}. Edge decoding might be incorrect.")
        actual_k_bbox_dim = edge_probs.shape[0]
        actual_k_kpts_dim = edge_probs.shape[1]
        k_bbox_edges = min(k_bbox_edges, actual_k_bbox_dim)
        k_kpts_edges = min(k_kpts_edges, actual_k_kpts_dim)
        if k_bbox_edges == 0 or k_kpts_edges == 0:
             print("Warning: Adjusted edge K values to 0 due to shape mismatch. No edges decoded.")
             return np.zeros((0, 5), dtype=np.float32)

    _, bbox_topk_inds, bbox_topk_ys, bbox_topk_xs = _np_topk(heat[np.newaxis,...], k_bbox_edges)
    _, kpts_topk_inds, kpts_topk_ys, kpts_topk_xs = _np_topk(heat_kpts[np.newaxis,...], k_kpts_edges)
    bbox_topk_inds, bbox_topk_ys, bbox_topk_xs = bbox_topk_inds[0], bbox_topk_ys[0], bbox_topk_xs[0]
    kpts_topk_inds, kpts_topk_ys, kpts_topk_xs = kpts_topk_inds[0], kpts_topk_ys[0], kpts_topk_xs[0]

    actual_k_bbox = bbox_topk_inds.shape[0]
    actual_k_kpts = kpts_topk_inds.shape[0]
    if actual_k_bbox == 0 or actual_k_kpts == 0:
        return np.zeros((0, 5), dtype=np.float32)

    reg_bbox_g = _np_tranpose_and_gather_feat(reg, bbox_topk_inds[np.newaxis,...])[0]
    refined_bbox_xs = bbox_topk_xs[:, np.newaxis] + reg_bbox_g[:, 0:1]
    refined_bbox_ys = bbox_topk_ys[:, np.newaxis] + reg_bbox_g[:, 1:2]
    refined_bbox_coords = np.concatenate([refined_bbox_xs, refined_bbox_ys], axis=1)

    reg_kpts_g = _np_tranpose_and_gather_feat(reg_kpts, kpts_topk_inds[np.newaxis,...])[0]
    refined_kpts_xs = kpts_topk_xs[:, np.newaxis] + reg_kpts_g[:, 0:1]
    refined_kpts_ys = kpts_topk_ys[:, np.newaxis] + reg_kpts_g[:, 1:2]
    refined_kpts_coords = np.concatenate([refined_kpts_xs, refined_kpts_ys], axis=1)

    valid_edge_probs = edge_probs[:actual_k_bbox, :actual_k_kpts]
    edge_indices = np.argwhere(valid_edge_probs >= edge_conf_thresh)

    valid_edges_fm = np.zeros((0, 5), dtype=np.float32)
    if edge_indices.shape[0] > 0:
        bbox_indices = edge_indices[:, 0]
        kpts_indices = edge_indices[:, 1]
        edge_start_coords = refined_bbox_coords[bbox_indices]
        edge_end_coords = refined_kpts_coords[kpts_indices]
        edge_scores = valid_edge_probs[bbox_indices, kpts_indices]
        valid_edges_fm = np.concatenate([edge_start_coords, edge_end_coords, edge_scores[:, np.newaxis]], axis=1)

    return valid_edges_fm


# --- Scaling Functions (Unchanged) ---
def scale_poly_boxes(detections_fm, input_w, input_h, img_w, img_h, down_ratio):
    scaled_boxes_img = np.zeros((0, 9), dtype=np.float32)
    scaled_centers_img = np.zeros((0, 2), dtype=np.float32)
    if detections_fm.shape[0] > 0:
        scale_x = (down_ratio / input_w) * img_w
        scale_y = (down_ratio / input_h) * img_h
        scaled_boxes_img = detections_fm[:, :9].copy()
        scaled_boxes_img[:, 0:8:2] *= scale_x
        scaled_boxes_img[:, 1:8:2] *= scale_y
        scaled_centers_img = detections_fm[:, 9:11].copy()
        scaled_centers_img[:, 0] *= scale_x
        scaled_centers_img[:, 1] *= scale_y
    return scaled_boxes_img, scaled_centers_img

def scale_keypoints(detections_kpts_fm, input_w, input_h, img_w, img_h, down_ratio_kpts):
    scaled_kpts_img = np.zeros((0, 3), dtype=np.float32)
    if detections_kpts_fm.shape[0] > 0:
        scale_x = (down_ratio_kpts / input_w) * img_w
        scale_y = (down_ratio_kpts / input_h) * img_h
        scaled_kpts_img = detections_kpts_fm.copy()
        scaled_kpts_img[:, 0] *= scale_x
        scaled_kpts_img[:, 1] *= scale_y
    return scaled_kpts_img

def scale_edges(edges_fm, input_w, input_h, img_w, img_h, down_ratio, down_ratio_kpts):
    scaled_edges_img = np.zeros((0, 5), dtype=np.float32)
    if edges_fm.shape[0] > 0:
        scale_cx = (down_ratio / input_w) * img_w; scale_cy = (down_ratio / input_h) * img_h
        scale_kx = (down_ratio_kpts / input_w) * img_w; scale_ky = (down_ratio_kpts / input_h) * img_h
        scaled_edges_img = edges_fm.copy()
        scaled_edges_img[:, 0] *= scale_cx; scaled_edges_img[:, 1] *= scale_cy
        scaled_edges_img[:, 2] *= scale_kx; scaled_edges_img[:, 3] *= scale_ky
    return scaled_edges_img


# --- Modified TestModule ---
class TestModule(object):
    def __init__(self, K=100, conf_thresh=0.3):
        np.random.seed(317)
        self.K_decode = K
        self.conf_thresh = conf_thresh
        self.onnx_session = None
        self.input_name = None
        self.output_names_ordered = None # Will be set dynamically

    def test_single_image_onnx(self, args, down_ratio, down_ratio_kpts):
        onnx_model_path = args.onnx_path
        image_path = args.image_path
        output_dir = args.output_dir
        k_bbox_edges = args.k_bbox_edges
        k_kpts_edges = args.k_kpts_edges
        edge_conf_thresh = args.edge_conf_thresh
        nms_thresh = args.nms_thresh

        print(f"Loading ONNX model from: {onnx_model_path}")
        providers = ['CPUExecutionProvider']
        try:
            self.onnx_session = ort.InferenceSession(onnx_model_path, providers=providers)
            print(f"ONNX session created successfully using {self.onnx_session.get_providers()}.")
        except Exception as e:
            print(f"Error loading ONNX model: {e}")
            return

        self.input_name = self.onnx_session.get_inputs()[0].name
        session_output_names = [out.name for out in self.onnx_session.get_outputs()]
        # Request all outputs the model provides
        self.output_names_ordered = session_output_names
        print(f"Using Input Name: {self.input_name}")
        print(f"Model Output Names: {self.output_names_ordered}")

        # --- Dynamically Detect Output Components ---
        model_has_cls_theta = 'cls_theta' in session_output_names
        # Keypoints require both heatmap and regression outputs
        model_has_keypoints = ('hm_kpts' in session_output_names and 'reg_kpts' in session_output_names)
        # Edges require edge_probs AND the keypoint outputs
        model_has_edges = ('edge_probs' in session_output_names and model_has_keypoints)

        print(f"Detected Model Components:")
        print(f"  - cls_theta (for oriented boxes): {model_has_cls_theta}")
        print(f"  - Keypoints (hm_kpts, reg_kpts): {model_has_keypoints}")
        print(f"  - Edges (edge_probs + keypoints): {model_has_edges}")
        # Basic check for core outputs
        if not all(k in session_output_names for k in ['hm', 'wh', 'reg']):
             print(f"ERROR: Core outputs 'hm', 'wh', 'reg' not found in model outputs!")
             return

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")

        img_id = os.path.splitext(os.path.basename(image_path))[0]
        print(f'Processing image: {image_path}')

        ori_image_bgr = cv2.imread(image_path)
        if ori_image_bgr is None:
            print(f"Error: Could not load image {image_path}")
            return
        h_orig, w_orig, _ = ori_image_bgr.shape
        ori_image_for_viz = ori_image_bgr.copy()

        resized_image = cv2.resize(ori_image_bgr, (args.input_w, args.input_h), interpolation=cv2.INTER_LINEAR)
        image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        input_tensor = image_rgb.astype(np.float32) / 255.0 - 0.5
        input_tensor = input_tensor.transpose(2, 0, 1)
        input_np = np.expand_dims(input_tensor, axis=0)

        begin_time = time.time()
        try:
            # Request all available outputs
            onnx_outputs = self.onnx_session.run(self.output_names_ordered, {self.input_name: input_np})
        except Exception as e:
            print(f"\nError during ONNX inference: {e}")
            return
        end_time = time.time()
        inference_time = end_time - begin_time

        pr_decs_np = {name: onnx_outputs[i] for i, name in enumerate(self.output_names_ordered)}

        try:
            # 1. Decode Boxes and Keypoints (Pass detected booleans)
            decoded_boxes_fm, decoded_kpts_fm = numpy_decode_boxes_kpts(
                pr_decs_np, self.K_decode, self.conf_thresh,
                model_has_cls_theta, model_has_keypoints # Pass detected flags
            )

            # 2. Decode Edges (Only if detected)
            potential_edges_fm = np.zeros((0, 5), dtype=np.float32)
            if model_has_edges:
                potential_edges_fm = numpy_decode_onnx_edges(
                    pr_decs_np, k_bbox_edges, k_kpts_edges, edge_conf_thresh
                )

            # 3. Scale Boxes FOR NMS
            scaled_boxes_img_nms, scaled_centers_img_pre_nms = scale_poly_boxes(
                decoded_boxes_fm, args.input_w, args.input_h, w_orig, h_orig, down_ratio
            )

            # 4. Perform NMS
            nms_results_img = []
            nms_centers_keep_coords = []
            if scaled_boxes_img_nms.shape[0] > 0:
                nms_input_boxes_f64 = np.asarray(scaled_boxes_img_nms, dtype=np.float64)
                try:
                    keep_indices = py_cpu_nms_poly_fast(dets=nms_input_boxes_f64, thresh=nms_thresh)
                    if len(keep_indices) > 0:
                        nms_results_img = scaled_boxes_img_nms[keep_indices]
                        nms_centers_keep_coords = scaled_centers_img_pre_nms[keep_indices]
                except Exception as e:
                    print(f"\nError during NMS: {e}")

            # 5. Scale Keypoints (Only if detected)
            scaled_kpts_img = np.zeros((0, 3), dtype=np.float32)
            if model_has_keypoints:
                scaled_kpts_img = scale_keypoints(
                    decoded_kpts_fm, args.input_w, args.input_h, w_orig, h_orig, down_ratio_kpts
                )

            # 6. Scale Edges (Only if detected)
            scaled_edges_img = np.zeros((0, 5), dtype=np.float32)
            if model_has_edges:
                scaled_edges_img = scale_edges(
                    potential_edges_fm, args.input_w, args.input_h, w_orig, h_orig, down_ratio, down_ratio_kpts
                )

            # 7. Filter Scaled Edges (Only if detected and NMS results exist)
            filtered_edges_viz = []
            filtered_edge_scores_viz = []
            if model_has_edges and scaled_edges_img.shape[0] > 0 and len(nms_centers_keep_coords) > 0:
                distance_threshold_sq = ((down_ratio * 2) / args.input_w * w_orig)**2 + \
                                        ((down_ratio * 2) / args.input_h * h_orig)**2
                for edge_data in scaled_edges_img:
                    start_coord = (edge_data[0], edge_data[1])
                    edge_score = edge_data[4]
                    if edge_score < args.edge_vis_thresh: continue
                    is_close_to_kept_center = False
                    for nms_center in nms_centers_keep_coords:
                        dist_sq = (start_coord[0] - nms_center[0])**2 + (start_coord[1] - nms_center[1])**2
                        if dist_sq < distance_threshold_sq:
                            is_close_to_kept_center = True; break
                    if is_close_to_kept_center:
                        filtered_edges_viz.append(edge_data[:4])
                        filtered_edge_scores_viz.append(edge_score)

        except Exception as e:
            print(f"\nError during NumPy postprocessing: {e}")
            import traceback
            traceback.print_exc()
            return

        # --- Visualization ---
        category_name = 'object' # Hardcoded category

        # Draw boxes
        for pred in nms_results_img:
            score = pred[8]
            if score < args.vis_thresh: continue
            box_coords = pred[:8].reshape(4, 2); box_draw = np.int0(box_coords)
            ori_image_for_viz = cv2.drawContours(ori_image_for_viz, [box_draw], 0, (255,0,255), 2)
            cv2.putText(ori_image_for_viz, f'{score:.2f}', (box_draw[0, 0] + 5, box_draw[0, 1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)

        # Draw keypoints (Only if detected)
        if model_has_keypoints:
            kpt_indices_sorted = np.argsort(scaled_kpts_img[:, 2])[::-1]
            for idx in kpt_indices_sorted:
                kpt = scaled_kpts_img[idx, :2]; score = scaled_kpts_img[idx, 2]
                if score < args.kpt_thresh: continue
                x, y = int(kpt[0]), int(kpt[1])
                cv2.circle(ori_image_for_viz, (x, y), 3, (0, 255, 0), -1)
                cv2.putText(ori_image_for_viz, f'{score:.2f}', (x + 5, y + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)

        # Draw edges (Only if detected)
        if model_has_edges:
            for edge, score in zip(filtered_edges_viz, filtered_edge_scores_viz):
                pt1 = (int(edge[0]), int(edge[1])); pt2 = (int(edge[2]), int(edge[3]))
                cv2.line(ori_image_for_viz, pt1, pt2, (255, 0, 0), 1)
                mid_x = (pt1[0] + pt2[0]) // 2; mid_y = (pt1[1] + pt2[1]) // 2
                cv2.putText(ori_image_for_viz, f'{score:.2f}', (mid_x, mid_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

        output_filename = f"{img_id}_result.jpg"
        output_path = os.path.join(output_dir, output_filename)
        try:
            cv2.imwrite(output_path, ori_image_for_viz)
            print(f"\nSaved visualization to: {output_path}")
        except Exception as e:
            print(f"\nError saving image {output_path}: {e}")

        fps = 1.0 / inference_time if inference_time > 0 else 0
        print(f'Finished processing.')
        print(f'ONNX inference time: {inference_time:.4f} seconds ({fps:.2f} FPS)')


# --- Main Execution Block ---
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='ONNX Inference (Single Image, Dynamic Outputs) with NumPy Postprocessing')
    # Paths
    parser.add_argument('--onnx_path', type=str, required=True, help='Path to the ONNX model file.')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the single input image file.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save visualization results.')
    # Model Input/Output Params
    parser.add_argument('--input_h', type=int, default=1024, help='Model input height.')
    parser.add_argument('--input_w', type=int, default=1024, help='Model input width.')
    parser.add_argument('--down_ratio', type=int, default=8, help='Downsampling ratio for main heatmap/boxes.')
    parser.add_argument('--down_ratio_kpts', type=int, default=8, help='Downsampling ratio for keypoint heatmap (if different).')
    # Decoding Params
    parser.add_argument('--K', type=int, default=100, help='Max number of objects to detect (top-K).')
    parser.add_argument('--conf_thresh', type=float, default=0.5, help='Confidence threshold for detecting boxes/keypoints.')
    parser.add_argument('--nms_thresh', type=float, default=0.5, help='IoU threshold for Polygon NMS.') # Adjusted default NMS thresh
    # Edge Decoding Params (Needed if model *might* have edges)
    parser.add_argument('--k_bbox_edges', type=int, default=20, help='Top-K bbox candidates FOR edge prediction IN MODEL (required if edges present).')
    parser.add_argument('--k_kpts_edges', type=int, default=40, help='Top-K kpt candidates FOR edge prediction IN MODEL (required if edges present).')
    parser.add_argument('--edge_conf_thresh', type=float, default=0.5, help='Confidence threshold for potential edges (before NMS filtering).')
    # Visualization Params
    parser.add_argument('--vis_thresh', type=float, default=0.5, help='Score threshold for visualizing bounding boxes.')
    parser.add_argument('--kpt_thresh', type=float, default=0.5, help='Score threshold for visualizing keypoints.')
    parser.add_argument('--edge_vis_thresh', type=float, default=0.5, help='Score threshold for visualizing edges.')
    # --- Flags Removed ---

    args = parser.parse_args()

    if not os.path.exists(args.onnx_path): print(f"ERROR: ONNX model not found at {args.onnx_path}"); exit()
    if not os.path.exists(args.image_path): print(f"ERROR: Input image not found at {args.image_path}"); exit()

    test_module = TestModule(K=args.K, conf_thresh=args.conf_thresh)
    test_module.test_single_image_onnx(args=args,
                                       down_ratio=args.down_ratio,
                                       down_ratio_kpts=args.down_ratio_kpts)
    print("\nONNX single image testing script finished.")