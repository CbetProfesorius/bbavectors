import os
import torch
import numpy as np
from datasets.DOTA_devkit.ResultMerge_multi_process import py_cpu_nms_poly_fast, py_cpu_nms_poly


def decode_prediction(predictions, predictions_kpts, predictions_edges, dsets, args, img_id, down_ratio, down_ratio_kpts):
    predictions = predictions[0, :, :]
    predictions_kpts = predictions_kpts[0, :, :]
    ori_image = dsets.load_image(dsets.img_ids.index(img_id))
    h, w, c = ori_image.shape

    pts0 = {cat: [] for cat in dsets.category}
    centers0 = {cat: [] for cat in dsets.category}
    scores0 = {cat: [] for cat in dsets.category}
    kpts0 = []  # Store keypoints in a single list
    kpts_scores0 = []
    edges0 = []
    edges_scores0 = []

    for pred in predictions:
        cen_pt = np.asarray([pred[0], pred[1]], np.float32)
        tt = np.asarray([pred[2], pred[3]], np.float32)
        rr = np.asarray([pred[4], pred[5]], np.float32)
        bb = np.asarray([pred[6], pred[7]], np.float32)
        ll = np.asarray([pred[8], pred[9]], np.float32)
        tl = tt + ll - cen_pt
        bl = bb + ll - cen_pt
        tr = tt + rr - cen_pt
        br = bb + rr - cen_pt
        score = pred[10]
        clse = pred[11]
        pts = np.asarray([tr, br, bl, tl], np.float32)
        pts[:, 0] = pts[:, 0] * down_ratio / args.input_w * w
        pts[:, 1] = pts[:, 1] * down_ratio / args.input_h * h
        pts0[dsets.category[int(clse)]].append(pts)
        scores0[dsets.category[int(clse)]].append(score)
 
        cen_pt[0] = cen_pt[0] * down_ratio / args.input_w * w
        cen_pt[1] = cen_pt[1] * down_ratio / args.input_h * w
        centers0[dsets.category[int(clse)]].append(cen_pt)

    for pred_kpt in predictions_kpts:
        kpt_x = pred_kpt[0]
        kpt_y = pred_kpt[1]
        kpt_score = pred_kpt[2]
        kpt = np.asarray([kpt_x, kpt_y], np.float32)
        kpt[0] = kpt[0] * down_ratio_kpts / args.input_w * w
        kpt[1] = kpt[1] * down_ratio_kpts / args.input_h * h
        kpts0.append(kpt)
        kpts_scores0.append(kpt_score)

    for pred_edge in predictions_edges:
        scaled_edge = np.asarray(pred_edge, np.float32)
        scaled_edge[0] = scaled_edge[0] * down_ratio_kpts / args.input_w * w  # Scale x-coordinates
        scaled_edge[1] = scaled_edge[1] * down_ratio_kpts / args.input_h * h  # Scale y-coordinates
        scaled_edge[2] = scaled_edge[2] * down_ratio_kpts / args.input_w * w  # Scale x-coordinates
        scaled_edge[3] = scaled_edge[3] * down_ratio_kpts / args.input_h * h  # Scale y-coordinates
        edges0.append(scaled_edge)
        edges_scores0.append(scaled_edge[4])

    return pts0, scores0, kpts0, kpts_scores0, edges0, edges_scores0, centers0


def non_maximum_suppression(pts, scores, centers):
    nms_item = np.concatenate([pts[:, 0:1, 0],
                               pts[:, 0:1, 1],
                               pts[:, 1:2, 0],
                               pts[:, 1:2, 1],
                               pts[:, 2:3, 0],
                               pts[:, 2:3, 1],
                               pts[:, 3:4, 0],
                               pts[:, 3:4, 1],
                               scores[:, np.newaxis]], axis=1)
    nms_item = np.asarray(nms_item, np.float64)
    keep_index = py_cpu_nms_poly_fast(dets=nms_item, thresh=0.1)
    return nms_item[keep_index], centers[keep_index]


def write_results(args,
                  model,
                  dsets,
                  down_ratio,
                  down_ratio_kpts,
                  device,
                  decoder,
                  print_ps=False):
    results = {cat: {img_id: [] for img_id in dsets.img_ids} for cat in dsets.category}
    output_data = []

    for index in range(len(dsets)):
        data_dict = dsets.__getitem__(index, phase='test')
        image = data_dict['image'].to(device)
        gt = data_dict['gt']
        img_id = data_dict['img_id']
        image_w = data_dict['image_w']
        image_h = data_dict['image_h']

        with torch.no_grad():
            pr_decs = model(image)


        decoded_pts = []
        decoded_scores = []
        torch.cuda.synchronize(device)
        predictions, predictions_kpts = decoder.ctdet_decode(pr_decs)
        pts0, scores0, kpts0, kpts_scores0 = decode_prediction(predictions, predictions_kpts, dsets, args, img_id, down_ratio, down_ratio_kpts)
        decoded_pts.append(pts0)
        decoded_scores.append(scores0)

        pred_pts = []
        pred_cats = []
        pred_scores = []

        # nms
        for cat in dsets.category:
            pts_cat = []
            scores_cat = []
            for pts0, scores0 in zip(decoded_pts, decoded_scores):
                pts_cat.extend(pts0[cat])
                scores_cat.extend(scores0[cat])
            pts_cat = np.asarray(pts_cat, np.float32)
            scores_cat = np.asarray(scores_cat, np.float32)
            if pts_cat.shape[0]:
                nms_results = non_maximum_suppression(pts_cat, scores_cat)
                results[cat][img_id].extend(nms_results)

                for nms_res in nms_results:
                    pts = nms_res[:8]
                    confidence = nms_res[8]  # confidence score
                    
                    pred_pts.append(pts)
                    pred_cats.append(dsets.cat_ids[cat])
                    pred_scores.append(confidence)

        output_data.append({
            "image": image.cpu().numpy(),
            "gts": gt,
            "predictions": {
                "pts": np.asarray(pred_pts, np.float32),
                "cat": np.asarray(pred_cats, np.int32),
                "dif": np.asarray(pred_scores, np.float32),
                "kpts": np.asarray(kpts0, np.float32),
                "kpts_score": np.asarray(kpts_scores0, np.float32)
            }
        })

        if print_ps:
            print('testing {}/{} data {}'.format(index+1, len(dsets), img_id))

    return output_data

    # for cat in dsets.category:
    #     if cat == 'background':
    #         continue
    #     with open(os.path.join(result_path, 'Task1_{}.txt'.format(cat)), 'w') as f:
    #         for img_id in results[cat]:
    #             for pt in results[cat][img_id]:
    #                 f.write('{} {:.12f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.format(
    #                     img_id, pt[8], pt[0], pt[1], pt[2], pt[3], pt[4], pt[5], pt[6], pt[7]))
