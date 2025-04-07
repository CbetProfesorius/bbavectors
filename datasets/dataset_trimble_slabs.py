from .base import BaseDataset
import os
import cv2
import numpy as np
from .DOTA_devkit.ResultMerge_multi_process import mergebypoly
from .trimble_slabs_eval import voc_eval
from scipy.optimize import linear_sum_assignment


class TrimbleSlabs(BaseDataset):
    def __init__(self, data_dir, phase, input_h=None, input_w=None, down_ratio=None, down_ratio_kpts=None, kpts_radius=5):
        super(TrimbleSlabs, self).__init__(data_dir, phase, input_h, input_w, down_ratio, down_ratio_kpts, kpts_radius)
        self.category = ['slab']
        self.color_pans = [(204, 78, 210)]
        self.num_classes = len(self.category)
        self.cat_ids = {cat:i for i,cat in enumerate(self.category)}
        self.image_path = os.path.join(data_dir, phase, 'images')
        self.label_path = os.path.join(data_dir, phase, 'labels_qbox_pts')
        self.img_ids = self.load_img_ids()
        

    def load_img_ids(self):
        img_files = os.listdir(self.image_path)
        img_ids = [os.path.splitext(f)[0] for f in img_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))]
        return img_ids

    def load_image(self, index):
        img_id = self.img_ids[index]
        imgFile = os.path.join(self.image_path, img_id + '.png') # or .jpg, .jpeg
        if not os.path.exists(imgFile):
            imgFile = os.path.join(self.image_path, img_id + '.jpg')
        if not os.path.exists(imgFile):
            imgFile = os.path.join(self.image_path, img_id + '.jpeg')

        assert os.path.exists(imgFile), 'image {} not existed'.format(imgFile)
        img = cv2.imread(imgFile)
        return img

    def load_annoFolder(self, img_id):
        return os.path.join(self.label_path, img_id+'.txt')

    def load_annotation(self, index):
        image = self.load_image(index)
        h, w, c = image.shape
        valid_pts = []
        valid_cat = []
        valid_dif = []

        img_id = self.img_ids[index]
        anno_file = self.load_annoFolder(img_id)

        if not os.path.exists(anno_file):
            return {'pts': np.array([]), 'cat': np.array([]), 'dif': np.array([])}
    
        with open(anno_file, 'r') as f:
            lines = f.readlines()
        f.close()
        for line in lines[:-1]:
            obj = line.split(' ')  # list object
            if len(obj) == 10:
                x1 = min(max(float(obj[0]), 0), w - 1)
                y1 = min(max(float(obj[1]), 0), h - 1)
                x2 = min(max(float(obj[2]), 0), w - 1)
                y2 = min(max(float(obj[3]), 0), h - 1)
                x3 = min(max(float(obj[4]), 0), w - 1)
                y3 = min(max(float(obj[5]), 0), h - 1)
                x4 = min(max(float(obj[6]), 0), w - 1)
                y4 = min(max(float(obj[7]), 0), h - 1)

                # xmin = max(min(x1, x2, x3, x4), 0)
                # xmax = max(x1, x2, x3, x4)
                # ymin = max(min(y1, y2, y3, y4), 0)
                # ymax = max(y1, y2, y3, y4)

                valid_pts.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
                valid_cat.append(self.cat_ids[obj[8]])
                valid_dif.append(int(obj[9]))
        last_line = lines[-1].strip().split(' ')
        keypoints = [[float(last_line[i]), float(last_line[i + 1])] for i in range(0, len(last_line), 2)]
        annotation = {}
        annotation['pts'] = np.asarray(valid_pts, np.float32)
        annotation['cat'] = np.asarray(valid_cat, np.int32)
        annotation['dif'] = np.asarray(valid_dif, np.int32)
        annotation['corners'] = np.asanyarray(keypoints, np.float32)
        # pts0 = np.asarray(valid_pts, np.float32)
        # img = self.load_image(index)
        # for i in range(pts0.shape[0]):
        #     pt = pts0[i, :, :]
        #     tl = pt[0, :]
        #     tr = pt[1, :]
        #     br = pt[2, :]
        #     bl = pt[3, :]
        #     cv2.line(img, (int(tl[0]), int(tl[1])), (int(tr[0]), int(tr[1])), (0, 0, 255), 1, 1)
        #     cv2.line(img, (int(tr[0]), int(tr[1])), (int(br[0]), int(br[1])), (255, 0, 255), 1, 1)
        #     cv2.line(img, (int(br[0]), int(br[1])), (int(bl[0]), int(bl[1])), (0, 255, 255), 1, 1)
        #     cv2.line(img, (int(bl[0]), int(bl[1])), (int(tl[0]), int(tl[1])), (255, 0, 0), 1, 1)
        #     cv2.putText(img, '{}:{}'.format(valid_dif[i], self.category[valid_cat[i]]), (int(tl[0]), int(tl[1])), cv2.FONT_HERSHEY_TRIPLEX, 0.6,
        #                 (0, 0, 255), 1, 1)
        # cv2.imshow('img', np.uint8(img))
        # k = cv2.waitKey(0) & 0xFF
        # if k == ord('q'):
        #     cv2.destroyAllWindows()
        #     exit()
        return annotation


    def merge_crop_image_results(self, result_path, merge_path):
        mergebypoly(result_path, merge_path)


    def dec_evaluation(self, results, ovthresh):

        rec, prec, ap = voc_eval(results, ovthresh=0.5, use_07_metric=True)

        # total_precision = 0
        # total_recall = 0
        # total_f1 = 0
        # num_images = 0

        # for data in results:
        #     gt_corners = data["gts"]["corners"]
        #     pred_kpts = data["predictions"]["kpts"]

        #     num_gt = gt_corners.shape[0] 
        #     num_pred = pred_kpts.shape[0]

        #     if num_pred == 0:
        #         precision = 0
        #         recall = 0
        #         f1 = 0
        #     else:
        #         cost_matrix = np.zeros((num_gt, num_pred))
        #         for i in range(num_gt):
        #             for j in range(num_pred):
        #                 cost_matrix[i, j] = np.linalg.norm(gt_corners[i] - pred_kpts[j])

        #         gt_indices, pred_indices = linear_sum_assignment(cost_matrix)

        #         num_correct = sum(1 for i, j in zip(gt_indices, pred_indices) if cost_matrix[i, j] < ovthresh)

        #         precision = num_correct / num_pred if num_pred > 0 else 0
        #         recall = num_correct / num_gt if num_gt > 0 else 0
        #         f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        #     total_precision += precision
        #     total_recall += recall
        #     total_f1 += f1
        #     num_images += 1

        # kpts_prec = total_precision / num_images if num_images > 0 else 0
        # kpts_rec = total_recall / num_images if num_images > 0 else 0
        # kpts_f1 = total_f1 / num_images if num_images > 0 else 0


        return rec, prec, ap
    # , kpts_prec, kpts_rec, kpts_f1