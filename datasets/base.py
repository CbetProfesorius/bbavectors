import torch.utils.data as data
import cv2
import torch
import numpy as np
import math
from .draw_gaussian import draw_umich_gaussian, gaussian_radius
from .transforms import random_flip, load_affine_matrix, random_crop_info, ex_box_jaccard
from . import data_augment

class BaseDataset(data.Dataset):
    def __init__(self, data_dir, phase, input_h=None, input_w=None, down_ratio=None, down_ratio_kpts=None, kpts_radius=5):
        super(BaseDataset, self).__init__()
        self.data_dir = data_dir
        self.phase = phase
        self.input_h = input_h
        self.input_w = input_w
        self.down_ratio = down_ratio
        self.down_ratio_kpts = down_ratio_kpts
        self.img_ids = None
        self.num_classes = None
        self.max_objs = 500
        self.kpts_radius = kpts_radius
        # self.image_distort =  data_augment.PhotometricDistort()

    def load_img_ids(self):
        """
        Definition: generate self.img_ids
        Usage: index the image properties (e.g. image name) for training, testing and evaluation
        Format: self.img_ids = [list]
        Return: self.img_ids
        """
        return None

    def load_image(self, index):
        """
        Definition: read images online
        Input: index, the index of the image in self.img_ids
        Return: image with H x W x 3 format
        """
        return None

    def load_annoFolder(self, img_id):
        """
        Return: the path of annotation
        Note: You may not need this function
        """
        return None

    def load_annotation(self, index):
        """
        Return: dictionary of {'pts': float np array of [bl, tl, tr, br], 
                                'cat': int np array of class_index}
        Explaination:
                bl: bottom left point of the bounding box, format [x, y]
                tl: top left point of the bounding box, format [x, y]
                tr: top right point of the bounding box, format [x, y]
                br: bottom right point of the bounding box, format [x, y]
                class_index: the category index in self.category
                    example: self.category = ['ship]
                             class_index of ship = 0
        """
        return None

    def dec_evaluation(self, result_path):
        return None

    def data_transform(self, image, annotation):
        crop_center = np.asarray([np.round(image.shape[1] / 2), np.round(image.shape[0] / 2)], dtype=np.float32)
        crop_size = [max(image.shape[1], image.shape[0]), max(image.shape[1], image.shape[0])]

        image, annotation['pts'], annotation['corners'], crop_center = random_flip(image, annotation['pts'], annotation['corners'], crop_center)

        # self.visualize_annotations(image, annotation, "0")

        M = load_affine_matrix(crop_center=crop_center,
                               crop_size=crop_size,
                               dst_size=(self.input_w, self.input_h),
                               inverse=False,
                               rotation=True)
        
        image = cv2.warpAffine(src=image, M=M, dsize=(self.input_w, self.input_h), flags=cv2.INTER_LINEAR)

        if annotation['pts'].shape[0]:
            annotation['pts'] = np.concatenate([annotation['pts'], np.ones((annotation['pts'].shape[0], annotation['pts'].shape[1], 1))], axis=2)
            annotation['pts'] = np.matmul(annotation['pts'], np.transpose(M))
            annotation['pts'] = np.asarray(annotation['pts'], np.float32)

        if annotation['corners'].shape[0]:
            annotation['corners'] = np.concatenate([annotation['corners'], np.ones((annotation['corners'].shape[0], 1))], axis=1)
            annotation['corners'] = np.matmul(annotation['corners'], np.transpose(M))
            annotation['corners'] = np.asarray(annotation['corners'], np.float32)
            
        # self.visualize_annotations(image, annotation, "1")

        # object_indices = np.arange(len(annotation['pts']))
        # np.random.shuffle(object_indices)

        out_annotations = {}
        out_rects = []
        out_cat = []
        valid_corners = []
        points_per_object = []

        # for i in object_indices:
        #     pt_old = annotation['pts'][i]
        #     cat = annotation['cat'][i]
        for pt_old, cat in zip(annotation['pts'] , annotation['cat']):
            if (pt_old < 0).any() or (pt_old[:,0] > self.input_w-1).any() or (pt_old[:,1] > self.input_h-1).any():
                continue
            
            rect = cv2.minAreaRect(pt_old/self.down_ratio)
            # rect = cv2.minAreaRect(pt_old)
            
            # rect_center = rect[0]

            # if rect_center not in bbox_centers:
            #     bbox_centers[rect_center] = []

            # bbox_centers[rect_center].append(pt_old)

            points_per_object.append(pt_old/self.down_ratio)
            
            for pt in pt_old:
                if not any((pt/self.down_ratio == vc).all() for vc in valid_corners):
                    valid_corners.append(pt/self.down_ratio)

            out_rects.append([rect[0][0], rect[0][1], rect[1][0], rect[1][1], rect[2]])
            out_cat.append(cat)

        # valid_corners_old = []
        # for corner in annotation['corners']:
        #     if (corner < 0).any() or (corner[0] > self.input_w - 1) or (corner[1] > self.input_h - 1):
        #         continue
        #     valid_corners_old.append(corner/self.down_ratio_kpts)
        #     # valid_corners.append(corner)

        out_annotations['rect'] = np.asarray(out_rects, np.float32)
        out_annotations['cat'] = np.asarray(out_cat, np.uint8)
        out_annotations['corners'] = np.asarray(valid_corners, np.float32)
        out_annotations['points_per_object'] = np.asarray(points_per_object, np.float32)
        # new_w = int(self.input_w / self.down_ratio)
        # new_h = int(self.input_h / self.down_ratio)
        # image = cv2.resize(image, (new_w, new_h))

        # self.visualize_rect_annotations(image, out_annotations, "train")

        return image, out_annotations


    def val_data_transform(self, image, annotation):
        out_annotations = {}
        out_rects = []
        out_cat = []
        valid_corners = []
        points_per_object = []
        orig_h, orig_w = image.shape[:2]

        annotation['pts'] = np.asarray(annotation['pts'], np.float32)
        annotation['corners'] = np.asanyarray(annotation['corners'], np.float32)

        scale_x = self.input_w / orig_w
        scale_y = self.input_h / orig_h

        image = cv2.resize(image, (self.input_w, self.input_h))

        for pt_old, cat in zip(annotation['pts'] , annotation['cat']):

            pt_scaled = pt_old.copy()
            pt_scaled[:, 0] *= scale_x  
            pt_scaled[:, 1] *= scale_y

            rect = cv2.minAreaRect(pt_scaled / self.down_ratio)

            points_per_object.append(pt_scaled/self.down_ratio)

            for pt in pt_scaled:
                if not any((pt/self.down_ratio == vc).all() for vc in valid_corners):
                    valid_corners.append(pt/self.down_ratio)

            out_rects.append([rect[0][0], rect[0][1], rect[1][0], rect[1][1], rect[2]])
            out_cat.append(cat)

        # annotation['corners'][:, 0] *= scale_x
        # annotation['corners'][:, 1] *= scale_y

        out_annotations['rect'] = np.asarray(out_rects, np.float32)
        out_annotations['cat'] = np.asarray(out_cat, np.uint8)
        # out_annotations['corners'] = np.asarray(annotation['corners'] / self.down_ratio_kpts, np.float32)
        out_annotations['corners'] = np.asarray(valid_corners, np.float32)
        out_annotations['points_per_object'] = np.asarray(points_per_object, np.float32)
        
        # self.visualize_rect_annotations(image, out_annotations, "val")

        # new_w = int(self.input_w / self.down_ratio)
        # new_h = int(self.input_h / self.down_ratio)
        # image = cv2.resize(image, (new_w, new_h))

        return image, out_annotations


    def __len__(self):
        return len(self.img_ids)

    def processing_test(self, image, input_h, input_w):
        image = cv2.resize(image, (input_w, input_h))
        out_image = image.astype(np.float32) / 255.
        out_image = out_image - 0.5
        out_image = out_image.transpose(2, 0, 1).reshape(1, 3, input_h, input_w)
        out_image = torch.from_numpy(out_image)
        return out_image

    def cal_bbox_wh(self, pts_4):
        x1 = np.min(pts_4[:,0])
        x2 = np.max(pts_4[:,0])
        y1 = np.min(pts_4[:,1])
        y2 = np.max(pts_4[:,1])
        return x2-x1, y2-y1


    def cal_bbox_pts(self, pts_4):
        x1 = np.min(pts_4[:,0])
        x2 = np.max(pts_4[:,0])
        y1 = np.min(pts_4[:,1])
        y2 = np.max(pts_4[:,1])
        bl = [x1, y2]
        tl = [x1, y1]
        tr = [x2, y1]
        br = [x2, y2]
        return np.asarray([bl, tl, tr, br], np.float32)

    def reorder_pts(self, tt, rr, bb, ll):
        pts = np.asarray([tt,rr,bb,ll],np.float32)
        l_ind = np.argmin(pts[:,0])
        r_ind = np.argmax(pts[:,0])
        t_ind = np.argmin(pts[:,1])
        b_ind = np.argmax(pts[:,1])
        tt_new = pts[t_ind,:]
        rr_new = pts[r_ind,:]
        bb_new = pts[b_ind,:]
        ll_new = pts[l_ind,:]
        return tt_new,rr_new,bb_new,ll_new


    def generate_ground_truth(self, image, annotation):
        image = np.asarray(np.clip(image, a_min=0., a_max=255.), np.float32)
        # im = image.copy()
        image = np.transpose(image / 255. - 0.5, (2, 0, 1))

        image_h = self.input_h // self.down_ratio
        image_w = self.input_w // self.down_ratio

        hm = np.zeros((self.num_classes, image_h, image_w), dtype=np.float32)
        wh = np.zeros((self.max_objs, 10), dtype=np.float32)
        ## add
        cls_theta = np.zeros((self.max_objs, 1), dtype=np.float32)
        ## add end
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
        num_objs = min(annotation['rect'].shape[0], self.max_objs)

        image_h_kpts = self.input_h // self.down_ratio_kpts
        image_w_kpts = self.input_w // self.down_ratio_kpts

        hm_kpts = np.zeros((self.num_classes, image_h_kpts, image_w_kpts), dtype=np.float32)
        reg_kpts = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind_kpts = np.zeros((self.max_objs), dtype=np.int64)
        reg_mask_kpts = np.zeros((self.max_objs), dtype=np.uint8)
        num_objs_kpts = min(annotation['corners'].shape[0], self.max_objs)

        # ###################################### view Images #######################################
        # copy_image1 = cv2.resize(im, (image_w, image_h))
        # copy_image2 = copy_image1.copy()
        # self.visualize_rect_annotations(copy_image1, annotation, "2")
        # ##########################################################################################
        rectangle_pts = {} # store grid idx of center point and its corner point coordinates
        for k in range(num_objs):
            rect = annotation['rect'][k, :]
            cen_x, cen_y, bbox_w, bbox_h, theta = rect
            radius = gaussian_radius((math.ceil(bbox_h), math.ceil(bbox_w)))
            radius = max(0, int(radius))
            ct = np.asarray([cen_x, cen_y], dtype=np.float32)
            ct_int = ct.astype(np.int32)
            draw_umich_gaussian(hm[annotation['cat'][k]], ct_int, radius)

            center_idx = (ct_int[1], ct_int[0])
            rectangle_pts[center_idx] = annotation['points_per_object'][k]

            ind[k] = ct_int[1] * image_w + ct_int[0]
            reg[k] = ct - ct_int
            reg_mask[k] = 1
            # generate wh ground_truth
            pts_4 = cv2.boxPoints(((cen_x, cen_y), (bbox_w, bbox_h), theta))  # 4 x 2

            bl = pts_4[0,:]
            tl = pts_4[1,:]
            tr = pts_4[2,:]
            br = pts_4[3,:]

            tt = (np.asarray(tl,np.float32)+np.asarray(tr,np.float32))/2
            rr = (np.asarray(tr,np.float32)+np.asarray(br,np.float32))/2
            bb = (np.asarray(bl,np.float32)+np.asarray(br,np.float32))/2
            ll = (np.asarray(tl,np.float32)+np.asarray(bl,np.float32))/2

            if theta in [-90.0, -0.0, 0.0]:  # (-90, 0]
                tt,rr,bb,ll = self.reorder_pts(tt,rr,bb,ll)
            # rotational channel
            wh[k, 0:2] = tt - ct
            wh[k, 2:4] = rr - ct
            wh[k, 4:6] = bb - ct
            wh[k, 6:8] = ll - ct
            #####################################################################################
            # # draw
            # cv2.line(copy_image1, (int(cen_x), int(cen_y)), (int(tt[0]), int(tt[1])), (0, 0, 255), 1, 1)
            # cv2.line(copy_image1, (int(cen_x), int(cen_y)), (int(rr[0]), int(rr[1])), (255, 0, 255), 1, 1)
            # cv2.line(copy_image1, (int(cen_x), int(cen_y)), (int(bb[0]), int(bb[1])), (0, 255, 255), 1, 1)
            # cv2.line(copy_image1, (int(cen_x), int(cen_y)), (int(ll[0]), int(ll[1])), (255, 0, 0), 1, 1)
            #####################################################################################
            # horizontal channel
            w_hbbox, h_hbbox = self.cal_bbox_wh(pts_4)
            wh[k, 8:10] = 1. * w_hbbox, 1. * h_hbbox
            #####################################################################################
            # # draw
            # cv2.line(copy_image2, (int(cen_x), int(cen_y)), (int(cen_x), int(cen_y-wh[k, 9]/2)), (0, 0, 255), 1, 1)
            # cv2.line(copy_image2, (int(cen_x), int(cen_y)), (int(cen_x+wh[k, 8]/2), int(cen_y)), (255, 0, 255), 1, 1)
            # cv2.line(copy_image2, (int(cen_x), int(cen_y)), (int(cen_x), int(cen_y+wh[k, 9]/2)), (0, 255, 255), 1, 1)
            # cv2.line(copy_image2, (int(cen_x), int(cen_y)), (int(cen_x-wh[k, 8]/2), int(cen_y)), (255, 0, 0), 1, 1)
            #####################################################################################
            # v0
            # if abs(theta)>3 and abs(theta)<90-3:
            #     cls_theta[k, 0] = 1
            # v1
            jaccard_score = ex_box_jaccard(pts_4.copy(), self.cal_bbox_pts(pts_4).copy())
            if jaccard_score<0.95:
                cls_theta[k, 0] = 1
        # ###################################### view Images #####################################
        # hm_show = np.uint8(cv2.applyColorMap(np.uint8(hm[0, :, :] * 255), cv2.COLORMAP_JET))
        # copy_image = cv2.addWeighted(np.uint8(copy_image), 0.4, hm_show, 0.8, 0)
            # if jaccard_score>0.95:
            # print(theta, jaccard_score, cls_theta[k, 0])
            # import os
            # cv2.imshow('img1', cv2.resize(np.uint8(copy_image1), (image_w*4, image_h*4)))
            # cv2.imshow('img2', cv2.resize(np.uint8(copy_image2), (image_w*4, image_h*4)))
            # cv2.imwrite("/mnt/c/src/BBAVectors-Oriented-Object-Detection/temp_images/pirma.jpg", cv2.resize(np.uint8(copy_image1), (image_w*4, image_h*4)))
            # cv2.imwrite("/mnt/c/src/BBAVectors-Oriented-Object-Detection/temp_images/antra.jpg", cv2.resize(np.uint8(copy_image2), (image_w*4, image_h*4)))
            # key = cv2.waitKey(0)&0xFF
            # if key==ord('q'):
            #     cv2.destroyAllWindows()
            #     exit()
        # #########################################################################################

        # copy_image1 = cv2.resize(im, (image_w, image_h))
        # for c in range(self.num_classes):
        #     heatmap = (hm[c] * 255).astype(np.uint8)
        #     heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        #     im_overlay = cv2.addWeighted(np.uint8(copy_image1), 0.4, heatmap, 0.8, 0)

        #     for k in range(num_objs):
        #         if annotation['cat'][k] == c:
        #             rect = annotation['rect'][k, :]
        #             cen_x, cen_y, bbox_w, bbox_h, theta = rect
        #             pts_4 = cv2.boxPoints(((cen_x, cen_y), (bbox_w, bbox_h), theta))
        #             pts_4 = np.int32(pts_4)
        #             cv2.polylines(im_overlay, [pts_4], True, (0, 255, 0), 2)  # Draw bounding boxes

        #     cv2.imwrite(f"/mnt/c/src/BBAVectors-Oriented-Object-Detection/temp_images/heatmap_bboxes_class_{c}.png", im_overlay)

        center_corners_indices = {}
        for k in range(num_objs_kpts):
            kpts = annotation['corners'][k, :]
            x, y = kpts
            # radius = gaussian_radius((kpts_radius, kpts_radius))
            # radius = max(0, int(radius))
            ct = np.asarray([x, y], dtype=np.float32)
            ct_int = ct.astype(np.int32)
            draw_umich_gaussian(hm_kpts[0], ct_int, self.kpts_radius)

            corner_idx = (ct_int[1], ct_int[0])

            def array_exists_in_list(arr, arr_list):
                for existing_arr in arr_list:
                    if np.array_equal(arr, existing_arr):
                        return True
                return False
            
            for cnt, crns in rectangle_pts.items():
                if not cnt in center_corners_indices:
                    center_corners_indices[cnt] = []

                # if kpts in crns:
                if array_exists_in_list(kpts, crns):
                    center_corners_indices[cnt].append(corner_idx)

            ind_kpts[k] = ct_int[1] * image_w + ct_int[0]
            reg_kpts[k] = ct - ct_int
            reg_mask_kpts[k] = 1
        
        keys = list(center_corners_indices.keys())
        values = list(center_corners_indices.values())

        num_corners = max(len(c) for c in values)

        full_arr = np.full((self.max_objs, num_corners + 1, 2), -1, dtype=np.int32)

        for i, (center, corners) in enumerate(zip(keys, values)):
            full_arr[i, 0, :] = center
            full_arr[i, 1:len(corners)+1, :] = corners

        ret = {'input': image,
               'hm': hm,
               'reg_mask': reg_mask,
               'ind': ind,
               'wh': wh,
               'reg': reg,
               'cls_theta': cls_theta,
               'hm_kpts': hm_kpts,
               'reg_mask_kpts': reg_mask_kpts,
               'ind_kpts': ind_kpts,
               'reg_kpts': reg_kpts,
               'center_corners_indices': full_arr
               }
        return ret


    def val_generate_ground_truth(self, image, annotation):
        image = np.asarray(np.clip(image, a_min=0., a_max=255.), np.float32)
        image = np.transpose(image / 255. - 0.5, (2, 0, 1))

        image_h = self.input_h // self.down_ratio
        image_w = self.input_w // self.down_ratio

        hm = np.zeros((self.num_classes, image_h, image_w), dtype=np.float32)
        wh = np.zeros((self.max_objs, 10), dtype=np.float32)
        cls_theta = np.zeros((self.max_objs, 1), dtype=np.float32)
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
        num_objs = min(annotation['rect'].shape[0], self.max_objs)

        image_h_kpts = self.input_h // self.down_ratio_kpts
        image_w_kpts = self.input_w // self.down_ratio_kpts

        hm_kpts = np.zeros((self.num_classes, image_h_kpts, image_w_kpts), dtype=np.float32)
        reg_kpts = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind_kpts = np.zeros((self.max_objs), dtype=np.int64)
        reg_mask_kpts = np.zeros((self.max_objs), dtype=np.uint8)
        num_objs_kpts = min(annotation['corners'].shape[0], self.max_objs)

        rectangle_pts = {}
        for k in range(num_objs):
            rect = annotation['rect'][k, :]
            cen_x, cen_y, bbox_w, bbox_h, theta = rect
            radius = gaussian_radius((math.ceil(bbox_h), math.ceil(bbox_w)))
            radius = max(0, int(radius))
            ct = np.asarray([cen_x, cen_y], dtype=np.float32)
            ct_int = ct.astype(np.int32)
            draw_umich_gaussian(hm[annotation['cat'][k]], ct_int, radius)
            
            center_idx = (ct_int[1], ct_int[0])
            rectangle_pts[center_idx] = annotation['points_per_object'][k]
            
            ind[k] = ct_int[1] * image_w + ct_int[0]
            reg[k] = ct - ct_int
            reg_mask[k] = 1
            pts_4 = cv2.boxPoints(((cen_x, cen_y), (bbox_w, bbox_h), theta))  # 4 x 2

            bl = pts_4[0,:]
            tl = pts_4[1,:]
            tr = pts_4[2,:]
            br = pts_4[3,:]

            tt = (np.asarray(tl,np.float32)+np.asarray(tr,np.float32))/2
            rr = (np.asarray(tr,np.float32)+np.asarray(br,np.float32))/2
            bb = (np.asarray(bl,np.float32)+np.asarray(br,np.float32))/2
            ll = (np.asarray(tl,np.float32)+np.asarray(bl,np.float32))/2

            if theta in [-90.0, -0.0, 0.0]:  # (-90, 0]
                tt,rr,bb,ll = self.reorder_pts(tt,rr,bb,ll)

            wh[k, 0:2] = tt - ct
            wh[k, 2:4] = rr - ct
            wh[k, 4:6] = bb - ct
            wh[k, 6:8] = ll - ct

            w_hbbox, h_hbbox = self.cal_bbox_wh(pts_4)
            wh[k, 8:10] = 1. * w_hbbox, 1. * h_hbbox

            jaccard_score = ex_box_jaccard(pts_4.copy(), self.cal_bbox_pts(pts_4).copy())
            if jaccard_score<0.95:
                cls_theta[k, 0] = 1

        center_corners_indices = {}
        for k in range(num_objs_kpts):
            kpts = annotation['corners'][k, :]
            x, y = kpts
            # radius = gaussian_radius((kpts_radius, kpts_radius))
            # radius = max(0, int(radius))
            ct = np.asarray([x, y], dtype=np.float32)
            ct_int = ct.astype(np.int32)
            draw_umich_gaussian(hm_kpts[0], ct_int, self.kpts_radius)

            corner_idx = (ct_int[1], ct_int[0])

            def array_exists_in_list(arr, arr_list):
                for existing_arr in arr_list:
                    if np.array_equal(arr, existing_arr):
                        return True
                return False

            for cnt, crns in rectangle_pts.items():
                if not cnt in center_corners_indices:
                    center_corners_indices[cnt] = []

                # if kpts in crns:
                if array_exists_in_list(kpts, crns):
                    center_corners_indices[cnt].append(corner_idx)

            ind_kpts[k] = ct_int[1] * image_w + ct_int[0]
            reg_kpts[k] = ct - ct_int
            reg_mask_kpts[k] = 1

        keys = list(center_corners_indices.keys())
        values = list(center_corners_indices.values())

        num_corners = max(len(c) for c in values)

        full_arr = np.full((self.max_objs, num_corners + 1, 2), -1, dtype=np.int32)

        for i, (center, corners) in enumerate(zip(keys, values)):
            full_arr[i, 0, :] = center
            full_arr[i, 1:len(corners)+1, :] = corners

        ret = {'input': image,
               'hm': hm,
               'reg_mask': reg_mask,
               'ind': ind,
               'wh': wh,
               'reg': reg,
               'cls_theta': cls_theta,
               'hm_kpts': hm_kpts,
               'reg_mask_kpts': reg_mask_kpts,
               'ind_kpts': ind_kpts,
               'reg_kpts': reg_kpts,
               'center_corners_indices': full_arr
               }
        return ret


    def plot_annotations(image, annotation, category_mapping):
        """
        Plots multiple annotations on the given image.

        Args:
            image (numpy.ndarray): The image to draw on.
            annotation (dict): Dictionary containing 'pts' (polygon points), 'cat' (categories), and 'dif' (difficulty levels).
            category_mapping (dict): Mapping from category ID to category name.
        """
        pts = annotation['pts']  # Shape: (num_objects, 4, 2)
        cats = annotation['cat']  # Shape: (num_objects,)
        difs = annotation['dif']  # Shape: (num_objects,)

        for i in range(len(pts)):
            polygon = pts[i].astype(int)  # Convert to integers for OpenCV

            # Draw the quadrilateral
            cv2.polylines(image, [polygon], isClosed=True, color=(0, 255, 0), thickness=2)

            # Get category name
            category_name = category_mapping.get(cats[i], str(cats[i]))

            # Draw the category and difficulty label
            label = f"{category_name} (Dif: {difs[i]})"
            cv2.putText(image, label, (polygon[0][0], polygon[0][1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Show image with annotations
        cv2.imshow("Annotated Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def __getitem__(self, index, phase=None):
        image = self.load_image(index)
        image_h, image_w, c = image.shape

        if phase is not None:
            # self.phase = phase
            img_id = self.img_ids[index]
            annotation = self.load_annotation(index)
            image = self.processing_test(image, self.input_h, self.input_w)
            return {'image': image,
                    'gt': annotation,
                    'img_id': img_id,
                    'image_w': image_w,
                    'image_h': image_h}
        
        if self.phase == 'val':
            annotation = self.load_annotation(index)
            image, annotation = self.val_data_transform(image, annotation)
            data_dict = self.val_generate_ground_truth(image, annotation)
            return data_dict
        
        if self.phase == 'test':
            img_id = self.img_ids[index]
            image = self.processing_test(image, self.input_h, self.input_w)
            return {'image': image,
                    'img_id': img_id,
                    'image_w': image_w,
                    'image_h': image_h}

        elif self.phase == 'train':
            annotation = self.load_annotation(index)
            # TODO fully black part of image, unvisible keypoint there
            image, annotation = self.data_transform(image, annotation)
            data_dict = self.generate_ground_truth(image, annotation)
            return data_dict


    def visualize_annotations(self, image, annotation, index):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        pts = annotation['pts']
        cat = annotation['cat']
        corners = annotation['corners']

        for i in range(len(pts)):
            polygon = np.array(pts[i], np.int32).reshape((-1, 1, 2))

            color = (0, 255, 0)

            cv2.polylines(image, [polygon], isClosed=True, color=color, thickness=2)

            x, y = int(pts[i][0][0]), int(pts[i][0][1])  # Use first point for text location
            # cv2.putText(image, str(cat[i]), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        for corner in corners:
            x, y = int(corner[0]), int(corner[1])
            cv2.circle(image, (x, y), 3, (0, 0, 255), -1)

        cv2.imwrite(f"/mnt/c/src/BBAVectors-Oriented-Object-Detection/temp_images/{index}.jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    def visualize_rect_annotations(self, image, annotation, index):
        rects = annotation['rect']
        cat = annotation['cat']
        corners = annotation['corners']
        
        for i in range(len(rects)):
            center_x, center_y, width, height, angle = rects[i]
            box = cv2.boxPoints(((center_x, center_y), (width, height), angle))
            box = np.int0(box)

            color = (0, 255, 0)  

            cv2.drawContours(image, [box], 0, color, 2)

            x, y = int(center_x), int(center_y)  # Use center for text location
            # cv2.putText(image, str(cat[i]), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        for corner in corners:
            x, y = int(corner[0]), int(corner[1])
            cv2.circle(image, (x, y), 3, (0, 0, 255), -1)

        cv2.imwrite(f"/mnt/c/src/BBAVectors-Oriented-Object-Detection/temp_images/{index}.jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))