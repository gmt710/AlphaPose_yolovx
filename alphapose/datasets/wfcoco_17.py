# -----------------------------------------------------
# Copyright (c) Shanghai Jiao Tong University. All rights reserved.
# Written by Jiefeng Li (jeff.lee.sjtu@gmail.com)
# -----------------------------------------------------

"""MS COCO Human keypoint dataset."""
from functools import reduce

import cv2
import numpy as np

from alphapose.models.builder import DATASET
from alphapose.utils.bbox import bbox_clip_xyxy, bbox_xywh_to_xyxy

from .mscoco import Mscoco


@DATASET.register_module
class Wfcoco17(Mscoco):
    def _check_load_keypoints(self, coco, entry, img_path):
        """Check and load ground-truth keypoints"""
        ann_ids = coco.getAnnIds(imgIds=entry['id'], iscrowd=False)
        objs = coco.loadAnns(ann_ids)

        # check valid bboxes
        valid_objs = []
        width = entry['width']
        height = entry['height']
        if width == 0 or height == 0:
            image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            height, width = image.shape[0], image.shape[1]

        for obj in objs:
            contiguous_cid = self.json_id_to_contiguous[obj['category_id']]
            if contiguous_cid >= self.num_class:
                # not class of interest
                print("not class of interest: %d" % contiguous_cid)
                continue
            if max(obj['keypoints']) == 0:
                print("no visible keypoints")
                continue
            # Force wf coco data into the 17 keypoint label set expected by Alphapose
            if len(obj['keypoints']) == 18 * 3:
                obj['keypoints'] = obj['keypoints'][:17*3]
                obj['num_keypoints'] = reduce(lambda count, v: count + (v > 0), obj['keypoints'][2::3], 0)

            # convert from (x, y, w, h) to (xmin, ymin, xmax, ymax) and clip bound
            xmin, ymin, xmax, ymax = bbox_clip_xyxy(bbox_xywh_to_xyxy(obj['bbox']), width, height)
            # require non-zero box area
            if obj['area'] <= 0 or xmax <= xmin or ymax <= ymin:
                print("bbox area: Area - %d; Xmax - %d, Xmin - %d, Ymax - %d, Ymin - %d" % (obj['area'], xmax, xmin, ymax, ymin))
                continue
            if obj['num_keypoints'] == 0:
                print("no keypoints")
                continue
            # joints 3d: (num_joints, 3, 2); 3 is for x, y, z; 2 is for position, visibility
            joints_3d = np.zeros((self.num_joints, 3, 2), dtype=np.float32)
            for i in range(self.num_joints):
                joints_3d[i, 0, 0] = obj['keypoints'][i * 3 + 0]
                joints_3d[i, 1, 0] = obj['keypoints'][i * 3 + 1]
                # joints_3d[i, 2, 0] = 0
                visible = min(1, obj['keypoints'][i * 3 + 2])
                joints_3d[i, :2, 1] = visible
                # joints_3d[i, 2, 1] = 0

            if np.sum(joints_3d[:, 0, 1]) < 1:
                # no visible keypoint
                print("no visible keypoints - summed validation")
                continue

            if self._check_centers and self._train:
                bbox_center, bbox_area = self._get_box_center_area((xmin, ymin, xmax, ymax))
                kp_center, num_vis = self._get_keypoints_center_count(joints_3d)
                ks = np.exp(-2 * np.sum(np.square(bbox_center - kp_center)) / bbox_area)
                if (num_vis / 80.0 + 47 / 80.0) > ks:
                    print("check center validation failed")
                    continue

            valid_objs.append({
                'file_name': entry['file_name'],
                'bbox': (xmin, ymin, xmax, ymax),
                'width': width,
                'height': height,
                'joints_3d': joints_3d
            })

        if not valid_objs:
            if not self._skip_empty:
                # dummy invalid labels if no valid objects are found
                valid_objs.append({
                    'file_name': entry['file_name'],
                    'bbox': np.array([-1, -1, 0, 0]),
                    'width': width,
                    'height': height,
                    'joints_3d': np.zeros((self.num_joints, 2, 2), dtype=np.float32)
                })
        return valid_objs
