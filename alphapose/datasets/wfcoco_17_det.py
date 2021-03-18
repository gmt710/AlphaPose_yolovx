# -----------------------------------------------------
# Copyright (c) Shanghai Jiao Tong University. All rights reserved.
# Written by Jiefeng Li (jeff.lee.sjtu@gmail.com)
# -----------------------------------------------------

"""MS COCO Human Detection Box dataset."""
import os

import cv2
import torch

from alphapose.models.builder import DATASET

from .coco_det import Mscoco_det


@DATASET.register_module
class Wfcoco17_det(Mscoco_det):
    def __getitem__(self, index):
        det_res = self._det_json[index]
        img_id = det_res['image_id']
        img_path = os.path.join(self._root, self._img_prefix, det_res['file_name'])

        # Load image
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        imght, imgwidth = image.shape[1], image.shape[2]
        x1, y1, w, h = det_res['bbox']
        bbox = [x1, y1, x1 + w, y1 + h]
        inp, bbox = self.transformation.test_transform(image, bbox)
        return inp, torch.Tensor(bbox), torch.Tensor([det_res['bbox']]), torch.Tensor([img_id]), torch.Tensor([det_res['score']]), torch.Tensor([imght]), torch.Tensor([imgwidth])
