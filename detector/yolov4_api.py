# -----------------------------------------------------
# Copyright (c) Shanghai Jiao Tong University. All rights reserved.
# Written by Chao Xu (xuchao.19962007@sjtu.edu.cn)
# -----------------------------------------------------

"""API of yolo detector"""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
import platform

import torch
import numpy as np

from yolo_v4.detect import Detector
from yolo_v4.preprocess import prep_image, prep_frame

from detector.apis import BaseDetector

#only windows visual studio 2013 ~2017 support compile c/cuda extensions
#If you force to compile extension on Windows and ensure appropriate visual studio
#is intalled, you can try to use these ext_modules.
if platform.system() != 'Windows':
    from detector.nms import nms_wrapper


class YOLOV4Detector(BaseDetector):
    def __init__(self, cfg, opt=None):
        super(YOLOV4Detector, self).__init__()

        self.detector_cfg = cfg
        self.detector_opt = opt
        self.model_cfg = cfg.get('CONFIG', 'detector/yolo_v4/cfg/yolov4.cfg')
        self.model_weights = cfg.get('WEIGHTS', 'detector/yolo_v4/data/yolov4.weights')

        os.environ['YOLOV4_HOME'] = os.path.dirname(self.model_weights)

        self.inp_dim = cfg.get('INP_DIM', 608)
        self.nms_thresh = cfg.get('NMS_THRES', 0.6)
        self.confidence = cfg.get('CONFIDENCE', 0.05)
        # self.num_classes = cfg.get('NUM_CLASSES', 80)
        self.model = None

    def load_model(self):
        args = self.detector_opt

        print('Loading YOLOv4 model..')
        with torch.cuda.device(args.gpus[0] if len(args.gpus) > 0 else -1):
            self.model = Detector(configfile=self.model_cfg, weightsfile=self.model_weights, conf_threshold=self.confidence, nms_threshold=self.nms_thresh, device_ids=args.gpus, default_device=args.device)
        print("Network successfully loaded")

    def image_preprocess(self, img_source):
        """
        Pre-process the img before fed to the object detection network
        Input: image name(str) or raw image data(ndarray or torch.Tensor,channel GBR)
        Output: pre-processed image data(torch.FloatTensor,(1,3,h,w))
        """
        if isinstance(img_source, str):
            img, orig_img, im_dim_list = prep_image(img_source, self.inp_dim)
        elif isinstance(img_source, torch.Tensor) or isinstance(img_source, np.ndarray):
            img, orig_img, im_dim_list = prep_frame(img_source, self.inp_dim)
        else:
            raise IOError('Unknown image source type: {}'.format(type(img_source)))

        return img

    def images_detection(self, imgs, orig_dim_list):
        """
        Feed the img data into object detection network and 
        collect bbox w.r.t original image size
        Input: imgs(torch.FloatTensor,(b,3,h,w)): pre-processed mini-batch image input
               orig_dim_list(torch.FloatTensor, (b,(w,h,w,h))): original mini-batch image size
        Output: dets(torch.cuda.FloatTensor,(n,(batch_idx,x1,y1,x2,y2,c,s,idx of cls))): human detection results
        """
        args = self.detector_opt
        if not self.model:
            self.load_model()

        with torch.no_grad():
            imgs = imgs.to(args.device) if args else imgs.cuda()
            imgs, class_ids, scores, boxes = self.model.detect(imgs)

            dets = None
            write = False
            for idx, _ in enumerate(imgs):
                if len(boxes[idx]) == 0:
                    continue

                det_new = torch.empty([len(boxes[idx]), 8])
                det_new[:, 0] = idx  #index of img
                det_new[:, 1:3] = torch.from_numpy(boxes[idx][:, 0:2])  # bbox x1,y1
                det_new[:, 3:5] = torch.from_numpy(boxes[idx][:, 2:4])  # bbox x2,y2
                det_new[:, 6:7] = torch.from_numpy(scores[idx])  # cls conf
                det_new[:, 7:8] = torch.from_numpy(class_ids[idx])   # cls idx

                if not write:
                    dets = det_new
                    write = True
                else:
                    dets = torch.cat((dets, det_new))

            if dets is None:
                return 0

            dets = dets.cpu()

            w_scaling_factor = (self.inp_dim / orig_dim_list[:, 0]).view(-1, 1)
            h_scaling_factor = (self.inp_dim / orig_dim_list[:, 1]).view(-1, 1)

            w_scale = torch.empty([dets.shape[0], 1], dtype=torch.float32)
            h_scale = torch.empty([dets.shape[0], 1])

            start = 0
            for idx, _ in enumerate(boxes):
                if len(boxes[idx]) > 0:
                    w_scale[start:start+len(boxes[idx])] = w_scaling_factor[idx]
                    h_scale[start:start+len(boxes[idx])] = h_scaling_factor[idx]
                    start += len(boxes[idx])

            dets[:, [1, 3]] /= w_scale
            dets[:, [2, 4]] /= h_scale

            orig_dim_list = torch.index_select(orig_dim_list, 0, dets[:, 0].long())
            for i in range(dets.shape[0]):
                dets[i, [1, 3]] = torch.clamp(dets[i, [1, 3]], 0.0, orig_dim_list[i, 0])
                dets[i, [2, 4]] = torch.clamp(dets[i, [2, 4]], 0.0, orig_dim_list[i, 1])

            return dets

    def detect_one_img(self, img_id, img_name):
        """
        Detect bboxs in one image
        Input: 'str', full path of image
        Output: '[{"category_id":1,"score":float,"bbox":[x,y,w,h],"image_id":str},...]',
        The output results are similar with coco results type, except that image_id uses full path str
        instead of coco %012d id for generalization. 
        """
        if not self.model:
            self.load_model()

        #pre-process(scale, normalize, ...) the image
        img, orig_img, orig_dim_list = prep_image(img_name, self.inp_dim)

        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)

        # add one dimension at the front for batch if image shape (3,h,w)
        if img.dim() == 3:
            img = img.unsqueeze(0)

        img_dim_list = torch.FloatTensor([orig_dim_list]).repeat(1, 2)
        dets = self.images_detection(img, img_dim_list).tolist()

        dets_results = []
        for det in dets:
            det_dict = dict()
            det_dict["category_id"] = det[6]
            det_dict["score"] = float(det[5])
            det_dict["bbox"] = [det[1], det[2], det[3]-det[1], det[4]-det[2]]
            det_dict["file_name"] = os.path.basename(img_name)
            det_dict["image_id"] = img_id
            dets_results.append(det_dict)

        return dets_results
