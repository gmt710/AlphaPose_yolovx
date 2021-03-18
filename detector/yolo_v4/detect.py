import numpy as np
import torch

from yolov4.tool.class_names import COCO_NAMES
from yolov4.tool.config import YOLO_V4
from yolov4.tool.darknet2pytorch import Darknet
from yolov4.tool.torch_utils import do_detect
from yolov4.tool.utils import load_class_names
from yolov4.tool.weights import download_weights


class Detector(object):
    def __init__(self, configfile=YOLO_V4, weightsfile=None, conf_threshold=0.5, nms_threshold=0.6, default_device=-1, device_ids=[-1]):
        self.config_file = configfile
        self.weights_file = weightsfile
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold

        self.device_ids = device_ids
        self.default_device = default_device

        self.use_cuda = len(device_ids) > 0 and device_ids[0] >= 0

        self._init_detector()

    def _init_detector(self):
        if self.weights_file is None:
            self.weights_file = download_weights()

        detector = Darknet(self.config_file, use_cuda=self.use_cuda)
        detector.load_weights(self.weights_file)

        self.detector = detector

        if len(self.device_ids) > 1:
            self.detector = torch.nn.DataParallel(self.detector, device_ids=self.device_ids).to(self.default_device)
        else:
            self.detector.to(self.default_device)

    def _yolov4_detect(self, imgs):
        detections = do_detect(self.detector, imgs, self.conf_threshold, self.nms_threshold, self.use_cuda)

        class_names = load_class_names(COCO_NAMES)
        person_class_id = class_names.index('person')

        person_detections = []
        for det in detections:
            if len(det) == 0:
                person_detections.append(np.array([]))
                continue

            filtered_class_ids = np.array(det)[:, 6] == person_class_id
            filtered_detections = np.array(det)[filtered_class_ids, :]
            person_detections.append(filtered_detections)

        width = imgs.shape[2]
        height = imgs.shape[1]

        boxes = []
        scores = []
        class_ids = []
        for det in person_detections:
            if det.ndim == 2 and det.shape[1] == 7:
                det_boxes = det[:, 0:4]
                # Convert from 0.0-1.0 float value to specific pixel location
                det_boxes[:, [0, 2]] *= width
                det_boxes[:, [1, 3]] *= height

                det_scores = det[:, 5:6]
                det_class_ids = det[:, 6:7]
            else:
                det_boxes = np.array([])
                det_scores = np.array([])
                det_class_ids = np.array([])

            boxes.append(det_boxes)
            scores.append(det_scores)
            class_ids.append(det_class_ids)

        return imgs, class_ids, scores, boxes

    def detect(self, imgs):
        # transpose from [<<batch size>>, 3, 608, 608] to [<<batch size>>, 608, 608, 3]
        #timgs = np.array(imgs).transpose((0, 2, 3, 1)).copy()
        timgs = imgs.clone().permute((0, 2, 3, 1))
        imgs, class_ids, scores, bounding_boxes = self._yolov4_detect(timgs)

        filtered_scores = []
        filtered_boxes = []
        filtered_class_ids = []
        for idx, score in enumerate(scores):
            if scores[idx].ndim == 2:
                threshold_indices = np.where(score[:, 0] > self.conf_threshold)
            else:
                threshold_indices = np.array([])

            if len(threshold_indices) == 0 or len(threshold_indices[0]) == 0:
                filtered_scores.append([])
                filtered_boxes.append([])
                filtered_class_ids.append([])
                continue

            filtered_scores.append(score[threshold_indices])
            filtered_boxes.append(bounding_boxes[idx][threshold_indices])
            filtered_class_ids.append(class_ids[idx][threshold_indices])

        return imgs, filtered_class_ids, filtered_scores, filtered_boxes
