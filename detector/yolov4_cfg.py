from easydict import EasyDict as edict

cfg = edict()
cfg.CONFIG = 'detector/yolo_v4/cfg/yolov4.cfg'
cfg.WEIGHTS = 'detector/yolo_v4/data/yolov4.weights'
cfg.INP_DIM = 608
cfg.NMS_THRES = 0.6
cfg.CONFIDENCE = 0.05
cfg.NUM_CLASSES = 80
