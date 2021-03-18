from easydict import EasyDict as edict

cfg = edict()
cfg.CONFIG = 'detector/yolov5/models/yolov5x.yaml'
cfg.WEIGHTS = 'detector/yolov5/weights/yolov5x.pt'
cfg.INP_DIM =  640
cfg.NMS_THRES =  0.6
cfg.CONFIDENCE = 0.1
cfg.NUM_CLASSES = 80
