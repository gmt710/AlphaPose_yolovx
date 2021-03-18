import cv2
import numpy as np


def letterbox_image(img, inp_dim):
    '''resize image with unchanged aspect ratio using padding'''
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w / img_w, h / img_h))
    new_h = int(img_h * min(w / img_w, h / img_h))
    resized_image = cv2.resize(img, (new_w, new_h))#default is INTER_LINEAR, interpolation=cv2.INTER_CUBIC)

    canvas = np.full((inp_dim[1], inp_dim[0], 3), 0)

    canvas[0:new_h, 0:new_w, :] = resized_image

    return canvas


def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network.

    Returns a Variable
    """

    orig_im = cv2.imread(img)
    dim = orig_im.shape[1], orig_im.shape[0]
    img_ = orig_im
    img_ = cv2.resize(img_, (inp_dim, inp_dim))
    img_ = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
    img_ = img_.transpose((2, 0, 1)).copy()
    return img_, orig_im, dim


def prep_frame(img, inp_dim):
    """
    Prepare image for inputting to the neural network.

    Returns a Variable
    """

    orig_img = img
    dim = orig_img.shape[1], orig_img.shape[0]
    img_ = cv2.resize(orig_img, (inp_dim, inp_dim))
    img_ = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
    img_ = img_.transpose((2, 0, 1)).copy()

    return img_, orig_img, dim
