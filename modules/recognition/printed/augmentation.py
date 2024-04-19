import random
import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
import numpy as np
import torch
from PIL import Image, ImageFilter
import cv2


def HorizontalMotionBlur(img, bboxes):
    ksize = np.random.randint(3, 11)
    kernel = np.zeros((ksize, ksize))
    kernel[int((ksize-1) / 2), :] = np.ones(ksize)
    kernel /= ksize
    img = cv2.filter2D(np.array(img), -1, kernel)
    return Image.fromarray(img), bboxes

def VerticalMotionBlur(img, bboxes):
    ksize = np.random.randint(3, 11)
    kernel = np.zeros((ksize, ksize))
    kernel[:, int((ksize - 1) / 2)] = np.ones(ksize)
    kernel /= ksize
    img = cv2.filter2D(np.array(img), -1, kernel)
    return Image.fromarray(img), bboxes

def Rotation(img, bboxes):
    angle = np.random.randint(-7, 7)
    img = img.rotate(angle, expand=1)
    return img, bboxes

def AutoContrast(img, bboxes):
    return PIL.ImageOps.autocontrast(img), bboxes

def Invert(img, bboxes):
    return PIL.ImageOps.invert(img), bboxes

def Equalize(img, bboxes):
    return PIL.ImageOps.equalize(img), bboxes

def Solarize(img, bboxes):
    v = np.random.randint(0, 255)
    return PIL.ImageOps.solarize(img, v), bboxes

def SolarizeAdd(img, bboxes=None, addition=0, threshold=128):
    img_np = np.array(img).astype(np.int)
    img_np = img_np + addition
    img_np = np.clip(img_np, 0, 255)
    img_np = img_np.astype(np.uint8)
    img = Image.fromarray(img_np)
    return PIL.ImageOps.solarize(img, threshold), bboxes

def Posterize(img, bboxes):
    v = np.random.randint(4, 8)
    return PIL.ImageOps.posterize(img, v), bboxes

def Contrast(img, bboxes):
    return PIL.ImageEnhance.Contrast(img).enhance(1.5), bboxes

def Color(img, bboxes):
    return PIL.ImageEnhance.Color(img).enhance(1.5), bboxes

def Brightness(img, bboxes):
    return PIL.ImageEnhance.Brightness(img).enhance(1.5), bboxes

def Sharpness(img, bboxes):
    return img.filter(ImageFilter.EDGE_ENHANCE), bboxes

def Cutout(img, v):
    assert 0.0 <= v <= 0.2
    if v <= 0.:
        return img
    v = v * img.size[0]
    return CutoutAbs(img, v)

def CutoutAbs(img, bboxes):
    v = np.random.randint(0, 10)
    w, h = img.size
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)

    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)

    xy = (x0, y0, x1, y1)
    color = (125, 123, 114)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)

    if bboxes is not None:
        xy = torch.as_tensor(xy).unsqueeze(dim=0).repeat(bboxes.size(0), 1).long()
        dx = torch.clamp(
            torch.stack([xy[:, 2], bboxes[:, 2]], dim=-1).min(dim=-1)[0] -
            torch.stack([xy[:, 0], bboxes[:, 0]], dim=-1).max(dim=-1)[0],
            min=0)
        dy = torch.clamp(
            torch.stack([xy[:, 3], bboxes[:, 3]], dim=-1).min(dim=-1)[0] -
            torch.stack([xy[:, 1], bboxes[:, 1]], dim=-1).max(dim=-1)[0],
            min=0)
        intersection = dx * dy
        area = (bboxes[:, 2:] - bboxes[:, :2])[:, 0] * (bboxes[:, 2:] - bboxes[:, :2])[:, 1]
        conf = intersection / area
        valid_ind = torch.where(conf > 0.7, True, False)
        n_bboxes = bboxes[~valid_ind, :]
        return img, n_bboxes

    else:
        return img, bboxes

def augment_list():
    # ops = [
    #     (Rotation, 0, 10),
    #     (AutoContrast, 0, 1),
    #     (Equalize, 0, 1),
    #     (Invert, 0, 1),
    #     (Posterize, 0, 20),
    #     (Solarize, 0, 24),
    #     (SolarizeAdd, 0, 64),
    #     (Color, 0.1, 1.9),
    #     (Contrast, 0.1, 1.9),
    #     (Brightness, 0.1, 3),
    #     (Sharpness, 0.1, 1.9),
    #     (CutoutAbs, 0, 16)
    # ]

    ops = [
        HorizontalMotionBlur,
        VerticalMotionBlur,
        Rotation,
        AutoContrast,
        Equalize,
        Invert,
        Posterize,
        SolarizeAdd,
        Color,
        Contrast,
        Brightness,
        Sharpness,
        CutoutAbs
    ]
    return ops

class RandAugment:
    def __init__(self, n, m, augment_list):
        self._n = n
        self._m = m
        self._augment_list = augment_list

    def __call__(self, img, bboxes=None):
        rand = random.random()
        if rand > 0.25:
            ops = np.random.choice(self._augment_list, self._n)
            for op in ops:
                img, bboxes = op(img, bboxes)
        return img, bboxes
