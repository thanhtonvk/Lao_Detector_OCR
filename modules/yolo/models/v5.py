from config import config
from modules.yolo.models.experimental import *
from utils.datasets import *
from utils.utils import *
import numpy as np
import cv2


class YoloModel:
    """
    conf_thres: object confidence threshold
    iou_thres: IOU threshold for NMS
    augment: augmented inference
    """
    
    def __init__(self, model_weights, device=torch_utils.select_device(config.DEVICE), img_size=416, conf_thres=0.8,
                 iou_thres=0.45, augment=False, v5=False):
        self.model_weights = model_weights
        self.device = device
        print(device)
        self.half = self.device.type != 'cpu'
        if v5:
            self.model = attempt_load_v5(self.model_weights, map_location=self.device)
        else:
            self.model = attempt_load(self.model_weights, map_location=self.device)
        self.img_size = check_img_size(img_size, s=self.model.stride.max())  # check img_size
        if self.half:
            self.model.half()  # to FP16
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.augment = augment
        
        self.labels = self.model.names

        print('[YOLO]', self.model_weights, self.labels, self.device)

    def predict_image(self, img_path):
        classify = False
        dataset = LoadImages(img_path, img_size=self.img_size)
        img = torch.zeros((1, 3, self.img_size, self.img_size), device=self.device)  # init img
        _ = self.model(img.half() if self.half else img) if self.device.type != 'cpu' else None  # run once
        arr_im0 = []
        paths = []
        for path, img, im0s, vid_cap in dataset:
            arr_im0.append(im0s)
            paths.append(path)
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            pred = self.model(img, augment=self.augment)[0]
            
            # Apply NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)
            
            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, self.model, img, im0s)
        return img, paths, arr_im0, pred
    
    def predict_image_data(self, img0):
        img = torch.zeros((1, 3, self.img_size, self.img_size), device=self.device)  # init img
        _ = self.model(img.half() if self.half else img) if self.device.type != 'cpu' else None  # run once
        
        # Padded resize
        img = letterbox(img0, new_shape=self.img_size)[0]
        
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        
        arr_im0 = []
        
        arr_im0.append(img0)
        # paths.append(path)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        pred = self.model(img, augment=self.augment)[0]
        
        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)
        
        return img, arr_im0, pred
    
    def detect_card_bbox(self, img_path=None, image_data=None):
        card_bbox = None
        label = None
        score = None
        if img_path is not None:
            img, paths, arr_im0, pred = self.predict_image(img_path)
        if image_data is not None:
            img, arr_im0, pred = self.predict_image_data(image_data)
        
        for i, det in enumerate(pred):
            im0 = arr_im0[i]
            # Rescale boxes from img_size to im0 size
            if det is not None:
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                for item in det:
                    x0, y0, x1, y1, score, category = item
                    card_bbox = [int(x0), int(y0), int(x1), int(y1)]
                    label = self.model.names[int(category)]
                    score = float(score)
        return card_bbox, label, score

    def detect_bboxes(self, img_path=None, image_data=None):
        if img_path is not None:
            img, paths, arr_im0, pred = self.predict_image(img_path)
        if image_data is not None:
            img, arr_im0, pred = self.predict_image_data(image_data)
        bboxes = []
        for i, det in enumerate(pred):
            im0 = arr_im0[i]
            # Rescale boxes from img_size to im0 size
            if det is not None:
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                for item in det:
                    x0, y0, x1, y1, score, category = item
                    bbox = [int(x0), int(y0), int(x1), int(y1)]
                    label = self.model.names[int(category)]
                    score = float(score)
                    bboxes.append((bbox, label, score))
        return bboxes
