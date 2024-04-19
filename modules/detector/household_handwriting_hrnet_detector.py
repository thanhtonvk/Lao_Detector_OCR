import logging
import traceback
import cv2
import numpy as np
from config import config
from modules.hrnet.dataset import *
from modules.detector.model_hrnet import HRModel
from entities.card_detection import (
    HouseholdHandwritingDetection,
    HouseholdHandwritingDetectionFields,
)
from modules.detector.detector_base import HouseholdHandwritingDetector


def calculate_iou(box1, box2):
    x1_box1, y1_box1, x2_box1, y2_box1 = box1
    x1_box2, y1_box2, x2_box2, y2_box2 = box2
    area_box1 = (x2_box1 - x1_box1) * (y2_box1 - y1_box1)
    area_box2 = (x2_box2 - x1_box2) * (y2_box2 - y1_box2)
    x_overlap = max(0, min(x2_box1, x2_box2) - max(x1_box1, x1_box2))
    y_overlap = max(0, min(y2_box1, y2_box2) - max(y1_box1, y1_box2))
    overlap_area = x_overlap * y_overlap
    iou = overlap_area / (area_box1 + area_box2 - overlap_area)
    return iou


def remove_high_iou_boxes(issue_date_boxes, all_boxes, iou_threshold=0.3):
    filtered_boxes = []
    for issue_box in issue_date_boxes:
        is_high_iou = False
        for field_name in all_boxes:
            if field_name != "issue_date":
                for box in all_boxes[field_name]:
                    if calculate_iou(issue_box, box) > iou_threshold:
                        is_high_iou = True
                        break
                if is_high_iou:
                    break
        if not is_high_iou:
            filtered_boxes.append(issue_box)
    return filtered_boxes


class HouseholdHandwritingHrnetDetector(HouseholdHandwritingDetector):
    model_path = config.MODEL_HOUSEHOLD_HANDWRITING_DETECTOR
    labels = [
        "number",
        "street",
        "unit",
        "block",
        "ward",
        "district",
        "province",
        "total",
        "male",
        "female",
        "issue_date",
        "name",
    ]

    def __init__(self) -> None:
        super().__init__()
        self.detector = HRModel(self.model_path)

    def detect_fields(self, cropped_image: np.ndarray) -> HouseholdHandwritingDetection:
        pred = self.detector.predict_image_data(cropped_image)
        img0 = cropped_image
        boxes = {}
        height, width = img0.shape[0], img0.shape[1]
        pred["issue_date"] = remove_high_iou_boxes(pred["issue_date"], pred)
        for label, box in pred.items():
            if len(box) != 0:
                if label not in boxes:
                    boxes[label] = []
                for b in box:
                    boxes[label].append(
                        {
                            "top_left_y": round(b[1]),
                            "bot_right_y": round(b[3]),
                            "top_left_x": round(b[0]),
                            "bot_right_x": round(b[2]),
                        }
                    )
        output_result = HouseholdHandwritingDetectionFields()
        for k in boxes:
            img_arr = []
            sorted_boxes = sorted(boxes[k], key=lambda x: x["top_left_x"])
            for item in sorted_boxes:
                cropped__img = img0[
                    item["top_left_y"] : item["bot_right_y"],
                    item["top_left_x"] : item["bot_right_x"],
                ]
                item["img"] = cropped__img
                img_arr.append(item)
            output_result[k] = img_arr

        return HouseholdHandwritingDetection(content=output_result)
