import numpy as np

from config import config
from modules.detector.detector_base import HouseholdPrintDetector
from entities.card_detection import HouseholdPrintDetection, HouseholdDetectionFields
from modules.yolo.models.v5 import YoloModel
from utils.utils import scale_coords


class HouseholdPrintYoloDetector(HouseholdPrintDetector):
    model_weights = config.MODEL_HOUSEHOLD_DETECTOR
    conf_threshold = 0.5
    iou_threshold = 0.25
    v5 = True

    def __init__(self) -> None:
        super().__init__()
        self.detector = YoloModel(
            model_weights=self.model_weights,
            conf_thres=self.conf_threshold,
            iou_thres=self.iou_threshold,
            v5=self.v5,
        )

    def detect_fields(self, cropped_image: np.ndarray) -> HouseholdPrintDetection:
        model_labels = self.detector.labels
        img, arr_im0, pred = self.detector.predict_image_data(cropped_image)
        img0 = cropped_image
        boxes = {}
        for i, det in enumerate(pred):  # detections per image
            s, im0 = "", arr_im0[i]
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                # Write results
                for item in det:
                    x0, y0, x1, y1, percent, category = item
                    x0 = int(x0)
                    y0 = int(y0)
                    x1 = int(x1)
                    y1 = int(y1)
                    category = int(category)
                    w = x1 - x0
                    h = y1 - y0
                    label = model_labels[category]
                    if label not in boxes:
                        boxes[label] = []
                    boxes[label].append(
                        {
                            "top_left_y": round(y0),
                            "bot_right_y": round(y1),
                            "top_left_x": round(x0),
                            "bot_right_x": round(x1),
                            "percent": percent,
                        }
                    )
        detection_fields = HouseholdDetectionFields()
        for k in boxes:
            if len(boxes[k]) > 1:
                img_arr = []
                if k in [
                    "issue_date",
                    "province",
                    "district",
                ]:
                    sorted_boxes = sorted(boxes[k], key=lambda x: -x["percent"])
                    for item in sorted_boxes:
                        # item = boxes[k][i]
                        cropped__img = img0[
                            item["top_left_y"] : item["bot_right_y"],
                            item["top_left_x"] : item["bot_right_x"],
                        ]
                        item["img"] = cropped__img
                        img_arr.append(item)
                        break
                else:
                    sorted_boxes = sorted(boxes[k], key=lambda x: x["top_left_x"])
                    for item in sorted_boxes:
                        cropped__img = img0[
                            item["top_left_y"] : item["bot_right_y"],
                            item["top_left_x"] : item["bot_right_x"],
                        ]
                        item["img"] = cropped__img
                        img_arr.append(item)
                detection_fields[k] = img_arr
            else:
                item = boxes[k][0]
                cropped__img = img0[
                    item["top_left_y"] : item["bot_right_y"],
                    item["top_left_x"] : item["bot_right_x"],
                ]
                item["img"] = cropped__img
                detection_fields[k] = [item]
        return HouseholdPrintDetection(content=detection_fields)
