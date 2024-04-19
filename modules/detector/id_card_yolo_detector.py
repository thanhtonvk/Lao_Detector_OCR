import numpy as np

from config import config
from modules.detector.detector_base import IdCardDetector
from entities.card_detection import IdCardDetection, IdCardDetectionFields
from modules.yolo.models.v5 import YoloModel
from utils.utils import scale_coords


class IdCardYoloDetector(IdCardDetector):
    model_weights = config.MODEL_DETECTOR
    conf_threshold = 0.4
    iou_threshold = 0.2

    def __init__(self) -> None:
        super().__init__()
        self.detector = YoloModel(
            model_weights=self.model_weights,
            conf_thres=self.conf_threshold,
            iou_thres=self.iou_threshold,
        )
        # self.labels = self.detector.labels

    def detect_fields(self, cropped_image: np.ndarray) -> IdCardDetection:
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
        detection_fields = IdCardDetectionFields()
        for k in boxes:
            if len(boxes[k]) > 1:
                img_arr = []
                if k in [
                    "id_number",
                    "issue_date",
                    "expiry",
                    "ethnic",
                    "nationality",
                    "province",
                    "place_of_birth",
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
        return IdCardDetection(content=detection_fields)
import os, matplotlib.pyplot as plt
import cv2
import pandas as pd
from modules.readers.reader import read
from PIL import Image
from tqdm import tqdm

detector = IdCardYoloDetector()
file_names = []
avg_scores = []
overall_scores = []
root = "DataLaos/LAOS_ID-CARD_OCR/OCR_SUCCESS"
root_save = "DataLaos/LAOS_ID-CARD_OCR"
for file_name in tqdm(os.listdir(root)):
    try:
        image = cv2.imread(os.path.join(root, file_name))
        overall_score = 1
        avg_score = 0
        count = 0
        output = detector.detect_fields(image)
        detector_result = output.content.dict(exclude_none=True)
        detector_result_keys = detector_result.keys()
        for key in detector_result_keys:
            for img in detector_result[key]:
                result = read([Image.fromarray(img['img'])])
                for _, score in result:
                    overall_score *= score.item()
                    avg_score += score.item()
                    count += 1
        avg_score = avg_score / count
        file_names.append(file_name)
        avg_scores.append(avg_score)
        overall_scores.append(overall_score)
    except Exception as e:
        print(file_name)
        print(e)
data = {'file_name': file_names, 'avg_score': avg_scores, 'overall_score': overall_scores}
df = pd.DataFrame(data)
df.to_csv(f'{root_save}/score.csv', index=False)