import numpy as np
import logging
import traceback

from config import config
from entities.card_detection import (
    HouseholdHandwritingDetection,
    HouseholdHandwritingDetectionFields,
)
from modules.yolov8.YOLOv8 import YOLOv8
from modules.detector.detector_base import HouseholdHandwritingDetector


def calculate_area(box):
    return (box["bot_right_x"] - box["top_left_x"]) * (
        box["bot_right_y"] - box["top_left_y"]
    )


def is_overlapping(box1, box2, overlap_threshold):
    overlap_x = max(
        0,
        min(box1["bot_right_x"], box2["bot_right_x"])
        - max(box1["top_left_x"], box2["top_left_x"]),
    )
    overlap_y = max(
        0,
        min(box1["bot_right_y"], box2["bot_right_y"])
        - max(box1["top_left_y"], box2["top_left_y"]),
    )
    overlap_area = overlap_x * overlap_y
    area_box1 = calculate_area(box1)
    area_box2 = calculate_area(box2)
    overlap_ratio = overlap_area / min(area_box1, area_box2)
    return overlap_ratio > overlap_threshold


def remove_overlapping_boxes(boxes, overlap_threshold=0.7):
    sorted_boxes = sorted(boxes, key=calculate_area, reverse=True)
    non_overlapping_boxes = []
    while sorted_boxes:
        current_box = sorted_boxes.pop(0)
        non_overlapping_boxes.append(current_box)
        sorted_boxes = [
            box
            for box in sorted_boxes
            if not is_overlapping(current_box, box, overlap_threshold)
        ]
    return non_overlapping_boxes


class HouseholdHandwritingYoloDetector(HouseholdHandwritingDetector):
    model_path = config.MODEL_HOUSEHOLD_HANDWRITING_DETECTOR_YOLO
    conf_threshold = 0.3
    iou_threshold = 0.5

    labels = {
        0: "number",
        1: "street",
        2: "unit",
        3: "block",
        4: "ward",
        5: "district",
        6: "province",
        7: "total",
        8: "male",
        9: "female",
        10: "issue_date",
        11: "name",
    }

    def __init__(self) -> None:
        super().__init__()
        self.detector = YOLOv8(
            path=self.model_path,
            conf_thres=self.conf_threshold,
            iou_thres=self.iou_threshold,
        )

    def detect_fields(self, cropped_image: np.ndarray) -> HouseholdHandwritingDetection:
        model_labels = self.labels.values()
        logging.debug(f"detector labels: {model_labels}")
        img0 = cropped_image.copy()
        boxes = {}
        output_result = HouseholdHandwritingDetectionFields()

        household_handwriting_detection = HouseholdHandwritingDetection(
            content=output_result
        )

        # try:
        if cropped_image is not None:
            xyxy, scores, class_ids = self.detector.detect_objects(cropped_image)
            for box, score, class_id in zip(xyxy, scores, class_ids):
                x0, y0, x1, y1 = [int(i) for i in box]
                class_name = self.labels[class_id]

                if class_name not in boxes:
                    boxes[class_name] = [
                        {
                            "top_left_y": round(y0),
                            "bot_right_y": round(y1),
                            "top_left_x": round(x0),
                            "bot_right_x": round(x1),
                            "score": float(score),
                        }
                    ]
                else:
                    boxes[class_name].append(
                        {
                            "top_left_y": round(y0),
                            "bot_right_y": round(y1),
                            "top_left_x": round(x0),
                            "bot_right_x": round(x1),
                            "score": float(score),
                        }
                    )

            if "name" in boxes:
                boxes["name"] = remove_overlapping_boxes(boxes["name"])

            boxes_ = boxes.copy()
            for k in boxes:
                if len(boxes[k]) > 1:
                    if k in ["name", "issue_date"]:
                        img_arr = []
                        sorted_boxes = sorted(boxes[k], key=lambda x: x["top_left_x"])
                        boxes_[k] = sorted_boxes
                        for item in sorted_boxes:
                            field_img = img0[
                                item["top_left_y"] : item["bot_right_y"],
                                item["top_left_x"] : item["bot_right_x"],
                            ]
                            img_arr.append(field_img)
                        output_result[k] = img_arr
                    else:
                        sorted_boxes = sorted(
                            boxes[k], key=lambda x: x["score"], reverse=True
                        )
                        boxes_[k] = sorted_boxes
                        item = sorted_boxes[0]
                        field_img = img0[
                            item["top_left_y"] : item["bot_right_y"],
                            item["top_left_x"] : item["bot_right_x"],
                        ]
                        output_result[k] = [field_img]
                else:
                    item = boxes[k][0]
                    field_img = img0[
                        item["top_left_y"] : item["bot_right_y"],
                        item["top_left_x"] : item["bot_right_x"],
                    ]
                    output_result[k] = [field_img]
        # except Exception as e:
        #     logging.error("{}\n{}".format(e, traceback.format_exc()))

        household_handwriting_detection.content = output_result
        return household_handwriting_detection


# if __name__ == "__main__":
#     import os, matplotlib.pyplot as plt
#     detector = HouseholdHandwritingYoloDetector()

#     # for i in os.listdir("/home/duongnh/Downloads/08/cropped"):
#     #     image = cv2.imread(os.path.join("/home/duongnh/Downloads/08/cropped", i))
#     #     detector.detect_fields(i, image)

#     img_path = os.path.join("/home/duongnh/Downloads/08/cropped", "CM_10-10035108303_Id2.jpg")
#     image = cv2.imread(img_path)

#     output = detector.detect_fields(image)
#     detector_result = output.content.dict(exclude_none=True)
#     # print(content)
#     detector_result_keys = detector_result.keys()
#     print(detector_result_keys)
#     # for key in detector_result_keys:
#     #     # print(key)
#     #     for img in detector_result[key]:
#     #         plt.imshow(img)
#     #         plt.show()

import os, matplotlib.pyplot as plt
import cv2
import pandas as pd
from modules.readers.reader_handwriting import read
from PIL import Image
from tqdm import tqdm

detector = HouseholdHandwritingYoloDetector()
file_names = []
avg_scores = []
overall_scores = []
root_image = "/media/thanhton/TonDz/Eway/Research/Laos/DataLaos/LAOS_HOUSEHOLD_OCR/OCR_SUCCESS"
root_save = "/media/thanhton/TonDz/Eway/Research/Laos/DataLaos/LAOS_HOUSEHOLD_OCR"
for file_name in tqdm(os.listdir(root_image)):
    try:
        image = cv2.imread(os.path.join(root_image, file_name))
        overall_score = 1
        avg_score = 0
        count = 0
        output = detector.detect_fields(image)
        detector_result = output.content.dict(exclude_none=True)
        detector_result_keys = detector_result.keys()
        for key in detector_result_keys:
            for img in detector_result[key]:
                result = read([Image.fromarray(img)])
                for _, score in result:
                    overall_score *= score.item()
                    avg_score += score.item()
                    count += 1
        avg_score = avg_score / count
        file_names.append(file_name)
        avg_scores.append(avg_score)
        overall_scores.append(overall_score)
    except:
        print(file_name)
data = {'file_name': file_names, 'avg_score': avg_scores, 'overall_score': overall_scores}
df = pd.DataFrame(data)
df.to_csv(f'{root_save}/score.csv', index=False)