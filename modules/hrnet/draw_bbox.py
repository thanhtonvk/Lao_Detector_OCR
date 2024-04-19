import os, json
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import cv2

def draw_bbox(img_name):
    img_folder = "/content/drive/MyDrive/detector-laos/dataset/Laos_household_handwriting_bb_122021"
    img_file = os.path.join(img_folder, img_name)

    processed_folder = "/content/drive/MyDrive/household-laos/label"
    processed_file = os.path.join(processed_folder, img_name.split(".")[0] + ".txt")

    image = cv2.imread(filename=img_file)

    with open(processed_file, "r") as f:
        data = f.readlines()
    # print([i.strip() for i in data])

    data = [i.strip() for i in data]

    for field in data:
        field_name = field.split("|")[1]
        x1, y1, x2, y2 = int(field.split("|")[2]), int(field.split("|")[3]), \
                         int(field.split("|")[4]), int(field.split("|")[5])
        cv2.rectangle(image, (x1, y1), (x2, y2), color = (255, 0, 0), thickness = 1)
    plt.imshow(image)
    plt.show()

# draw_bbox("renderfile (261).jpg")