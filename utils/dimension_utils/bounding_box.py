from typing import List

def rotate_bbox_90_deg(bbox, img_width): # just passing width of image is enough for 90 degree rotation.
   x_min, y_min, x_max, y_max = bbox
   x_min_new = y_min
   y_min_new = img_width - x_max
   x_max_new = y_max
   y_max_new = img_width - x_min
   return [x_min_new, y_min_new, x_max_new, y_max_new]


def compute_original_bbox(bbox: List[int], angle_deg: int, img_width: int, img_height: int):
    """
    Compute bounding box in original image after rotating counterclockwise by multiples of 90 degrees

    :param List[int] bbox: [x_min, y_min, x_max, y_max]
    :param int angle_deg: rotated counterclockwise angle in degrees 0|90|180|270|360
    :param int img_width: rotated image's width
    :param int img_height: rotated image's height
    :return List[int] original bbox
    :raises ValueError: if the angle_deg is not multiples of 90 degrees
    """
    if angle_deg not in [0, 90, 180, 270, 360]:
        raise ValueError("angle_dev must be multiples of 90 degrees")
    bbox_new = bbox
    if angle_deg == 0 or angle_deg == 360:
        return bbox_new
    bbox_new = rotate_bbox_90_deg(bbox, img_width)
    if angle_deg == 270:
        return bbox_new
    bbox_new = rotate_bbox_90_deg(bbox_new, img_height)
    if angle_deg == 180:
        return bbox_new
    bbox_new = rotate_bbox_90_deg(bbox_new, img_width)
    if angle_deg == 90:
        return bbox_new
    raise ValueError("angle_dev must be multiples of 90 degrees")
