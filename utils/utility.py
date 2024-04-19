import base64
import hashlib
import io
import os
import cv2
from PIL import Image
import numpy as np


def is_grey_scale(img):
    h, w, c = img.shape
    total_pixels = w * h
    r = img[:, :, 0]
    g = img[:, :, 1]
    result = np.absolute(r - g)
    gray_pixels = (result < 6).sum()
    score = gray_pixels / total_pixels
    result = score >= 0.7
    print('[is_grey_scale]', result, str(score))
    return result


def md5_file(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def get_md5_string(base64_string):
    return hashlib.md5(base64_string.encode('utf-8')).hexdigest()


# Take in base64 string and return cv image
def stringToRGB(base64_string):
    imgdata = base64.b64decode(base64_string.replace("data:image/jpeg;base64,", ""))
    img = Image.open(io.BytesIO(imgdata))

    img = img.convert("RGB")
    return img


def resize(img, maxH=32):
    # print(img.shape)
    height = img.shape[0]
    width = img.shape[1]
    ratio = width / height

    # print(ratio)

    img = cv2.resize(img, (int(maxH * ratio), maxH))

    # print(img.size)

    return img


def resizePIL(image, maxH=32):
    width, height = image.size
    ratio = width / height

    new_width = int(ratio * maxH)

    resize_image = image.resize((new_width, maxH))

    return resize_image


def imageToBase64(image):
    # cropped_image_file = "cropped.jpg"
    # cv2.imwrite(cropped_image_file, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    #
    # encoded_string = base64.b64encode(open(cropped_image_file, "rb").read())
    # return "data:image/jpeg;base64," + str(encoded_string, "utf-8")

    rgb = image[..., ::-1].copy()

    # buffered = BytesIO()
    retval, buffered = cv2.imencode('.jpg', rgb)
    io_buf = io.BytesIO(buffered)

    img_str = base64.b64encode(io_buf.getvalue())
    return "data:image/jpeg;base64," + str(img_str, "utf-8")


def convertBGRtoRGB(img):
    b, g, r = img.split()
    return Image.merge("RGB", (r, g, b))


def convertRGBtoBGR(img):
    r, g, b = img.split()
    return Image.merge("RGB", (b, g, r))


def fileToBase64(file_location):
    if os.path.exists(file_location):
        encoded_string = base64.b64encode(open(file_location, "rb").read())
        return "data:image/jpeg;base64," + str(encoded_string, "utf-8")
    else:
        return None


def cv2ToPil(cv_image):
    # convert from openCV2 to PIL. Notice the COLOR_BGR2RGB which means that
    # the color is converted from BGR to RGB
    color_coverted = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(color_coverted)

    return pil_image


if __name__ == '__main__':
    import numpy as np

    line = "test/201911010115_verify=true&id=233132574_2Q=="
    img = Image.open(line)

    print(type(img))

    img = img.convert("RGB")
    size = img.size
    img = np.array(img)

    img = resize(img)

    from matplotlib import pyplot as plt

    f = plt.figure()
    f.set_size_inches(6, 4)
    plt.imshow(img)
    plt.show()
