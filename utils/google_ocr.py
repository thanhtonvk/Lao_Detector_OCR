from google.cloud import vision
import io

gg_ocr_client = vision.ImageAnnotatorClient.from_service_account_json('data/translate-key.json')


def detect_text(path):
    """Detects text in the file."""
    text = None
    with io.open(path, 'rb') as image_file:
        content = image_file.read()
    
    image = vision.Image(content=content)
    
    response = gg_ocr_client.text_detection(image=image)
    texts = response.text_annotations
    if len(texts) > 0:
        text = texts[0].description
    if response.error.message:
        print('{}\nFor more info on error messages, check: '
              'https://cloud.google.com/apis/design/errors'.format(response.error.message))
    return text
