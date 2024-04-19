from config import env

ENVIRONMENT = env.ENVIRONMENT
PORT = env.get_var("PORT", 23000)
DEVICE = env.get_var("DEVICE", "cpu")
DEBUG = env.get_bool("DEBUG", False)
ENCODED_MODELS = env.get_bool("ENCODED_MODELS", False)

# Version control
TARGET_VERSION = env.get_int("TARGET_VERSION", 0)

# License
LICENSE_FILE = env.get_var("LICENSE_FILE", "")
PUBLIC_KEY_FILE = env.get_var("PUBLIC_KEY_FILE", "")
LICENSE = None
PUBLIC_KEY = None

# Directories
LOGS_DIR = "logs"

# Model faces
MODEL_FACE_RETINA = "models/RetinaFace/Resnet50_Final.pth"
MODEL_FACE_LANDMARK = "models/onnx/face_landmark_detection.onnx"
MODEL_FACE_POSE = "models/onnx/face_head_pose_estimate.onnx"
MODEL_FACE_DETECT = "models/onnx/face_detect.onnx"

MODEL_GENERAL_READER = (
    "models/reader/v7.0.1_str_with_augment_vm_lm_with_label_smoothing_CELoss.pth"
)

# Model Laos ID Card Config
MODEL_CROPPER_BBOX = "models/cropper/classify_laos_100322.onnx"
MODEL_CROPPER_BBOX_V2 = "models/classification/best_v8_2703.onnx"
MODEL_CROPPER_POINTS = "models/cropper/laos_card_points_v5s_416.pt"
# MODEL_DETECTOR = "models/detector/laos_detector_fields_v5s_416_20210605.pt"
MODEL_DETECTOR = "models/detector/laos_id_detector_20220331_sz416_exp2.pt"
MODEL_READER = "models/reader/None_ResNet_BiLSTM_Attn_20220321_H48.pth"
# Model Laos Passport Config
MODEL_PASSPORT_CROPPER_BBOX = (
    "models/cropper/passport/best_laos_passport_classify_07072021_v5s_416.pt"
)
MODEL_PASSPORT_CROPPER_POINTS = (
    "models/cropper/passport/best_laos_passport_points_07072021_v5s_416.pt"
)
MODEL_PASSPORT_DETECTOR = (
    "models/detector/passport/laos_passport_detector_v5_13082021.pt"
)
MODEL_READER_FIELD_PASSPORT = "models/reader/passport/lao_passport_20211210.pth"

# Model passport mrz
MODEL_CROPPER_PASSPORT_MRZ = "models/mrz/v4.0.0_passport_cropper.onnx"
MODEL_READER_PASSPORT_MRZ = "models/mrz/v2.0.3_passport_reader.onnx"
MODEL_CROPPER_PASSPORT_MRZ_V2 = "models/mrz/v5.0.0_passport_cropper.onnx"
MODEL_READER_PASSPORT_MRZ_V2 = "models/mrz/v2.0.3_passport_reader.onnx"

# Model Laos Household Config
MODEL_HOUSEHOLD_CROPPER_BBOX = (
    "models/cropper/laos_household_classify_20220105_v5s_416.pt"
)
MODEL_HOUSEHOLD_CROPPER_POINTS = (
    "models/cropper/laos_household_points_20211224_v5s_416.pt"
)
MODEL_HOUSEHOLD_DETECTOR = "models/detector/laos_household_fields_20211230_v5s_416.pt"
MODEL_HOUSEHOLD_HANDWRITING_DETECTOR = "models/detector/hrnet_detector_v2.pth"
MODEL_HOUSEHOLD_HANDWRITING_DETECTOR_YOLO = "models/detector/laos-hh-handwriting-yolo-v8-251023-duongnh.onnx"
MODEL_HOUSEHOLD_HANDWRITING_READER = "models/reader/v2.2.0_model.onnx"
MODEL_HOUSEHOLD_NEXT_HANDWRITING_CROPPER_BBOX = (
    "models/cropper/laos_household_next_handwriting_classify_20220124_v5s_416.pt"
)
# MODEL_HOUSEHOLD_NEXT_HANDWRITING_CROPPER_POINTS = ""
MODEL_HOUSEHOLD_NEXT_HANDWRITING_DETECTOR = (
    "models/detector/laos_household_next_handwriting_fields_20220126_v5s_416.pt"
)
MODEL_HOUSEHOLD_NEXT_HANDWRITING_READER = (
    "models/reader/handwriting_None_ResNet_BiLSTM_Attn_20220122.pth"
)
MODEL_HOUSEHOLD_NEXT_PRINT_CROPPER_BBOX = (
    "models/cropper/laos_household_next_print_classify_20220127_v5s_640.pt"
)
MODEL_HOUSEHOLD_NEXT_PRINT_CROPPER_POINTS = (
    "models/cropper/laos_household_next_print_points_20220128_v5s_640.pt"
)
MODEL_HOUSEHOLD_NEXT_PRINT_DETECTOR = (
    "models/detector/laos_household_next_print_fields_20220128_v5s_640.pt"
)
MODEL_ANTIFRAUD = "models/antifraud/onnx_antifraud.onnx"

MODEL_HOUSEHOLD_NEXT_CROPPER_BBOX = (
    "models/cropper/classify_household_next_laos_130423.onnx"
)

MODEL_CARD_QUALITY = "models/valid_card/card_quality.pth"

# YOLOv5 pretrained model
MODEL_YOLOV5s = "models/classification/yolov5s_v5.pt"

# SCR Facedetector
MODEL_FACE_DETECTOR = "models/face/face_detector.onnx"
# arc face
MODEL_FACE_COMPARE = "models/face/r50_compare_face.onnx"
# model face quality
MODEL_FACE_QUALITY = "models/face/face_quality.onnx"
# model face part
MODEL_DETECT_FULL_FACE = "models/face/face_part.onnx"

# Sentry configs
SENTRY_DSN = env.get_var("SENTRY_DSN", "")
SENTRY_TRACES_SAMPLE_RATE = env.get_float("SENTRY_TRACES_SAMPLE_RATE", 0.5)
SENTRY_PROFILES_SAMPLE_RATE = env.get_float(
    "SENTRY_PROFILES_SAMPLE_RATE", 0.25)

# Production environment
# if ENVIRONMENT == "production":
# Sentry configs is required in production
# SENTRY_DSN = env.get_var("SENTRY_DSN")
