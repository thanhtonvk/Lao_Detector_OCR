import torch
import onnxruntime
import numpy as np
import scipy.special
import argparse
from modules.recognition.handwriting.dataset import DataTransformer, Converter
from config import config
from utils.model_utils.loader import onnx_model_inference

parser = argparse.ArgumentParser()
parser.add_argument('--batch_max_length', type=int, default=24, help='maximum-label-length')
parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
parser.add_argument('--imgW', type=int, default=192, help='the width of the input image')
# parser.add_argument('--onnx_model_path', type=str, default='./models/reader/v2.2.0_model.onnx', help='path to ONNX model for inference')
parser.add_argument('--onnx_model_path', type=str, default=config.MODEL_HOUSEHOLD_HANDWRITING_READER, help='path to ONNX model for inference')
opt = parser.parse_args()


data_trans = DataTransformer(opt, 'valid')
convert = Converter()
# ort_session = onnxruntime.InferenceSession(opt.onnx_model_path)
ort_session = onnx_model_inference(opt.onnx_model_path)

def read(image_list, dictionary_list=None):
    input_pairs = [data_trans(image) for image in image_list]

    images = [i[0].tolist() for i in input_pairs]
    pad_masks = [i[1].tolist() for i in input_pairs]

    images = torch.tensor(images).cpu().numpy()
    pad_masks = torch.tensor(pad_masks).cpu().numpy().astype('bool')

    ort_inputs = {ort_session.get_inputs()[0].name: images, ort_session.get_inputs()[1].name: pad_masks}
    ort_outputs = ort_session.run(None, ort_inputs)
    _, _, logit = ort_outputs
    prob = scipy.special.softmax(logit, axis=1)
    pred = np.argmax(prob, axis=1)
    max_prob = np.max(prob, axis=1)
    pred_txt = [''.join(list(map(convert.decode, p.tolist()))) for p in pred]

    confident_score = np.cumprod(max_prob, axis=1)[:, -1].tolist()
    output = list(zip(pred_txt, torch.as_tensor(confident_score)))

    # output = []
    # for img, pad_mask in input_pairs:
    #     img = img.unsqueeze(dim=0).cpu().numpy()
    #     pad_mask = pad_mask.unsqueeze(dim=0).cpu().numpy().astype('bool')
        
    #     ort_inputs = {ort_session.get_inputs()[0].name: img, ort_session.get_inputs()[1].name: pad_mask}
    #     ort_outputs = ort_session.run(None, ort_inputs)
    
    #     _, _, logit = ort_outputs
    #     prob = scipy.special.softmax(logit, axis=1)
    #     pred = np.argmax(prob, axis=1)
    #     max_prob = np.max(prob, axis=1)
    #     pred_txt = [''.join(list(map(convert.decode, p.tolist()))) for p in pred]
    #     confident_score = np.cumprod(max_prob, axis=1)[:, -1].tolist()
    #     output += list(zip(pred_txt, torch.as_tensor(confident_score)))
    return output
