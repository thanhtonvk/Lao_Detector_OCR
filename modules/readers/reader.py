# import torch
# import onnxruntime
# import numpy as np
# import scipy.special
# import argparse
# from modules.recognition.printed.dataset import DataTransformer, Converter

import torch
import numpy as np
import argparse
from modules.recognition.printed.dataset import s2i, symbols, DataTransformer, Converter
from modules.recognition.printed.model import Model
from modules.recognition.printed.utils import post_process
from config import config
from utils.model_utils.loader import torch_load_content
from utils.ocr_paddle import PaddleReader

parser = argparse.ArgumentParser()
parser.add_argument("--rgb", type=bool, default=True, help="use rgb input")
# parser.add_argument('--saved_model', default='./models/reader/v7.0.1_str_with_augment_vm_lm_with_label_smoothing_CELoss.pth', help="path to model to continue training")
parser.add_argument(
    "--saved_model",
    default=config.MODEL_GENERAL_READER,
    help="path to model to continue training",
)
parser.add_argument(
    "--batch_max_length", type=int, default=15, help="maximum-label-length"
)
parser.add_argument(
    "--input_channels", type=int, default=1, help="the number of input channel of Model"
)
parser.add_argument(
    "--output_channels",
    type=int,
    default=512,
    help="the number of output channel of Feature extractor",
)
parser.add_argument(
    "--imgH", type=int, default=32, help="the height of the input image"
)
parser.add_argument(
    "--imgW", type=int, default=192, help="the width of the input image"
)
parser.add_argument(
    "--d_model", type=int, default=512, help="the embeding size of the language model"
)
parser.add_argument(
    "--n_heads", type=int, default=8, help="number of heads in multihead attention"
)
parser.add_argument(
    "--n_layers", type=int, default=6, help="number of layers in language model"
)
parser.add_argument("--pad_token", type=int, default=s2i["<PAD>"], help="pad token")


opt = parser.parse_args()
opt.symbols = symbols
data_trans = DataTransformer(opt, "valid")
convert = Converter()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = torch.nn.DataParallel(Model(opt)).cuda()
model = torch.nn.DataParallel(Model(opt)).to(device)
model = model.eval()

# ckpt = torch.load(opt.saved_model, map_location='cuda:0')
# ckpt = torch.load(opt.saved_model, map_location=device)
ckpt = torch.load(torch_load_content(opt.saved_model), map_location=device)
model.load_state_dict(ckpt["state_dict"], strict=False)
print("Restore model")

for module in model.modules():
    if hasattr(module, "switch_to_deploy"):
        module.switch_to_deploy()

paddleReader = PaddleReader()


def read(image_list, dictionary_list=None):
    with torch.no_grad():
        input_pairs = [data_trans(image) for image in image_list]
        input_tensor = torch.stack([i[0] for i in input_pairs], dim=0)
        pad_mask = torch.stack([i[1] for i in input_pairs], dim=0).bool()

        if device == "cuda":
            input_tensor, pad_mask = input_tensor.cuda(), pad_mask.cuda()

        _, _, logit = model(input_tensor, pad_mask)
        prob = logit.softmax(dim=1)
        max_prob, pred = prob.max(dim=1)
        pred_sym = [list(map(convert.decode, p.tolist())) for p in pred.cpu().numpy()]
        pred_txt = [post_process("".join(p)) for p in pred_sym]

        confidence = torch.cumprod(max_prob, dim=1)[-1].cpu().numpy().tolist()

        output = list(zip(pred_txt, confidence))
        return output


def read_with_paddle(image_list, key_list):
    field_ocr_paddle = ["id", "unit", "total", "female", "issue_date", "lane", "number"]

    # data_ocr_model = []
    # data_ocr_paddle = []
    # for key, img in zip(key_list, image_list):
    #     if key in field_ocr_paddle:
    #         data_ocr_paddle.append([img, key])
    #     else:
    #         data_ocr_model.append([img, key])

    with torch.no_grad():
        input_pairs = [data_trans(image) for image in image_list]
        input_tensor = torch.stack([i[0] for i in input_pairs], dim=0)
        pad_mask = torch.stack([i[1] for i in input_pairs], dim=0).bool()

        if device == "cuda":
            input_tensor, pad_mask = input_tensor.cuda(), pad_mask.cuda()

        _, _, logit = model(input_tensor, pad_mask)
        prob = logit.softmax(dim=1)
        max_prob, pred = prob.max(dim=1)
        pred_sym = [list(map(convert.decode, p.tolist())) for p in pred.cpu().numpy()]
        pred_txt = [post_process("".join(p)) for p in pred_sym]

        confidence = torch.cumprod(max_prob, dim=1)[-1].cpu().numpy().tolist()

        output_model = list(zip(pred_txt, confidence, key_list, image_list))

    output = []

    for txt, conf, key, img in output_model:
        if key in field_ocr_paddle:
            res = paddleReader.read(np.array(img))[0]
            txt_paddle, conf_paddle = res[0][0], res[0][1]
            output.append([txt_paddle, conf_paddle])
        else:
            output.append([txt, conf])

    return output


# parser = argparse.ArgumentParser()
# parser.add_argument('--batch_max_length', type=int, default=15, help='maximum-label-length')
# parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
# parser.add_argument('--imgW', type=int, default=192, help='the width of the input image')
# parser.add_argument('--onnx_model_path', type=str, default='./models/reader/v7.0.1_model.onnx', help='path to ONNX model for inference')
# opt = parser.parse_args()


# data_trans = DataTransformer(opt, 'valid')
# convert = Converter()
# ort_session = onnxruntime.InferenceSession(opt.onnx_model_path,  providers=['CUDAExecutionProvider'])


# def read(image_list, dictionary_list=None):
#     input_pairs = [data_trans(image) for image in image_list]

#     output = []
#     for img, pad_mask in input_pairs:
#         img = img.unsqueeze(dim=0).cpu().numpy()
#         pad_mask = pad_mask.unsqueeze(dim=0).cpu().numpy().astype('bool')

#         ort_inputs = {ort_session.get_inputs()[0].name: img, ort_session.get_inputs()[1].name: pad_mask}
#         ort_outputs = ort_session.run(None, ort_inputs)

#         _, _, logit = ort_outputs
#         prob = scipy.special.softmax(logit, axis=1)
#         pred = np.argmax(prob, axis=1)
#         max_prob = np.max(prob, axis=1)
#         pred_txt = [''.join(list(map(convert.decode, p.tolist()))) for p in pred]
#         confident_score = np.cumprod(max_prob, axis=1)[:, -1].tolist()
#         output += list(zip(pred_txt, torch.as_tensor(confident_score)))

#     return output
