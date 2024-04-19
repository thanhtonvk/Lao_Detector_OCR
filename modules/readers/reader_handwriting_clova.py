import argparse
import string

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data

from modules.recognition.dataset import AlignCollate
from modules.recognition.model import Model
from modules.recognition.utils import CTCLabelConverter, AttnLaoLabelConverter
import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)
from config import config
from utils.model_utils.loader import torch_load_content
import numpy as np


import logging
logger = logging.getLogger(__name__)

device = torch.device('cuda' if config.DEVICE != 'cpu' else 'cpu')

logger.info(device)

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
# parser.add_argument('--saved_model', help="path to saved_model to evaluation",
#                     default="models/reader/name_char_list_43_None_ResNet_None_Attn_best_accuracy_83.pth")

# parser.add_argument('--saved_model', help="path to saved_model to evaluation",
#                     default="models/reader/handwriting_None_ResNet_BiLSTM_Attn_20220122.pth")
parser.add_argument('--saved_model', help="path to saved_model to evaluation", default=config.MODEL_HOUSEHOLD_NEXT_HANDWRITING_READER)

# parser.add_argument('--rgb', action='store_true', help='use rgb input', default=True)
parser.add_argument('--character', type=str,
                    default='်ငမေးာိုနအြကွသဝစလျဦ့ခထတဇဖူပရှါီဒယညဌဆဲဟံ္ဉဘ',
                    help='character label')
""" Data processing """
parser.add_argument('--batch_max_length', type=int, default=20, help='maximum-label-length')
parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
parser.add_argument('--imgW', type=int, default=320, help='the width of the input image')
parser.add_argument('--rgb', action='store_true', help='use rgb input')

parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize', default=True)
""" Model Architecture """
parser.add_argument('--Transformation', type=str, help='Transformation stage. None|TPS', default="None")
parser.add_argument('--FeatureExtraction', type=str, help='FeatureExtraction stage. VGG|RCNN|ResNet',
                    default="ResNet")
parser.add_argument('--SequenceModeling', type=str, help='SequenceModeling stage. None|BiLSTM', default="BiLSTM")
parser.add_argument('--Prediction', type=str, help='Prediction stage. CTC|Attn', default="Attn")
parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
parser.add_argument('--output_channel', type=int, default=512,
                    help='the number of output channel of Feature extractor')
parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')

opt = parser.parse_args()

print(opt.character)
print(len(opt.character))

""" vocab / character number configuration """
if opt.sensitive:
    opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

cudnn.benchmark = True
cudnn.deterministic = True
opt.num_gpu = torch.cuda.device_count()

""" model configuration """
if 'CTC' in opt.Prediction:
    converter = CTCLabelConverter(opt.character)
else:
    converter = AttnLaoLabelConverter()
opt.num_class = len(converter.character)

if opt.rgb:
    opt.input_channel = 3
model = Model(opt)
print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
      opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
      opt.SequenceModeling, opt.Prediction)
# model = torch.nn.DataParallel(model, device_ids=device, output_device=device).to(device)

# load model
print('loading pretrained model from %s' % opt.saved_model)
print(opt.saved_model)

# checkpoint = torch.load(opt.saved_model, map_location="cpu")
checkpoint = torch.load(torch_load_content(opt.saved_model), map_location="cpu")

for key in list(checkpoint.keys()):
    if 'module.' in key:
        checkpoint[key.replace('module.', '')] = checkpoint[key]
        del checkpoint[key]
model.load_state_dict(checkpoint)
model.to(device)

# model.load_state_dict(torch.load(opt.saved_model, map_location=device))

# prepare data. two demo images from https://github.com/bgshih/crnn#run-demo
AlignCollate_demo = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)


def read(image_list, dictionary_list):
    output = []

    image_list = [(image, "") for image in image_list]
    image_tensors, image_path_list = AlignCollate_demo(image_list)
    # def demo(image):
    #     image_tensors, image_path_list = AlignCollate_demo([(image, "")])
    # print(image_tensors_list)

    # predict
    model.eval()
    with torch.no_grad():
        # for image_tensors, image_path_list in image_tensors_list:
        # print(type(image_tensors))
        # print(image_tensors.shape)
        batch_size = image_tensors.size(0)
        image = image_tensors.to(device)
        # For max length prediction
        length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
        text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

        if 'CTC' in opt.Prediction:
            preds = model(image, text_for_pred)

            # Select max probabilty (greedy decoding) then decode index to character
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            _, preds_index = preds.max(2)
            # preds_index = preds_index.view(-1)
            preds_str = converter.decode(preds_index, preds_size)

        else:
            preds = model(image, text_for_pred, is_train=False)

            for index, item in enumerate(preds):
                dictionary_tensor = dictionary_list[index]
                if dictionary_tensor is not None:
                    # print(item.type)
                    preds[index] = item + dictionary_tensor

            # print(preds)
            # select max probabilty (greedy decoding) then decode index to character
            _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index, length_for_pred)



        dashed_line = '-' * 80
        head = f'{"image_path":25s}\t{"predicted_labels":25s}\tconfidence score'

        # print(f'{dashed_line}\n{head}\n{dashed_line}')

        # print(preds)

        preds_prob = F.softmax(preds, dim=2)
        sorted_indices = np.argsort(preds_prob.detach().cpu().numpy(), axis=2)

        preds_max_prob, _ = preds_prob.max(dim=2)

        for img_name, pred, pred_max_prob, sorted_idx in zip(image_path_list, preds_str, preds_max_prob, sorted_indices):
            # pred = ""
            # confidence_score = 0
            if 'Attn' in opt.Prediction:
                pred_EOS = pred.find('[s]')
                if pred_EOS == 0:
                    pred_EOS = pred[1:].find('[s]')
                pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                pred_max_prob = pred_max_prob[:pred_EOS]

            # calculate confidence score (= multiply of pred_max_prob)
            confidence_score = pred_max_prob.cumprod(dim=0)[-1]

            # print(f'{img_name:25s}\t{pred:25s}\t{confidence_score:0.4f}')
            #     log.write(f'{img_name:25s}\t{pred:25s}\t{confidence_score:0.4f}\n')
            #
            # log.close()
            # for img_name, pred, pred_max_prob in zip(image_path_list, preds_str, preds_max_prob):
            #     if 'Attn' in opt.Prediction:
            #         pred_EOS = pred.find('[s]')
            #         pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
            #         pred_max_prob = pred_max_prob[:pred_EOS]
            #
            #     # calculate confidence score (= multiply of pred_max_prob)
            #     confidence_score = pred_max_prob.cumprod(dim=0)[-1]
            #
            #     print(f'{img_name:25s}\t{pred:25s}\t{confidence_score:0.4f}')

            output.append((pred, confidence_score, sorted_idx))
        return output


prediction_dictionary_map = {}


def get_prediction_dictionary(label="type"):
    dictionary_tensor = None

    # if label not in prediction_dictionary_map:
    #     if label == "issue_date" or label == "birthday" or label == "expiry":
    #         date_list = []
    #
    #         for d in range(1, 32):
    #             for m in range(1, 13):
    #                 for y in range(1900, 2040):
    #                     date_str_type_I = "%02d/%02d/%04d" % (d, m, y)
    #                     if m < 10:
    #                         date_str_type_I = "%02d/%01d/%04d" % (d, m, y)
    #                     date_list.append(date_str_type_I.lower())
    #
    #         dictionary_tensor = torch.FloatTensor(opt.batch_max_length + 1, opt.num_class).fill_(-20).to(device)
    #         text, length = converter.encode(date_list, 30)
    #         # print(converter.decode(text, length))
    #         # print(text)
    #         # print(text.shape)
    #         for i in range(1, opt.batch_max_length + 1):
    #             # print(_char_to_onehot(text[:, :-1][:, i], opt.num_class).max(dim=0)[0])
    #             dictionary_tensor[i - 1] = _char_to_onehot(text[:, :-1][:, i], opt.num_class).max(dim=0)[0]
    #         prediction_dictionary_map[label] = dictionary_tensor
    # else:
    #     dictionary_tensor = prediction_dictionary_map[label]

    return dictionary_tensor


def _char_to_onehot(input_char, onehot_dim=38):
    input_char = input_char.unsqueeze(1)
    batch_size = input_char.size(0)
    one_hot = torch.FloatTensor(batch_size, onehot_dim).fill_(-20).to(device)
    # print(one_hot.shape)
    one_hot = one_hot.scatter_(1, input_char, 0)
    return one_hot

if __name__ == '__main__':
    from PIL import Image

    img_list = []

    for i in range(62, 64):
        img = Image.open("/home/daotuanvu/Downloads/image (" + str(i) + ").png").convert('L')
        img_list.append(img)

    output = read(img_list, [None, None, None])
    # print(output)

    for data in output:
        print(data)