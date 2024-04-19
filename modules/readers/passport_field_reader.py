import argparse
import string

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data

from config import config
from modules.recognition.dataset import AlignCollate
from modules.recognition.model import Model
from modules.recognition.utils import CTCLabelConverter, AttnLabelConverter
from utils.model_utils.loader import torch_load_content

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--workers", type=int, help="number of data loading workers", default=2
)
parser.add_argument("--batch_size", type=int, default=8, help="input batch size")
# parser.add_argument('--character', type=str,
#                     default='ပဃဏယကဂရတဒဉသခစအဆမလထညဟဇဘငဓဗဖဝန0123456789()ncpavm/',
#                     help='character label')
# parser.add_argument('--saved_model', help="path to saved_model to evaluation",
#                     default="models/reader/ID_None_ResNet_BiLSTM_Attn_best_accuracy_94.688.pth")

parser.add_argument(
    "--character",
    type=str,
    default="abcdefghijklmnopqrstuvwxyz 0123456789()/.,+:-",
    help="character label",
)
parser.add_argument(
    "--saved_model",
    help="path to saved_model to evaluation",
    default=config.MODEL_READER_FIELD_PASSPORT,
)
""" Data processing """
parser.add_argument(
    "--batch_max_length", type=int, default=20, help="maximum-label-length"
)
parser.add_argument(
    "--imgH", type=int, default=32, help="the height of the input image"
)
parser.add_argument(
    "--imgW", type=int, default=512, help="the width of the input image"
)
parser.add_argument("--rgb", action="store_true", help="use rgb input", default=False)

parser.add_argument(
    "--sensitive",
    action="store_true",
    help="for sensitive character mode",
    default=False,
)
parser.add_argument(
    "--PAD",
    action="store_true",
    help="whether to keep ratio then pad for image resize",
    default=True,
)
""" Model Architecture """
parser.add_argument(
    "--Transformation", type=str, help="Transformation stage. None|TPS", default="None"
)
parser.add_argument(
    "--FeatureExtraction",
    type=str,
    help="FeatureExtraction stage. VGG|RCNN|ResNet",
    default="ResNet",
)
parser.add_argument(
    "--SequenceModeling",
    type=str,
    help="SequenceModeling stage. None|BiLSTM",
    default="None",
)
parser.add_argument(
    "--Prediction", type=str, help="Prediction stage. CTC|Attn", default="Attn"
)
parser.add_argument(
    "--num_fiducial", type=int, default=20, help="number of fiducial points of TPS-STN"
)
parser.add_argument(
    "--input_channel",
    type=int,
    default=1,
    help="the number of input channel of Feature extractor",
)
parser.add_argument(
    "--output_channel",
    type=int,
    default=512,
    help="the number of output channel of Feature extractor",
)
parser.add_argument(
    "--hidden_size", type=int, default=256, help="the size of the LSTM hidden state"
)

opt = parser.parse_args()

""" vocab / character number configuration """
if opt.sensitive:
    opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

cudnn.benchmark = True
cudnn.deterministic = True
opt.num_gpu = torch.cuda.device_count()

""" model configuration """
if "CTC" in opt.Prediction:
    converter = CTCLabelConverter(opt.character)
else:
    converter = AttnLabelConverter(opt.character)
opt.num_class = len(converter.character)

if opt.rgb:
    opt.input_channel = 3
model = Model(opt)
print(
    "model input parameters",
    opt.imgH,
    opt.imgW,
    opt.num_fiducial,
    opt.input_channel,
    opt.output_channel,
    opt.hidden_size,
    opt.num_class,
    opt.batch_max_length,
    opt.Transformation,
    opt.FeatureExtraction,
    opt.SequenceModeling,
    opt.Prediction,
)
model = torch.nn.DataParallel(model).to(device)

# load model
print("loading pretrained model from %s" % opt.saved_model)
# model.load_state_dict(torch.load(opt.saved_model, map_location=device))
model.load_state_dict(torch.load(torch_load_content(opt.saved_model), map_location=device))

# prepare data. two demo images from https://github.com/bgshih/crnn#run-demo
AlignCollate_demo = AlignCollate(
    imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD
)


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
        length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(
            device
        )
        text_for_pred = (
            torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)
        )

        if "CTC" in opt.Prediction:
            preds = model(image, text_for_pred)

            # Select max probabilty (greedy decoding) then decode index to character
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            _, preds_index = preds.max(2)
            # preds_index = preds_index.view(-1)
            preds_str = converter.decode(preds_index, preds_size)

        else:
            # print(image.shape)
            preds = model(image, text_for_pred, is_train=False)

            for index, item in enumerate(preds):
                dictionary_tensor = dictionary_list[index]
                if dictionary_tensor is not None:
                    # print(item.type)
                    preds[index] = item + dictionary_tensor

            # print(preds.shape)
            # print(preds)
            # select max probabilty (greedy decoding) then decode index to character
            _, preds_index = preds.max(2)

            # print(preds.topk(5))
            preds_str = converter.decode(preds_index, length_for_pred)

        dashed_line = "-" * 80
        head = f'{"image_path":25s}\t{"predicted_labels":25s}\tconfidence score'

        # print(f'{dashed_line}\n{head}\n{dashed_line}')

        # print(preds)
        preds_prob = F.softmax(preds, dim=2)
        # print(preds_prob.shape)
        # print(preds_prob)
        preds_max_prob, _ = preds_prob.max(dim=2)
        # print(preds_max_prob.shape)
        # print(preds_str)
        # print(preds_max_prob)

        for img_name, pred, pred_max_prob in zip(
            image_path_list, preds_str, preds_max_prob
        ):
            # pred = ""
            # confidence_score = 0
            # print(pred)
            if "Attn" in opt.Prediction:
                pred_EOS = pred.find("[s]")
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

            output.append((pred, confidence_score))
        return output


prediction_dictionary_map = {}


def get_prediction_dictionary(label="type"):
    dictionary_tensor = None

    if label not in prediction_dictionary_map:
        if label == "type":
            dictionary_tensor = (
                torch.FloatTensor(opt.batch_max_length + 1, opt.num_class)
                .fill_(-10)
                .to(device)
            )

            text, length = converter.encode(["p", "po"], 30)
            # print(converter.decode(text, length))
            # print(text)
            # print(text.shape)
            for i in range(1, opt.batch_max_length + 1):
                # print(_char_to_onehot(text[:, :-1][:, i], opt.num_class).max(dim=0)[0])
                dictionary_tensor[i - 1] = _char_to_onehot(
                    text[:, :-1][:, i], opt.num_class
                ).max(dim=0)[0]
            prediction_dictionary_map[label] = dictionary_tensor
            # print(dictionary_tensor.shape)
            # print(dictionary_tensor)
        # elif label == "state":
        #     dictionary_tensor = torch.FloatTensor(opt.batch_max_length + 1, opt.num_class).fill_(-10).to(device)

        #     text, length = converter.encode(["lao"], 30)
        #     # print(converter.decode(text, length))
        #     # print(text)
        #     # print(text.shape)
        #     for i in range(1, opt.batch_max_length + 1):
        #         # print(_char_to_onehot(text[:, :-1][:, i], opt.num_class).max(dim=0)[0])
        #         dictionary_tensor[i - 1] = _char_to_onehot(text[:, :-1][:, i], opt.num_class).max(dim=0)[0]
        #     prediction_dictionary_map[label] = dictionary_tensor
        # elif label == "authority":
        #     dictionary_tensor = torch.FloatTensor(opt.batch_max_length + 1, opt.num_class).fill_(-20).to(device)

        #     text, length = converter.encode(["MOFA. LAO PDR".lower(), "MOFA LAO PDR".lower()], 30)
        #     # print(converter.decode(text, length))
        #     # print(text)
        #     # print(text.shape)
        #     for i in range(1, opt.batch_max_length + 1):
        #         # print(_char_to_onehot(text[:, :-1][:, i], opt.num_class).max(dim=0)[0])
        #         dictionary_tensor[i - 1] = _char_to_onehot(text[:, :-1][:, i], opt.num_class).max(dim=0)[0]
        #     prediction_dictionary_map[label] = dictionary_tensor
        # elif label == "gender":
        #     dictionary_tensor = torch.FloatTensor(opt.batch_max_length + 1, opt.num_class).fill_(-10).to(device)

        #     text, length = converter.encode(["m", "f"], 30)
        #     # print(converter.decode(text, length))
        #     # print(text)
        #     # print(text.shape)
        #     for i in range(1, opt.batch_max_length + 1):
        #         # print(_char_to_onehot(text[:, :-1][:, i], opt.num_class).max(dim=0)[0])
        #         dictionary_tensor[i - 1] = _char_to_onehot(text[:, :-1][:, i], opt.num_class).max(dim=0)[0]
        #     prediction_dictionary_map[label] = dictionary_tensor
        # elif label == "nationality":
        #     dictionary_tensor = torch.FloatTensor(opt.batch_max_length + 1, opt.num_class).fill_(-10).to(device)

        #     text, length = converter.encode(["lao"], 30)
        #     # print(converter.decode(text, length))
        #     # print(text)
        #     # print(text.shape)
        #     for i in range(1, opt.batch_max_length + 1):
        #         # print(_char_to_onehot(text[:, :-1][:, i], opt.num_class).max(dim=0)[0])
        #         dictionary_tensor[i - 1] = _char_to_onehot(text[:, :-1][:, i], opt.num_class).max(dim=0)[0]
        #     prediction_dictionary_map[label] = dictionary_tensor
        # elif label == "issue_date" or label == "expire_date" or label == "dob":
        #     date_list = []
        #     months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

        #     for d in range(1, 32):
        #         for m in range(1, 13):
        #             for y in range(1900, 2040):
        #                 date_str_type_I = "%02d / %02d / %04d" % (d, m, y)
        #                 date_str_type_II = "%02d %s %04d" % (d, months[m - 1], y)
        #                 date_list.append(date_str_type_I.lower())
        #                 date_list.append(date_str_type_II.lower())

        #     dictionary_tensor = torch.FloatTensor(opt.batch_max_length + 1, opt.num_class).fill_(-20).to(device)
        #     text, length = converter.encode(date_list, 30)
        #     # print(converter.decode(text, length))
        #     # print(text)
        #     # print(text.shape)
        #     for i in range(1, opt.batch_max_length + 1):
        #         # print(_char_to_onehot(text[:, :-1][:, i], opt.num_class).max(dim=0)[0])
        #         dictionary_tensor[i - 1] = _char_to_onehot(text[:, :-1][:, i], opt.num_class).max(dim=0)[0]
        #     prediction_dictionary_map[label] = dictionary_tensor
        # elif label == "id":
        #     # for d in range(0, 1000000):
        #     #     date_str_type_I = "%p %s" % (str(d))
        #     #     date_str_type_II = "%02d %s %04d" % (d, months[m-1], y)
        #     #     date_list.append(date_str_type_I.lower())
        #     #     date_list.append(date_str_type_II.lower())

        #     dictionary_tensor = torch.FloatTensor(opt.batch_max_length + 1, opt.num_class).fill_(-10).to(device)
        #     dictionary_tensor[0, 17] = 0
        #     dictionary_tensor[1, 2] = 0
        #     dictionary_tensor[1, 28] = 0

        #     for i in range(2, 15):
        #         for j in [28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 1]:
        #             dictionary_tensor[i, j] = 0
        # dictionary_tensor[:, 37] = -2
        # text, length = converter.encode(date_list, 30)
        # # print(converter.decode(text, length))
        # # print(text)
        # # print(text.shape)
        # for i in range(1, opt.batch_max_length + 1):
        #     # print(_char_to_onehot(text[:, :-1][:, i], opt.num_class).max(dim=0)[0])
        #     dictionary_tensor[i - 1] = _char_to_onehot(text[:, :-1][:, i], opt.num_class).max(dim=0)[0]
        # prediction_dictionary_map[label] = dictionary_tensor
        # elif label == "sur_name" or label == "given_name" or label == "place_of_birth":
        #     dictionary_tensor = torch.FloatTensor(opt.batch_max_length + 1, opt.num_class).fill_(-10).to(device)
        #     for i in [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
        #               1]:
        #         dictionary_tensor[:, i] = 0
        #     dictionary_tensor[:, 18] = -3
        # text, length = converter.encode(date_list, 30)
        # # print(converter.decode(text, length))
        # # print(text)
        # # print(text.shape)
        # for i in range(1, opt.batch_max_length + 1):
        #     # print(_char_to_onehot(text[:, :-1][:, i], opt.num_class).max(dim=0)[0])
        #     dictionary_tensor[i - 1] = _char_to_onehot(text[:, :-1][:, i], opt.num_class).max(dim=0)[0]
        # prediction_dictionary_map[label] = dictionary_tensor
        # print(dictionary_tensor.shape)
        # print(dictionary_tensor)
    else:
        dictionary_tensor = prediction_dictionary_map[label]

    return dictionary_tensor


def _char_to_onehot(input_char, onehot_dim=38):
    input_char = input_char.unsqueeze(1)
    batch_size = input_char.size(0)
    one_hot = torch.FloatTensor(batch_size, onehot_dim).fill_(-20).to(device)
    # print(one_hot.shape)
    one_hot = one_hot.scatter_(1, input_char, 0)
    return one_hot


if __name__ == "__main__":
    # from PIL import Image
    #
    # img_list = []
    # img_1 = Image.open("/data/work/project/open_source/iagcwd/input/id.jpg").convert('L')
    # img_2 = Image.open("/data/work/project/open_source/iagcwd/output/id.jpg").convert('L')
    # img_list.append(img_1)
    # img_list.append(img_2)
    # img_list.append(img)
    # # img1 = Image.open("/home/hulk/Downloads/image (58).png").convert('L')
    #
    # # for i in range(53, 59):
    # #     img = Image.open("/home/hulk/Downloads/image (" + str(i) + ").png").convert('L')
    # #     img_list.append(img)
    # output = read(img_list, [get_prediction_dictionary("id"), get_prediction_dictionary("id")])
    # print(output)

    # print(converter.encode(["0123456789 "], 30))
    # print(converter.encode(["p ", "pa "], 30))
    print(converter.encode(["q"], 30))

    # if i
