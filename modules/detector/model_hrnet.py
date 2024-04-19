from modules.hrnet.modules.hrnet import hrnet18
import logging
from modules.hrnet.dataset import *
import numpy as np
from config import config
from utils.model_utils.loader import torch_load_content

class Model(torch.nn.Module):
    def __init__(self, num_cls=12):
        super(Model, self).__init__()
        # self.categories = categories
        self.num_cls = num_cls
        self.fpn = hrnet18(pretrained=True)
        self.head = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=270, out_channels=270, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(num_features=270),
            torch.nn.GELU(),
            torch.nn.Conv2d(in_channels=270, out_channels=2 * self.num_cls, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        fused_feat = self.fpn(x)
        out_head = self.head(fused_feat)
        P, T = out_head[:, :self.num_cls, :, :], out_head[:, self.num_cls:, :, :]
        B = 1 / (1 + torch.exp(-50 * (P - T)))

        if self.training:
            return P, T, B
        else:
            return P, T


class Opt():
    def __init__(self):
        self.root_data = "/content/drive/MyDrive/household-laos/lmdb_dataset"
        self.imgH = 700
        self.imgW = 1000


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def area_calc(bbox, imgW, imgH):
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    area = (x2 - x1) * (y2 - y1)
    if area / (imgW * imgH) < 2e-3:
        return False
    return True


class HRModel:
    def __init__(self, model_weight_path):
        self.categories = ["number", "street", "unit", "block", "ward", "district", "province", "total", "male",
                           "female", "issue_date", "name"]

        self.model_weight_path = model_weight_path
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.nn.DataParallel(Model())
        self.use_gpu = config.DEVICE != "cpu"
        if self.use_gpu:
            logging.info('Device name: %s' % torch.cuda.get_device_name(torch.cuda.current_device()))

        if self.use_gpu:
            self.model.cuda()
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True

        if self.use_gpu:
            map_location = lambda storage, loc: storage.cuda()
        else:
            map_location = 'cpu'
        # ckpt = torch.load(self.model_weight_path, map_location=map_location)
        ckpt = torch.load(torch_load_content(self.model_weight_path), map_location=map_location)

        self.half = self.use_gpu

        self.model.load_state_dict(ckpt['state_dict'], strict=False)
        if self.half:
            self.model.half()

    def predict_image_data(self, image):
        self.model.eval()
        logging.info('Inference...')
        opt = Opt()
        width, height = image.shape[1], image.shape[0]
        image_ = cv2.resize(image, (914, 640))
        # image = cv2.fastNlMeansDenoisingColored(image_, None, 10, 10, 7, 21)
        sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        image = cv2.filter2D(image_, -1, sharpen_kernel)
        image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

        color_coverted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(color_coverted)
        dt = DataTransformer(opt)
        image = dt(image)
        image = image.unsqueeze(0)
        image = image.half() if self.half else image.float()

        with torch.no_grad():
            if self.use_gpu:
                image = image.cuda()
            # print(image.shape)
            P, T = self.model(image)
            binary_map = P >= T

            # Post-processing
            image = unnormalize(image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            image = image[0].permute(1, 2, 0).cpu().numpy()
            image = (image * 255).astype('uint8').copy()

            P = P[0].sum(dim=0)
            P = normalize(P.flatten(), dim=0).reshape(P.size())
            P = P.cpu().numpy()

            post_binary_map = [m for m in binary_map[0].cpu().numpy().astype('uint8')]
            pred_bboxes = multi_apply(post_process, post_binary_map)
            # print(pred_bboxes)

            pred_bboxes_new = []
            for bboxes in pred_bboxes:
                if len(bboxes) == 0:
                    pred_bboxes_new.append([])
                else:
                    tmp = []
                    for bbox in bboxes:
                        if area_calc(bbox, 914, 640):
                            tmp.append([int(i) for i in
                                        [bbox[0] / 914 * width, bbox[1] / 640 * height, bbox[2] / 914 * width,
                                         bbox[3] / 640 * height]])
                    pred_bboxes_new.append(tmp)

            map = dict(zip(categories, pred_bboxes_new))
        return map



