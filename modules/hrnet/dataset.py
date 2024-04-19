import lmdb
import os
import glob, sys
import lz4framed
import cv2
import tqdm
import ast
from PIL import Image, ImageOps, ImageDraw
import six
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import torchvision.transforms as transforms
from .utils import *
from .augmentation import RandAugment

# with open("./raw_data/categories", "r") as f:
#     categories = f.readlines()

categories = ["number", "street", "unit", "block", "ward", "district", "province", "total", "male",
                          "female", "issue_date", "name"]
def load_categories():
    s2i = {sbj: idx for idx, sbj in enumerate(categories)}
    i2s = {idx: sbj for idx, sbj in enumerate(categories)}
    return s2i, i2s

c2i, i2c = load_categories()


class LmdbDataset(Dataset):
    def __init__(self, root, opt, mode):
        self.transforms = DataTransformer(opt, mode=mode)
        self.env = lmdb.open(root, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
        if not self.env:
            print('cannot create lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            self.nSamples = int(txn.get('num-samples'.encode()))
            # print(self.nSamples)
            self.filtered_index_list = []
            for index in tqdm.tqdm(range(self.nSamples), desc='Preparing'):
                index += 1
                gt_key = 'label-%09d'.encode() % index
                gt = ast.literal_eval(txn.get(gt_key).decode())

                # if len(gt.keys()) == len(categories):
                #     self.filtered_index_list.append(index)
            # print(self.filtered_index_list)
            # self.nSamples = len(self.filtered_index_list)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        if index == len(self):
            raise IndexError
        # index = self.filtered_index_list[index]
        tmp = [i for i in range(1, self.nSamples+1)]
        index = tmp[index]
        with self.env.begin(write=False) as txn:
            img_key = 'image-%09d'.encode() % (index)
            gt_key = 'label-%09d'.encode() % (index)

            imgbuf = txn.get(img_key)
            gt = ast.literal_eval(txn.get(gt_key).decode())

            buf = six.BytesIO()
            buf.write(lz4framed.decompress(imgbuf))
            buf.seek(0)
            img = Image.open(buf).convert('RGB')
            img = ImageOps.exif_transpose(img)
            img, gt_masks = self.transforms(img, gt)
        return img, gt_masks, gt


class Resize:
    def __init__(self, nsize):
        self.nsize = nsize

    def _resize_mask(self, mask, new_size):
        foreground = cv2.resize(mask, new_size)
        background = np.zeros((self.nsize[1], self.nsize[0]))
        background[: foreground.shape[0], : foreground.shape[1]] = foreground
        return background

    def __call__(self, image, mask=None):
        factor_x = image.width / self.nsize[0]
        factor_y = image.height / self.nsize[1]
        factor = max(factor_x, factor_y)
        new_size = (min(self.nsize[0], int(image.width / factor)), min(self.nsize[1], int(image.height / factor)))
        image = image.resize(size=new_size)
        new_image = Image.new('RGB', self.nsize, color=(0, 0, 0))
        new_image.paste(image, (0, (self.nsize[1] - new_size[1]) // 2))

        if mask is not None:
            result = multi_apply(self._resize_mask, [m for m in mask], [new_size] * mask.shape[0])
            mask = np.stack(result, axis=0)
            return new_image, mask
        else:
            return new_image


class DataTransformer:
    def __init__(self, opt, mode='train'):
        self.opt = opt
        self.mode = mode
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        self.randaug = RandAugment(3)
        self.resize = Resize((opt.imgW, opt.imgH))
        self.cate_sort = ["number", "street", "unit", "block", "ward", "district", "province", "total", "male",
                          "female", "issue_date", "name"]
    def _draw_bounding_mask(self, bboxes):
        mask = Image.new('L', (self.opt.imgW, self.opt.imgH), color=0)
        drawer = ImageDraw.Draw(mask)
        for bbox in bboxes:
            bbox = [int(float(i)) for i in bbox]  # convert to integer
            drawer.rectangle(bbox, fill=1)
        return np.array(mask)

    def __call__(self, image, gt=None):
        if gt is not None:
            # print(gt)
            if len(gt.keys()) != len(self.cate_sort):
                for i in self.cate_sort:
                    if i not in gt.keys():
                        gt[i] = []
            # print(gt)
            coordinates = []
            for i in self.cate_sort:
                coordinates.append(gt[i])
            # print(coordinates)
            gt_masks = multi_apply(self._draw_bounding_mask, coordinates)
            gt_masks = np.stack(gt_masks, axis=0)

            # if self.mode == 'train':
            #     image, gt_masks = self.randaug(image, gt_masks)

            image, gt_masks = self.resize(image, gt_masks)
            image = self.transform(image)
            gt_masks = torch.as_tensor(gt_masks).bool().float()

            return image, gt_masks[:, ::4, ::4]

        else:
            image = self.resize(image)
            image = self.transform(image)
            return image


def collate_fn(batch):
    image = torch.stack([sample[0] for sample in batch], dim=0)
    gt_masks = torch.stack([sample[1] for sample in batch], dim=0)
    gt = [sample[2] for sample in batch]
    return image, gt_masks, gt

if __name__ == '__main__':
    import argparse
    import matplotlib.pyplot as plt
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_data', required=True, help='path to root dataset')
    parser.add_argument('--imgH', type=int, default=700, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=1000, help='the width of the input image')
    opt = parser.parse_args()

    dataset = LmdbDataset(opt.root_data, opt, 'train')
    # for image, gt_masks, _ in dataset:
    #     from utils import unnormalize
    #
    #     image = unnormalize(image.unsqueeze(dim=0), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    #     image = image[0].permute(1, 2, 0).cpu().numpy()
    #     image = (image * 255).astype('uint8')
    #
    #     gt_masks = gt_masks.cpu().numpy()
    #     print(gt_masks.shape)
    #     fig, axs = plt.subplots(2, 1)
    #     axs[0].imshow(image)
    #     axs[0].axis('off')
    #     axs[1].imshow(gt_masks.sum(axis=0))
    #     axs[1].axis('off')
    #     fig.tight_layout()
    #     plt.show()
    #     plt.close('all')
    #     break
