import argparse, os, time, sys, tqdm
import cv2
import logging
import torch
from dataset import LmdbDataset, collate_fn, categories
from modules.model import Model
from modules.loss import Criterion
from torch.utils.data import DataLoader
from utils import unnormalize, normalize, multi_apply, post_process
import matplotlib.pyplot as plt

logging.basicConfig(format='%(levelname)s - %(message)s', level=logging.INFO)
use_gpu = torch.cuda.is_available()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(opt):
    train_data = torch.utils.data.ConcatDataset([LmdbDataset(data_path, opt, 'train') for data_path in opt.train_data])
    # valid_data = LmdbDataset(opt.valid_data, opt, 'valid')

    train_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, drop_last=True,
                              num_workers=opt.workers, collate_fn=collate_fn, pin_memory=use_gpu)
    # valid_loader = DataLoader(valid_data, batch_size=1, shuffle=True, drop_last=False, num_workers=opt.workers,
    #                           collate_fn=collate_fn, pin_memory=use_gpu)

    model = torch.nn.DataParallel(Model(opt))
    optimizer = torch.optim.Adadelta(model.parameters(), lr=opt.lr)
    criterion = Criterion()

    logging.info('Number of parameters: %d' % count_parameters(model))
    if use_gpu:
        logging.info('Device name: %s' % torch.cuda.get_device_name(torch.cuda.current_device()))

    if use_gpu:
        model.cuda()
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        criterion = criterion.cuda()

    curr_epoch = 0
    if os.path.exists("./model/model5.pth"):
        if use_gpu:
            map_location = lambda storage, loc: storage.cuda()
        else:
            map_location = 'cpu'
        ckpt = torch.load(opt.saved_model, map_location=map_location)
        model.load_state_dict(ckpt['state_dict'], strict=False)
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        curr_epoch = ckpt['global_step']
        logging.info('Restored model is loaded')

    logging.info('Training...')

    for epoch in range(curr_epoch+1, opt.epochs+1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        for step, (image, gt_masks, gt) in enumerate(train_loader):
            start = time.time()
            if use_gpu:
                image, gt_masks = image.cuda(), gt_masks.cuda()

            P, _, B = model(image)
            loss_P, loss_B = criterion([P, B], [gt_masks, gt_masks])
            loss = loss_P + loss_B
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=opt.grad_clip, norm_type=2)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            end = time.time()
            sys.stdout.write('\rEpoch: %03d, Step: %04d/%d, Probability Loss: %.9f, Binary Loss: %.9f, Time training: %.2f secs' % (epoch, step+1, len(train_loader), loss_P.item(), loss_B.item(), end-start))

        torch.save({'state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'global_step': epoch}, "model/model5.pth")

    #     if epoch % opt.valInterval == 0 or epoch == opt.epochs:
    # print()
    # logging.info('Validating...')
    # model.eval()
    #
    # with torch.no_grad():
    #     for image, _, gt in tqdm.tqdm(valid_loader):
    #         if use_gpu:
    #             image = image.cuda()
    #
    #         P, T = model(image)
    #         binary_map = P >= T
    #
    #         # Post-processing
    #         image = unnormalize(image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    #         image = image[0].permute(1, 2, 0).cpu().numpy()
    #         image = (image * 255).astype('uint8').copy()
    #
    #         P = P[0].sum(dim=0)
    #         P = normalize(P.flatten(), dim=0).reshape(P.size())
    #         P = P.cpu().numpy()
    #
    #         post_binary_map = [m for m in binary_map[0].cpu().numpy().astype('uint8')]
    #         pred_bboxes = multi_apply(post_process, post_binary_map)
    #         for bboxes in pred_bboxes:
    #             for (x1, y1, x2, y2) in bboxes:
    #                 cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #
    #         binary_map = binary_map[0].sum(dim=0)
    #         binary_map = normalize(binary_map.flatten(), dim=0).reshape(binary_map.size())
    #         binary_map = binary_map.cpu().numpy()
    #
    #         fig, axs = plt.subplots(3, 1)
    #         axs[0].imshow(image)
    #         axs[0].axis('off')
    #         axs[0].set_title('Output')
    #         axs[1].imshow(P)
    #         axs[1].axis('off')
    #         axs[1].set_title('Probability map')
    #         axs[2].imshow(binary_map)
    #         axs[2].axis('off')
    #         axs[2].set_title('Adaptive threshold')
    #         fig.tight_layout()
    #         plt.show()
    #         # plt.savefig('result.svg', format='svg', dpi=1000)
    #         plt.close('all')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', nargs='+', required=True, help='path to training dataset')
    parser.add_argument('--valid_data', required=False, help='path to validation dataset')
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--batch_size', type=int, default=4, help='input batch size')
    parser.add_argument('--epochs', type=int, default=100, help='number of iterations to train for')
    parser.add_argument('--valInterval', type=int, default=4, help='Interval between each validation')
    parser.add_argument('--saved_model', default="./model/model5.pth", help="path to model to continue training")
    parser.add_argument('--lr', type=float, default=1., help='learning rate, default=1.0 for Adadelta')
    parser.add_argument('--grad_clip', type=float, default=2, help='gradient clipping value. default=2')
    parser.add_argument('--imgH', type=int, default=700, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=1000, help='the width of the input image')
    parser.add_argument('--k', type=int, default=50, help='the confidence k')

    opt = parser.parse_args()
    opt.categories = categories
    train(opt)