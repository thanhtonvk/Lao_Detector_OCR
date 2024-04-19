import sys, lmdb, six
from PIL import Image
import torch
import torchvision.transforms as transforms
import modules.recognition.printed.lao_parser as lao_parser
from torch.utils.data import Dataset
from modules.recognition.printed.utils import post_process
from modules.recognition.printed.augmentation import RandAugment, augment_list



symbols = ['<PAD>', ' ', '-', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'ກ', 'ກັ', 'ກັ່', 'ກິ', 'ກິ່', 'ກີ', 'ກີຼ', 'ກີ່', 'ກຶ', 'ກື', 'ກື້', 'ກຸ', 'ກຸ່', 'ກຸ້', 'ກູ', 'ກູ່', 'ກູ້', 'ກົ', 'ກົ່', 'ກົ້', 'ກ່', 'ກ່ໍ', 'ກ້', 'ກ໊', 'ກ໋', 'ກໍ', 'ຂ', 'ຂັ', 'ຂິ', 'ຂີ', 'ຂີ້', 'ຂື', 'ຂື່', 'ຂຸ', 'ຂົ', 'ຂົ້', 'ຂ່', 'ຂ້', 'ຂ້ໍ', 'ຂໍ', 'ຄ', 'ຄັ', 'ຄິ', 'ຄີ', 'ຄຶ', 'ຄື', 'ຄື່', 'ຄື້', 'ຄຸ', 'ຄຸ້', 'ຄູ', 'ຄູ່', 'ຄູ້', 'ຄົ', 'ຄົ້', 'ຄ່', 'ຄ້', 'ຄ້ໍ', 'ຄໍ', 'ງ', 'ງິ', 'ງິ້', 'ງີ', 'ງື່', 'ງູ', 'ງົ', 'ງົ້', 'ງ່', 'ງ້', 'ຈ', 'ຈັ', 'ຈັ່', 'ຈັ້', 'ຈິ', 'ຈິ່', 'ຈີ', 'ຈຶ', 'ຈຶ່', 'ຈື', 'ຈື່', 'ຈື້', 'ຈຸ', 'ຈຸ່', 'ຈູ', 'ຈູ່', 'ຈົ', 'ຈົ່', 'ຈົ້', 'ຈ່', 'ຈ້', 'ຈ໋', 'ຈ໌', 'ຈໍ', 'ຊ', 'ຊັ', 'ຊັ່', 'ຊັ້', 'ຊິ', 'ຊີ', 'ຊີ່', 'ຊີ້', 'ຊຶ', 'ຊື', 'ຊື່', 'ຊື້', 'ຊຸ', 'ຊູ', 'ຊົ', 'ຊົ່', 'ຊົ້', 'ຊ່', 'ຊ້', 'ຊ້ໍ', 'ຊໍ', 'ຍ', 'ຍັ', 'ຍິ', 'ຍິ່', 'ຍີ', 'ຍີ່', 'ຍື', 'ຍຸ', 'ຍູ', 'ຍູ້', 'ຍົ', 'ຍ່', 'ຍ້', 'ຍ້ໍ', 'ຍໍ', 'ດ', 'ດັ', 'ດັ່', 'ດັ້', 'ດິ', 'ດີ', 'ດີ່', 'ດີ້', 'ດື', 'ດື່', 'ດື້', 'ດຸ', 'ດູ', 'ດູ່', 'ດົ', 'ດ່', 'ດ້', 'ດໍ', 'ຕ', 'ຕັ', 'ຕັ່', 'ຕັ້', 'ຕິ', 'ຕິ່', 'ຕິ້', 'ຕີ', 'ຕີຼ', 'ຕີ່', 'ຕີ້', 'ຕື່', 'ຕື້', 'ຕຸ', 'ຕຸ່', 'ຕຸ້', 'ຕູ', 'ຕູ່', 'ຕູ້', 'ຕົ', 'ຕົ່', 'ຕົ້', 'ຕ່', 'ຕ່ໍ', 'ຕ້', 'ຕ້ໍ', 'ຕ໋', 'ຕໍ', 'ຖ', 'ຖັ', 'ຖິ', 'ຖິ່', 'ຖີ', 'ຖີ່', 'ຖື', 'ຖົ', 'ຖົ້', 'ຖ່', 'ຖ່ໍ', 'ຖ້', 'ຖ້ໍ', 'ຖໍ', 'ທ', 'ທັ', 'ທິ', 'ທີ', 'ທີ່', 'ທຶ', 'ທື', 'ທື່', 'ທຸ', 'ທຸ່', 'ທູ', 'ທູ້', 'ທົ', 'ທົ່', 'ທ່', 'ທ່ໍ', 'ທ້', 'ທ້ໍ', 'ທໍ', 'ນ', 'ນັ', 'ນັ່', 'ນັ້', 'ນິ', 'ນິ້', 'ນີ', 'ນີ່', 'ນີ້', 'ນຶ', 'ນຶ່', 'ນຶ້', 'ນື', 'ນື່', 'ນຸ', 'ນຸ່', 'ນຸ້', 'ນູ', 'ນູ່', 'ນູ້', 'ນົ', 'ນ່', 'ນ່ໍ', 'ນ້', 'ນ້ໍ', 'ນໍ', 'ບ', 'ບັ', 'ບັ້', 'ບິ', 'ບີ', 'ບີ້', 'ບຶ', 'ບຸ', 'ບຸ່', 'ບູ', 'ບົ', 'ບົ້', 'ບ່', 'ບ່ໍ', 'ບ້', 'ບໍ', 'ປ', 'ປັ', 'ປິ', 'ປິ່', 'ປີ', 'ປີ້', 'ປຶ', 'ປື', 'ປຸ', 'ປຸ່', 'ປຸ້', 'ປູ', 'ປູ່', 'ປູ້', 'ປົ', 'ປົ່', 'ປົ້', 'ປຼ', 'ປ່', 'ປ້', 'ປ້ໍ', 'ປໍ', 'ຜ', 'ຜັ', 'ຜິ', 'ຜິ້', 'ຜີ', 'ຜື', 'ຜຸ', 'ຜູ', 'ຜູ້', 'ຜົ', 'ຜົ່', 'ຜົ້', 'ຜ່', 'ຝ', 'ຝັ່', 'ຝັ້', 'ຝີ', 'ຝົ', 'ພ', 'ພັ', 'ພັ່', 'ພິ', 'ພີ', 'ພຶ', 'ພື່', 'ພຸ', 'ພູ', 'ພົ', 'ພົ້', 'ພ່', 'ພ່ໍ', 'ພ້', 'ພ້ໍ', 'ພໍ', 'ຟ', 'ຟິ', 'ຟີ', 'ຟື', 'ຟູ', 'ຟົ', 'ຟ້', 'ມ', 'ມັ', 'ມັ່', 'ມັ້', 'ມິ', 'ມິ່', 'ມີ', 'ມີ່', 'ມີ້', 'ມື', 'ມື່', 'ມື້', 'ມຸ', 'ມຸ່', 'ມຸ້', 'ມູ', 'ມົ', 'ມົ່', 'ມົ້', 'ມ່', 'ມ້', 'ມ້ໍ', 'ມໍ', 'ຢ', 'ຢັ', 'ຢັ່', 'ຢັ້', 'ຢິ່', 'ຢີ', 'ຢີ້', 'ຢຸ', 'ຢູ', 'ຢູ່', 'ຢົ', 'ຢົ່', 'ຢົ້', 'ຢ່', 'ຢ້', 'ຣ', 'ຣັ', 'ຣິ', 'ຣີ', 'ຣຸ', 'ຣູ', 'ຣົ', 'ຣ໌', 'ລ', 'ລັ', 'ລັ່', 'ລິ', 'ລິ່', 'ລິ້', 'ລີ', 'ລີ່', 'ລີ້', 'ລຶ', 'ລື', 'ລື່', 'ລື້', 'ລຸ', 'ລຸ່', 'ລຸ້', 'ລູ', 'ລົ', 'ລົ່', 'ລົ້', 'ລ່', 'ລ່ໍ', 'ລ້', 'ລ້ໍ', 'ລ໌', 'ລໍ', 'ວ', 'ວັ', 'ວິ', 'ວີ', 'ວື', 'ວື່', 'ວຸ', 'ວົ', 'ວ່', 'ວ້', 'ວໍ', 'ສ', 'ສັ', 'ສັ້', 'ສິ', 'ສິ່', 'ສີ', 'ສີ່', 'ສຶ', 'ສື', 'ສຸ', 'ສູ', 'ສູ່', 'ສູ້', 'ສົ', 'ສົ່', 'ສົ້', 'ສ່', 'ສ້', 'ສ້ໍ', 'ສ໌', 'ສໍ', 'ຫ', 'ຫັ', 'ຫັຼ', 'ຫັ້', 'ຫິ', 'ຫິຼ', 'ຫີ', 'ຫີຼ', 'ຫີຼ່', 'ຫີຼ້', 'ຫືຼ', 'ຫຸ', 'ຫູ', 'ຫົ', 'ຫົຼ', 'ຫົຼ່', 'ຫົຼ້', 'ຫຼ', 'ຫຼ່', 'ຫຼ່ໍ', 'ຫຼ້', 'ຫຼໍ', 'ຫ່', 'ຫ້', 'ຫ້ໍ', 'ຫໍ', 'ອ', 'ອັ', 'ອິ', 'ອິ່', 'ອີ', 'ອີ່', 'ອີ້', 'ອື', 'ອື້', 'ອຸ', 'ອຸ່', 'ອຸ້', 'ອູ', 'ອູ່', 'ອົ', 'ອົ້', 'ອ່', 'ອ້', 'ອ້ໍ', 'ອ໋', 'ອໍ', 'ຮ', 'ຮັ', 'ຮັ່', 'ຮິ', 'ຮີ', 'ຮີ່', 'ຮື', 'ຮຸ', 'ຮຸ່', 'ຮຸ້', 'ຮູ', 'ຮູ້', 'ຮົ', 'ຮົ່', 'ຮົ້', 'ຮ່', 'ຮ້', 'ຮໍ', 'ະ', 'າ', 'າ້', 'າ້ໍ', 'ຽ', 'ຽ່', 'ຽ້', 'ເ', 'ແ', 'ໂ', 'ໃ', 'ໄ'] 



def load_categories():
    s2i = {sbj: idx for idx, sbj in enumerate(symbols)}
    i2s = {idx: sbj for idx, sbj in enumerate(symbols)}
    return s2i, i2s


s2i, i2s = load_categories()

class Converter:
    def __init__(self):
        self.s2i, self.i2s = load_categories()

    def encode(self, s):
        return self.s2i[s]

    def decode(self, i):
        s = self.i2s[i]
        if s in ['<PAD>']:
            return ''
        else:
            return s
        

class LmdbDataset(Dataset):
    def __init__(self, root, set_name, opt):
        self.root = root
        self.opt = opt
        self.data_trans = DataTransformer(opt, set_name)
        self.env = lmdb.open(root, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
        if not self.env:
            print('cannot create lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode()))
            self.nSamples = nSamples

            self.filtered_index_list = []
            for index in range(self.nSamples):
                index += 1
                label_key = 'label-%09d'.encode() % index
                label = txn.get(label_key).decode('utf-8')
                label = label.replace("ຳ", "ໍາ").replace("ໝ", "ຫມ").replace("ໜ", "ຫນ").replace(',', '').replace('.', '')

                if lao_parser.is_valid(label) and label != '':
                    if len(lao_parser.parse_text(label)) > self.opt.batch_max_length:
                        continue
                    else:
                        self.filtered_index_list.append(index)

            self.nSamples = len(self.filtered_index_list)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index = self.filtered_index_list[index]

        with self.env.begin(write=False) as txn:
            label_key = 'label-%09d'.encode() % index
            label = txn.get(label_key).decode('utf-8')
            label = label.replace("ຳ", "ໍາ").replace("ໝ", "ຫມ").replace("ໜ", "ຫນ").replace(',', '').replace('.', '')
            label = lao_parser.parse_text(label)
            img_key = 'image-%09d'.encode() % index
            tag_key = 'tag-%09d'.encode() % index
            imgbuf = txn.get(img_key)
            tag = txn.get(tag_key)

            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:
                if self.opt.rgb:
                    img = Image.open(buf).convert('RGB')
                else:
                    img = Image.open(buf).convert('L')

            except IOError:
                print(f'Corrupted image for {index}')
                if self.opt.rgb:
                    img = Image.new('RGB', (self.opt.imgW, self.opt.imgH))
                else:
                    img = Image.new('L', (self.opt.imgW, self.opt.imgH))
                label = '[dummy_label]'

        image, pad_mask, target, label = self.data_trans(img, label)
        return image, pad_mask, target, label, img_key.decode(), tag.decode()


class Resize:
    def __init__(self, nsize):
        self.nsize = nsize
        
    def __call__(self, image):
        factor_x = image.width / self.nsize[0]
        factor_y = image.height / self.nsize[1]
        factor = max(factor_x, factor_y)
        new_size = (min(self.nsize[0], int(image.width / factor)), min(self.nsize[1], int(image.height / factor)))
        image = image.resize(size=new_size)
        new_image = Image.new('RGB', self.nsize, color=(0, 0, 0))
        new_image.paste(image, (0, (self.nsize[1] - new_size[1]) // 2))

        foreground = Image.new('L', new_size, color=0)
        background = Image.new('L', self.nsize, color=1)
        background.paste(foreground, (0, (self.nsize[1] - new_size[1]) // 2))

        return new_image, background


class DataTransformer:
    def __init__(self, opt, set_name):
        self.resize = Resize((opt.imgW, opt.imgH))
        self.opt = opt
        self.set_name = set_name
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=0.445, std=0.269)
        ])
        self.convert = Converter()
        self.randaug = RandAugment(3, 5, augment_list())

    def __call__(self, image, label=None):
        if self.set_name == 'train':
            image = image.convert('RGB')
            image, _ = self.randaug(image)

        image, pad_mask = self.resize(image)
        image = image.convert('L')
        image = self.transform(image)
        pad_mask = transforms.ToTensor()(pad_mask).bool()
        if label is not None:
            target = torch.as_tensor(list(map(self.convert.encode, label)) + [self.convert.encode('<PAD>')] * (self.opt.batch_max_length - len(label))).long()
            return image, pad_mask[:, ::4, ::4], target, post_process(''.join(label))
        else:
            return image, pad_mask[:, ::4, ::4]

def train_collate_fn(batch):
    image = torch.stack([sample[0] for sample in batch], dim=0)
    pad_mask = torch.stack([sample[1] for sample in batch], dim=0)
    target = torch.stack([sample[2] for sample in batch], dim=0)
    return image, pad_mask, target


def valid_collate_fn(batch):
    image = torch.stack([sample[0] for sample in batch], dim=0)
    pad_mask = torch.stack([sample[1] for sample in batch], dim=0)
    target = torch.stack([sample[2] for sample in batch], dim=0)
    label = [sample[3] for sample in batch]
    img_key = [sample[4] for sample in batch]
    tag = [sample[5] for sample in batch]
    return image, pad_mask, target, label, img_key, tag

