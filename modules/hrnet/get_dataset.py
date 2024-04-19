import os, json
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from tqdm import tqdm

if __name__ == "__main__":
    input_dir = 'D:/Intern-DYNO/detector_laos/raw_data/Laos_household_handwriting_bb_122021'
    save_dir = '/label'

    with open("D:/Intern-DYNO/detector_laos/raw_data/export_122021.json",
              encoding='utf-8') as f:
        raw_data = json.load(f)

    # print(raw_data)

    buffer = dict()
    for idx, item in tqdm(enumerate(raw_data)):
        if not item['annotations'][0]['was_cancelled']:
            try:
                fname = item['data']['image'].split('/')[-1]
                image = Image.open(os.path.join(input_dir, fname))
                drawer = ImageDraw.Draw(image)
                font = ImageFont.load_default()

                for field in item['annotations'][0]['result']:
                    x1 = int((field['value']['x'] * field['original_width']) / 100)
                    y1 = int((field['value']['y'] * field['original_height']) / 100)
                    x2 = int(((field['value']['x'] + field['value']['width']) * field['original_width']) / 100)
                    y2 = int(((field['value']['y'] + field['value']['height']) * field['original_height']) / 100)

                    field_name = field['value']["rectanglelabels"][0]

                    #     drawer.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)
                    #     drawer.rectangle([x1, y1-16, x2, y1], fill=(0, 255, 0))
                    #     drawer.text((x1, y1-14), field_name, (255, 0, 0), font)

                    # buffer[fname] = image
                    # fig, axs = plt.subplots(1, 1)
                    # for i, (fname, image) in enumerate(buffer.items()):
                    #     print(fname)
                    #     axs.imshow(image)
                    #     axs.set_xlabel(fname)

                    # fig.tight_layout()
                    # plt.show()
                    # plt.close('all')
                    # buffer = dict()

                    with open(os.path.join(save_dir, fname[:-4] + '.txt'), 'a', encoding='utf-8') as fw:
                        print('%s|%s|%d|%d|%d|%d' % (fname, field_name, x1, y1, x2, y2), file=fw)
            except:
                pass