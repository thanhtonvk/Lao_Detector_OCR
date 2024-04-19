import numpy as np
import lmdb
import json
import lz4framed
from tqdm import tqdm
from PIL import Image
import os

def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)

def createLmdb(inputDir, outputDir, gtDir):
    env = lmdb.open(outputDir, map_size=400557776)
    cache = {}
    cnt = 1

    for i, ann_name in tqdm(enumerate(os.listdir(gtDir))):
        ann_data = dict()
        with open(os.path.join(gtDir, ann_name), 'r', encoding='utf-8') as f:
            for line in f.readlines():
                fname, field_name, x1, y1, x2, y2 = line.strip().split('|')
                if field_name not in ann_data.keys():
                    ann_data[field_name] = [[x1, y1, x2, y2]]
                else:
                    ann_data[field_name] += [[x1, y1, x2, y2]]

        imagePath = os.path.join(inputDir, fname)
        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue
        with open(imagePath, 'rb') as f:
            imageBin = f.read()

        imageKey = 'image-%09d'.encode() % cnt
        labelKey = 'label-%09d'.encode() % cnt
        nameKey = 'name-%09d'.encode() % cnt

        cache[imageKey] = lz4framed.compress(imageBin)
        cache[labelKey] = json.dumps(ann_data, indent=2).encode('utf-8')
        cache[nameKey] = fname.encode()

        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}

        cnt += 1

    nSamples = cnt - 1
    cache['num-samples'.encode()] = str(nSamples).encode()
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)

if __name__ == '__main__':
    createLmdb(
        inputDir='D:/Intern-DYNO/detector_laos/raw_data/data',
        outputDir='D:/Intern-DYNO/detector_laos/lmdb_dataset',
        gtDir='D:/Intern-DYNO/detector_laos/label'
    )