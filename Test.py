from datetime import datetime
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import pickle
import math


def save_result():
    dimg_paths = []
    timg_paths = []
    dfile = open('./data/database_img.txt', 'r')
    for line in dfile.readlines():
        dimg_paths.append(line.strip('\n'))
    dfile.close()
    tfile = open('./data/test_img.txt', 'r')
    for line in tfile.readlines():
        timg_paths.append(line.strip('\n'))
    tfile.close()
    result_dir = '-'.join(['result/result-ADCH-nuswide', datetime.now().strftime("%y-%m-%d-%H-%M-%S")])
    os.mkdir(result_dir)
    for i in range(len(timg_paths)):
        ind = np.ones([1, len(dimg_paths)])
        resultset_dir = os.path.join(result_dir, str(i + 1))
        os.mkdir(resultset_dir)
        qimg_path = os.path.join('data/NUS-WIDE', timg_paths[i])
        shutil.copy(qimg_path, result_dir)
        rimg_dir = os.path.join(result_dir, 'top-10')
        os.mkdir(rimg_dir)
        for j in range(10):
            rimg_path = os.path.join('data/NUS-WIDE', dimg_paths[ind[j]])
            shutil.copy(rimg_path, os.path.join(rimg_dir, str(j + 1) + '.jpg'))


def create_resultfig():
    images = []
    for figname in os.listdir('data/plane'):
        figpath = os.path.join('data/plane', figname)
        image = Image.open(figpath)
        image = image.resize((256, 224))
        draw = ImageDraw.Draw(image)
        draw.line([(2, 4), (2, image.size[1]-4), (image.size[0]-3, image.size[1]-4), (image.size[0]-3, 4), (2, 4)],
                  width=4, fill='red')
        images.append(image)
    plt.figure(figsize=(13.5, 5.5))
    for i in range(5 * len(images)):
        plt.subplot(5, 11, i + 1)
        plt.imshow(images[i % 11])
        plt.axis('off')
    plt.subplots_adjust(left=0, top=1, right=1, bottom=0, wspace=0.02, hspace=0)
    plt.savefig('image_name', bbox_inches='tight')
    plt.show()


def create_subtable(rB):
    subtable1 = {}
    subtable2 = {}
    subtable3 = {}

    def updatedic(subcode, subtable, index):
        if subcode in subtable:
            subtable[subcode].append(index)
        else:
            subtable[subcode] = [index]

    for i in range(rB.shape[0]):
        subcode1 = tuple(rB[i][0:16])
        updatedic(subcode1, subtable1, i)
        subcode2 = tuple(rB[i][16:32])
        updatedic(subcode2, subtable2, i)
        subcode3 = tuple(rB[i][32:48])
        updatedic(subcode3, subtable3, i)
    return subtable1, subtable2, subtable3


def hamming_space_retrieval(k):
    f = open('record/cifar10-adch-cifar10-48bits-record.pkl', 'rb')
    record = pickle.load(f)
    f.close()
    rB = record['rB']
    subtable1, subtable2, subtable3 = create_subtable(rB)
    f = open('record/cifar10-adch-cifar10-48bits-record.pkl', 'rb')
    record = pickle.load(f)
    f.close()
    qB = record['qB']

    def retrieval(subcode, subtable, candidates_ind, usemap, r):
        max_dist = math.ceil(r / 3)
        for key in subtable.keys():
            key = np.array(key)
            dist = 0.5 * (16 - np.dot(subcode, key.transpose()))
            if dist <= max_dist:
                for ind in subtable[tuple(key)]:
                    if not usemap[ind]:
                        candidates_ind.append(ind)

    starttime = datetime.now()
    print(starttime)
    for i in range(qB.shape[0]):
        r = 2
        candidates_ind = []
        while len(candidates_ind) < k:
            usemap = [False] * rB.shape[0]
            subcodes1 = qB[i][0:16]
            retrieval(subcodes1, subtable1, candidates_ind, usemap, r)
            subcodes2 = qB[i][16:32]
            retrieval(subcodes2, subtable2, candidates_ind, usemap, r)
            subcodes3 = qB[i][16:32]
            retrieval(subcodes3, subtable3, candidates_ind, usemap, r)
            for ind in candidates_ind:
                qcode = qB[i]
                rcode = rB[ind]
                dist = 0.5 * (16 - np.dot(qcode, rcode.transpose()))
                if dist > r:
                    candidates_ind.remove(ind)
            r = r + 1
    endtime = datetime.now()
    print(endtime)
    timeconsume = (endtime - starttime).seconds
    print(timeconsume)
    return timeconsume


if __name__ == "__main__":
    create_resultfig()
