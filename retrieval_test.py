import math
import os
import pickle
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

import utils.cnn_model as cnn_model
import utils.data_processing as dp

import shutil


def create_dataset(dataset_name):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transformations = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])
    if dataset_name == 'NUS-WIDE':
        dset_test = dp.DatasetProcessingNUS_WIDE('data/NUS-WIDE', 'test_img.txt', transformations)
        return dset_test
    if dataset_name == 'CIFAR-10':
        dset_test = dp.DatasetProcessingCIFAR_10('data/CIFAR-10', 'test_img.txt', transformations)
        return dset_test
    if dataset_name == 'Project':
        if not os.path.exists('dcodes/adch-project-48bits-record.pkl'):
            record = {}
            dset_database = dp.DatasetProcessingPorject('data/Project', 'database_img.txt', transformations)
            databaseloader = DataLoader(dset_database, batch_size=1, shuffle=False, num_workers=4)
            model = cnn_model.CNNNet('resnet50', 48)
            model.load_state_dict(torch.load('dict/adch-nuswide-48bits.pth', map_location=torch.device('cpu')))
            model.eval()
            rB = encode(model, databaseloader, 4985, 48)
            record['rB'] = rB
            with open('dcodes/adch-project-48bits-record.pkl', 'wb') as fp:
                pickle.dump(record, fp)
        dset_test = dp.DatasetProcessingPorject('data/Project', 'test_img.txt', transformations)
        return dset_test


def encode(model, data_loader, num_data, bit):
    B = np.zeros([num_data, bit], dtype=np.float32)
    for iter, data in enumerate(data_loader, 0):
        data_input, data_ind = data
        output = model(data_input)
        B[data_ind.numpy(), :] = torch.sign(output).detach().numpy()
    return B


def create_retrieval_result_fig(model_name, dataset_name):
    if model_name == 'adch':
        result_dir = '-'.join(['result/result-ADCH', dataset_name, datetime.now().strftime("%y-%m-%d-%H-%M-%S")])
    else:
        result_dir = '-'.join(['result/result-ADSH', dataset_name, datetime.now().strftime("%y-%m-%d-%H-%M-%S")])
    os.mkdir(result_dir)
    model = cnn_model.CNNNet('resnet50', 48)
    dimg_paths = []
    timg_paths = []
    timg_ori_paths = []
    if dataset_name == 'CIFAR-10':
        rLfile = open('data/' + dataset_name + '/database_label.txt')
        qLfile = open('data/' + dataset_name + '/test_label.txt')
        rlables = [int(x.strip()) for x in rLfile]
        qLables = [int(x.strip()) for x in qLfile]
        rL = np.zeros([len(rlables), 10])
        for i in range(len(rlables)):
            rL[i, rlables[i]] = 1
        qL = np.zeros([len(qLables), 10])
        for i in range(len(qLables)):
            qL[i, qLables[i]] = 1
        rLfile.close()
        qLfile.close()
    elif dataset_name == 'NUS-WIDE':
        rL = np.loadtxt('data/NUS-WIDE/database_label.txt', dtype=np.int64)
        qL = np.loadtxt('data/NUS-WIDE/test_label.txt', dtype=np.int64)
    if dataset_name == 'Project':
        dfile = open('data/' + dataset_name + '/database_img.txt', 'r', encoding='utf-8')
        tfile = open('data/' + dataset_name + '/test_img.txt', 'r', encoding='utf-8')
        tfile_ori = open('data/' + dataset_name + '/test_img_bak.txt', 'r', encoding='utf-8')
    else:
        dfile = open('data/' + dataset_name + '/database_img.txt', 'r')
        tfile = open('data/' + dataset_name + '/test_img.txt', 'r')
        tfile_ori = open('data/' + dataset_name + '/test_img_bak.txt', 'r')
    if model_name == 'adch':
        if dataset_name == 'CIFAR-10':
            model.load_state_dict(torch.load('dict/adch-cifar10-48bits.pth', map_location=torch.device('cpu')))
        else:
            model.load_state_dict(torch.load('dict/adch-nuswide-48bits.pth', map_location=torch.device('cpu')))
    else:
        if dataset_name == 'CIFAR-10':
            model.load_state_dict(torch.load('dict/adsh-cifar10-48bits.pth', map_location=torch.device('cpu')))
        else:
            model.load_state_dict(torch.load('dict/adsh-nuswide-48bits.pth', map_location=torch.device('cpu')))
    for line in dfile.readlines():
        dimg_paths.append(line.strip())
    for line in tfile.readlines():
        timg_paths.append(line.strip())
    for line in tfile_ori.readlines():
        timg_ori_paths.append(line.strip())
    dfile.close()
    tfile.close()
    tfile_ori.close()
    tind = []
    for i in range(5):
        tind.append(timg_ori_paths.index(timg_paths[i]))
    model.eval()
    dset_test = create_dataset(dataset_name)
    testloader = DataLoader(dset_test, batch_size=1, shuffle=False, num_workers=4)
    if dataset_name == 'CIFAR-10':
        f = open('dcodes/adch-cifar10-48bits-record.pkl', 'rb')
    elif dataset_name == 'NUS-WIDE':
        f = open('dcodes/adch-nuswide-48bits-record.pkl', 'rb')
    else:
        f = open('dcodes/adch-project-48bits-record.pkl', 'rb')
    record = pickle.load(f)
    f.close()
    qB = encode(model, testloader, len(dset_test), 48)
    rB = record['rB']
    qimgs = []
    rimgs = []
    accuracies = []
    for i in range(5):
        # accuracy = []
        # gnd = (np.dot(qL[tind[i], :], rL.transpose()) > 0).astype(np.float32)
        hamm = calc_hamming_dist(qB[i], rB)
        ind = np.argsort(hamm)
        rimg_dir = os.path.join(result_dir, timg_paths[i].split('/')[1].split('.')[0])
        os.mkdir(rimg_dir)
        for j in range(24):
            rimg_path = os.path.join('data/' + dataset_name, dimg_paths[ind[j]])
            dest_path = os.path.join(rimg_dir, dimg_paths[ind[j]].split('/')[1])
            shutil.copy(rimg_path, dest_path)
        qimg = Image.open(os.path.join('data/' + dataset_name, timg_paths[i]))
        qimg = qimg.resize((200, 100))
        qimgs.append(qimg)
        for j in range(10):
            # accuracy.append(gnd[ind[j]])
            rimg = Image.open(os.path.join('data/' + dataset_name, dimg_paths[ind[j]]))
            rimg = rimg.resize((200, 100))
            rimgs.append(rimg)
        # accuracies.append(accuracy)
    plt.figure(figsize=(24, 5.5))
    for i in range(5):
        plt.subplot(5, 11, i * 11 + 1)
        plt.imshow(qimgs[i])
        plt.axis('off')
        for j in range(10):
            plt.subplot(5, 11, i * 11 + j + 2)
            image = rimgs[i * 10 + j]
            # if accuracies[i][j] < 1:
            #     draw = ImageDraw.Draw(image)
            #     draw.line(
            #         [(2, 4), (2, image.size[1] - 4), (image.size[0] - 3, image.size[1] - 4), (image.size[0] - 3, 4),
            #          (2, 4)], width=4, fill='red')
            plt.imshow(image)
            plt.axis('off')
    plt.subplots_adjust(left=0, top=0.8, right=0.8, bottom=0, wspace=0.02, hspace=0.02)
    plt.savefig(os.path.join(result_dir, 'result'), bbox_inches='tight')
    plt.show()


def calc_hamming_dist(B1, B2):
    q = 48
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH


def calc_linearscan_time_consume(model_name, dataset_name):
    if model_name == 'adch':
        if dataset_name == 'cifar10':
            f = open('dcodes/adch-cifar10-48bits-record.pkl', 'rb')
        else:
            f = open('dcodes/adch-nuswide-48bits-record.pkl', 'rb')
    else:
        if dataset_name == 'cifar10':
            f = open('dcodes/adsh-cifar10-48bits-record.pkl', 'rb')
        else:
            f = open('dcodes/adsh-nuswide-48bits-record.pkl', 'rb')
    record = pickle.load(f)
    f.close()
    rB = record['rB']
    qB = record['qB']
    hamm = np.zeros(rB.shape[0])
    starttime = datetime.now()
    print(starttime)
    for i in range(len(qB)):
        for j in range(len(rB)):
            hamm[j] = calc_hamming_dist(qB[i], rB[j])
        np.argsort(hamm)
    endtime = datetime.now()
    print(endtime)
    timeconsume = (endtime - starttime).seconds
    print(timeconsume)
    return timeconsume


def create_subtable(rB, m):
    subtables = []
    for i in range(m):
        subtable = {}
        subtables.append(subtable)

    def updatedic(subcode, subtable, index):
        if subcode in subtable:
            subtable[subcode].append(index)
        else:
            subtable[subcode] = [index]

    for i in range(rB.shape[0]):
        subcode_length = int(48 / m)
        for j in range(m):
            subcode = tuple(rB[i][j * subcode_length:(j + 1) * subcode_length])
            updatedic(subcode, subtables[j], i)
    return subtables


def cacl_hamming_space_retrieval_time_consume(model_name, dataset_name, m, k):
    if model_name == 'adch':
        if dataset_name == 'cifar10':
            f = open('dcodes/adch-cifar10-48bits-record.pkl', 'rb')
        else:
            f = open('dcodes/adch-nuswide-48bits-record.pkl', 'rb')
    else:
        if dataset_name == 'cifar10':
            f = open('dcodes/adsh-cifar10-48bits-record.pkl', 'rb')
        else:
            f = open('dcodes/adsh-nuswide-48bits-record.pkl', 'rb')
    record = pickle.load(f)
    f.close()
    rB = record['rB']
    qB = record['qB']
    subtables = create_subtable(rB, m)
    subcode_length = int(48 / m)

    def retrieval(subcode, subtable, candidates_ind, r):
        max_dist = math.ceil(r / m)
        for key in subtable.keys():
            ndkey = np.array(key)
            dist = calc_hamming_dist(subcode, ndkey)
            if dist <= max_dist:
                candidates_ind.update(subtable[key])

    starttime = datetime.now()
    print(starttime)
    for i in range(qB.shape[0]):
        print(i)
        r = 2
        candidates_ind = set()
        while len(candidates_ind) < k:
            for j in range(m):
                subcode = rB[i][j * subcode_length:(j + 1) * subcode_length]
                retrieval(subcode, subtables[j], candidates_ind, r)
            for ind in candidates_ind.copy():
                qcode = qB[i]
                rcode = rB[ind]
                dist = calc_hamming_dist(qcode, rcode)
                if dist > r:
                    candidates_ind.remove(ind)
            r = r + 1
    endtime = datetime.now()
    print(endtime)
    timeconsume = (endtime - starttime).seconds
    print(timeconsume)
    return timeconsume


if __name__ == "__main__":
    # cacl_hamming_space_retrieval_time_consume('adch', 'nuswide', 6, 10)
    # cacl_hamming_space_retrieval_time_consume('adch', 'nuswide', 6, 100)
    # cacl_hamming_space_retrieval_time_consume('adch', 'nuswide', 6, 1000)
    create_retrieval_result_fig('adch', 'Project')
    # create_retrieval_result_fig('adsh', 'NUS-WIDE')
