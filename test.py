# -*- coding: utf-8 -*-
"""
Created on 18-5-30 下午4:55

@author: ronghuaiyang
"""
from __future__ import print_function
import os
import cv2
from models import *
import torch
import numpy as np
import time
from config import Config
from torch.nn import DataParallel
from inference.network_inf import builder_inf


def get_lfw_list(pair_list):
    with open(pair_list, 'r') as fd:
        pairs = fd.readlines()
    data_list = []
    for pair in pairs:
        splits = pair.split()

        if splits[0] not in data_list:
            data_list.append(splits[0])

        if splits[1] not in data_list:
            data_list.append(splits[1])
    return data_list


def load_image(img_path):
    image = cv2.imread(img_path)
    if image is None:
        return None
    # print("image.shape1",image.shape)
    # image = np.dstack((image, np.fliplr(image)))
    # print("image.shape2",image.shape)
    image = image.transpose((2, 0, 1))
    # print("image.shape3",image.shape)
    image = image[np.newaxis, ::]
    # print("image.shape4",image.shape)
    image = image.astype(np.float32, copy=False)
    image -= 127.5
    image /= 127.5
    return image


def get_featurs(model, test_list, batch_size=16):
    images = None
    features = None
    cnt = 0
    for i, img_path in enumerate(test_list):
        image = load_image(img_path)
        if image is None:
            print('read {} error'.format(img_path))

        if images is None:
            images = image
        else:
            images = np.concatenate((images, image), axis=0)

        if images.shape[0] % batch_size == 0: # or i == len(test_list) - 1:
            cnt += 1

            data = torch.from_numpy(images)
            data = data.to(torch.device("cuda"))
            # print('data.shape', data.shape)
            output = model(data)
            output = output.data.cpu().numpy()

            # fe_1 = output[::2]
            # fe_2 = output[1::2]
            # feature = np.hstack((fe_1, fe_2))
            feature = output
            print(cnt * batch_size, len(test_list), feature.shape)

            if features is None:
                features = feature
            else:
                features = np.vstack((features, feature))

            images = None

    return features, cnt


def load_model(model, model_path):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)


def get_feature_dict(test_list, features):
    fe_dict = {}
    for i, each in enumerate(test_list):
        # key = each.split('/')[1]
        fe_dict[each] = features[i]
    return fe_dict


def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


def cal_accuracy(y_score, y_true):
    y_score = np.asarray(y_score)
    y_true = np.asarray(y_true)
    best_acc = 0
    best_th = 0
    for i in range(len(y_score)):
        th = y_score[i]
        y_test = (y_score >= th)
        acc = np.mean((y_test == y_true).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_th = th

    return (best_acc, best_th)


def test_performance(fe_dict, pair_list):
    with open(pair_list, 'r') as fd:
        pairs = fd.readlines()

    sims = []
    labels = []
    skipped = 0
    for pair in pairs:
        splits = pair.split()
        if splits[0] not in fe_dict:
            skipped += 1
            continue
        if splits[1] not in fe_dict:
            skipped += 1
            continue
        fe_1 = fe_dict[splits[0]]
        fe_2 = fe_dict[splits[1]]
        label = int(splits[2])
        sim = cosin_metric(fe_1, fe_2)

        sims.append(sim)
        labels.append(label)
    print(f"skipped {skipped}")
    acc, th = cal_accuracy(sims, labels)
    return acc, th


def lfw_test(model, img_paths, identity_list, compair_list, batch_size):
    s = time.time()
    cnt = 1
    if os.path.isfile('features.npy'):
        features = np.load('features.npy')
    else:
        features, cnt = get_featurs(model, img_paths, batch_size=batch_size)    
        np.save('features.npy', features)
        
    print(features.shape)
    t = time.time() - s
    print('total time is {}, average time is {}'.format(t, t / cnt))

    print(f"truncating identity_list from {len(identity_list)} to {len(features)}")
    identity_list = identity_list[:len(features)]

    fe_dict = get_feature_dict(identity_list, features)
    acc, th = test_performance(fe_dict, compair_list)
    print('lfw face verification accuracy: ', acc, 'threshold: ', th)
    return acc


if __name__ == '__main__':

    opt = Config()
    opt.arch = 'iresnet100'
    opt.embedding_size = 512
    opt.resume = "weights/magface_epoch_00025.pth"
    opt.cpu_mode = False

    model = None
    if not os.path.isfile('features.npy'):

        model = builder_inf(opt)
        # if opt.backbone == 'resnet18':
        #     model = resnet_face18(opt.use_se)
        # elif opt.backbone == 'resnet34':
        #     model = resnet34()
        # elif opt.backbone == 'resnet50':
        #     model = resnet50()

        model = DataParallel(model)
        # load_model(model, opt.test_model_path)
        # model.load_state_dict(torch.load(opt.test_model_path))
        model.to(torch.device("cuda"))
        model.eval()

    identity_list = get_lfw_list(opt.lfw_test_list)
    img_paths = [os.path.join(opt.lfw_root, each) for each in identity_list]

    lfw_test(model, img_paths, identity_list, opt.lfw_test_list, opt.test_batch_size)

