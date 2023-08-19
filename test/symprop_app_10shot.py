#!/usr/bin python3
# -*- coding: utf-8 -*-
# here put the import lib

import os
import sys
import json
import logging
import copy
import joblib

from flask import Flask
from flask import request

import numpy as np

from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import dataloader
from torch.nn import init
from torch.optim import lr_scheduler
from torch.nn.utils.rnn import pack_padded_sequence

sys.path.insert(0, os.path.abspath(""))
from models import baseline_mlp, symprop_maml_basenet, symprop_maml_metalearner
from utils.data_utils import data_process, data_utils
from utils.fetch_features import *

app = Flask(__name__)

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y/%m/%d %H:%M:%S", stream=sys.stdout, level=logging.DEBUG
)

# init torch
print("torch.version:  ", torch.__version__)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("current_device: ", device)
if torch.cuda.is_available():
    print("device_name:    ", torch.cuda.get_device_name(device))
    torch.cuda.empty_cache()

TIME_STEPS = 50
n_way = 2
k_spt = 8  # support data 的个数
k_query = 8  # query data 的个数
time_step = TIME_STEPS
data_feature_num = 65
task_num = 8
batch_size = task_num

N_SHOT = 1
# action_index_4step = {"move": (1, 3), "pick_cube": (4, 6), "transport": (7, 9), "place_cube": (10, 12)}
action_index_4step = {"move": (0, 2), "pick_cube": (2, 4), "transport": (4, 6), "place_cube": (6, 8)}
finetuning_dataset = np.load(os.path.join("symbolic_proposition", "{}_shot_dataset.npy".format(str(N_SHOT))))


def load_data_cache(dataset):

    setsz = k_spt * n_way
    querysz = k_query * n_way
    data_cache = []

    # print('preload next 10 caches of batch_size of batch.')
    # 提前载入10个batch的数据
    for sample in range(10):  # num of epochs

        x_spts, y_spts, x_qrys, y_qrys = [], [], [], []

        for i in range(batch_size):  # one batch means one set

            x_spt, y_spt, x_qry, y_qry = [], [], [], []

            # 从多类中选择n_way个类
            # selected_cls = np.random.choice(
            #     dataset.shape[0], n_way, replace=False)
            selected_cls = np.array([0, 1])

            for j, cur_class in enumerate(selected_cls):

                selected_data = np.random.choice(dataset.shape[1], k_spt + k_query, replace=False)

                # 构造support集和query集
                x_spt.append(dataset[cur_class][selected_data[:k_spt]])
                x_qry.append(dataset[cur_class][selected_data[k_spt:]])
                y_spt.append([j for _ in range(k_spt)])
                y_qry.append([j for _ in range(k_query)])

            # shuffle inside a batch
            perm = np.random.permutation(n_way * k_spt)
            x_spt = np.array(x_spt).reshape(n_way * k_spt, time_step, data_feature_num)[perm]
            y_spt = np.array(y_spt).reshape(n_way * k_spt)[perm]
            perm = np.random.permutation(n_way * k_query)
            x_qry = np.array(x_qry).reshape(n_way * k_query, time_step, data_feature_num)[perm]
            y_qry = np.array(y_qry).reshape(n_way * k_query)[perm]

            x_spts.append(x_spt)
            y_spts.append(y_spt)
            x_qrys.append(x_qry)
            y_qrys.append(y_qry)

        #         print(x_spts[0].shape)
        # [b, setsz = n_way * k_spt, 1, 84, 84]
        x_spts = np.array(x_spts).astype(np.float32).reshape(batch_size, setsz, time_step, data_feature_num)
        y_spts = np.array(y_spts).astype(int).reshape(batch_size, setsz)
        # [b, qrysz = n_way * k_query, 1, 84, 84]
        x_qrys = np.array(x_qrys).astype(np.float32).reshape(batch_size, querysz, time_step, data_feature_num)
        y_qrys = np.array(y_qrys).astype(int).reshape(batch_size, querysz)
        #         print(x_qrys.shape)
        data_cache.append([x_spts, y_spts, x_qrys, y_qrys])

    return data_cache


def next(mode, indexes, datasets_cache, datasets):
    """
    Return next batch from the dataset with name.
    """
    # update cache if indexes is larger than len(data_cache)
    if indexes[mode] >= len(datasets_cache[mode]):
        indexes[mode] = 0
        datasets_cache[mode] = load_data_cache(datasets[mode])

    next_batch = datasets_cache[mode][indexes[mode]]
    indexes[mode] += 1

    return next_batch


maml_model = symprop_maml_metalearner.MetaLearner().to(device)
maml_model.load_state_dict(torch.load(os.path.join("symbolic_proposition", "maml.pt")))


@app.route("/prediction/maml/", methods=["POST"])
def predict_maml():
    if request.method == "POST":
        logging.info(" ------------------------------------------------------ ")
        logging.info(" symbolic propositions check service: /prediction/maml/ ")
        logging.info(" ------------------------------------------------------ ")
        input_payload_str = request.get_data()
        input_payload = json.loads(input_payload_str)
        action = input_payload["action"]
        data = input_payload["data"]  # list type
        logging.info(" check action: {} ".format(action))

        z_score_mean = np.load("symbolic_proposition/symprop_maml_mean.npy")
        z_score_std = np.load("symbolic_proposition/symprop_maml_std.npy")

        data = data_process.z_score_normalization(
            data_process.select_features(np.array(data), FETCH_FEATURE_65), z_score_mean, z_score_std
        )

        finetuning_index = action_index_4step[action]
        # finetuning_dataset = np.load(os.path.join("symbolic_proposition", "20_shot_dataset.npy"))
        x_finetuning = finetuning_dataset[finetuning_index[0] : finetuning_index[1]]
        datasets_cache = {"predict": load_data_cache(x_finetuning)}
        indexes = {"predict": 0}
        datasets = {"predict": x_finetuning}

        # model_out_list = []
        X = data
        pred_x = torch.FloatTensor().to(device)
        for i in range(8):

            start_index = i * 5
            end_index = TIME_STEPS + i * 5
            x = X[start_index:end_index]

            x = torch.FloatTensor(x).to(device)
            x = torch.unsqueeze(x, 0)
            if pred_x.size(0) == 0:
                pred_x = x
            else:
                pred_x = torch.cat((pred_x, x), dim=0)

        # accs = []
        pred_ys = []

        for _ in range(16):
            x_spt, y_spt, _, _ = next("predict", indexes, datasets_cache, datasets)
            x_spt, y_spt  = (
                torch.from_numpy(x_spt).to(device),
                torch.from_numpy(y_spt).to(device),
                # torch.from_numpy(x_qry).to(device),
                # torch.from_numpy(y_qry).to(device),
            )
            # print("y_qry: ", y_qry)

            for x_spt_one, y_spt_one in zip(x_spt, y_spt):

                pred_y = maml_model.predict(x_spt_one, y_spt_one,  pred_x)
                # accs.append(test_acc)
                pred_ys.append(pred_y.cpu().numpy())

        # accs = np.array(accs).mean(axis=0).astype(np.float16)
        # logging.debug("Test Accuracy: {}".format(accs))
        _1_prob = np.mean(pred_ys, axis=0)
        # model_out_list.append(_1_prob)
        # logging.debug("prediction details: {}".format(pred_ys))
        logging.debug("prediction details: {}".format(_1_prob))
        prob = float(np.mean(_1_prob))
        logging.debug("prob: {}".format(prob))
        
        if prob >= 0.5:
            res = 1
        elif prob < 0.5:
            res = 0

        logging.info("prediction result: {}".format(dict([(action, res)])))
        print()
        print()
        return json.dumps(dict([(action, res)]))


@app.route("/prediction/baseline/mlp/", methods=["POST"])
def predict_baseline_mlp():
    if request.method == "POST":
        logging.info(" -------------------------------------------------- ")
        logging.info(" symbolic propositions check service: baseline/mlp/ ")
        logging.info(" -------------------------------------------------- ")
        input_payload_str = request.get_data()
        input_payload = json.loads(input_payload_str)
        action = input_payload["action"]
        data = input_payload["data"]  # list type
        logging.info(" check action: {} ".format(action))
        z_score_mean = np.load("test/baseline/baseline_lstm_mean.npy")
        z_score_std = np.load("test/baseline/baseline_lstm_std.npy")

        data = data_process.z_score_normalization(
            data_process.select_features(np.array(data), FETCH_FEATURE_65), z_score_mean, z_score_std
        )

        with open("./test/baseline/action_state_variable.json") as f:
            action_state = json.load(f)

        res = []
        for symbol in action_state[action]:
            logging.debug("check symbol: {}".format(symbol))
            X = data_process.select_features(data, action_state[action][symbol])

            n_features = len(action_state[action][symbol])
            model = baseline_mlp.baseline_lstm(n_features)
            model.load_state_dict(torch.load(os.path.join("test", "baseline", "models", symbol + ".pt")))
            model.to(device)
            model.eval()

            model_out_list = []
            for i in range(25):

                start_index = i * 5
                end_index = TIME_STEPS + i * 5
                x = X[start_index:end_index]

                pred_x = torch.FloatTensor(x).to(device)
                pred_x = torch.unsqueeze(pred_x, 0)
                pred_x = pack_padded_sequence(pred_x, torch.tensor([TIME_STEPS]), batch_first=True)
                out = model(pred_x)
                logging.debug("out: {}".format(out))
                pm = torch.max(out, 1)[1].data.cpu().numpy()[0]
                model_out_list.append(pm)

            logging.debug("prediction details: {}".format(model_out_list))
            prob = float(np.mean(model_out_list))
            logging.debug("prob: {}".format(prob))
            print()
            if prob >= 0.5:
                res.append(1)
            elif prob < 0.5:
                res.append(0)

        logging.info("prediction result: {}".format(dict(zip(action_state[action], res))))
        print()
        print()
        return json.dumps(dict(zip(action_state[action], res)))


@app.route("/read_yaml/", methods=["post"])
def read_yaml_data():
    if request.method == "POST":
        logging.info(" ---------------------------------- ")
        logging.info(" read yaml data service: read_yaml/ ")
        logging.info(" ---------------------------------- ")
        input_payload_str = request.get_data()
        input_payload = json.loads(input_payload_str)
        joint_data_path = input_payload["joint_data_path"]
        object_data_path = input_payload["object_data_path"]
        logging.debug(" joint_data_path: {}".format(joint_data_path))
        logging.debug(" object_data_path: {}".format(object_data_path))
        joint_data = data_utils.read_yaml_data(joint_data_path, "joint_state")
        object_data = data_utils.read_yaml_data(object_data_path, "object_pose")
        logging.debug(" data preprocess ... ")
        env_data = data_process.data_preprocess(joint_2d_list=joint_data, object_2d_list=object_data)
        _len = 150
        if len(env_data) >= _len:
            env_data = env_data[-_len:]
        else:
            logging.debug(" Length < _len ")
            raise IndexError
        print()
        print()
        return json.dumps({"env_data": env_data})


if __name__ == "__main__":
    app.debug = False
    app.run(host="0.0.0.0", port=5001)
