#!/usr/bin python3
# -*- coding: utf-8 -*-

# here put the import libimp
import os
import platform
import yaml
import pickle
import numpy as np
import pandas as pd
import copy
from typing import Any, Dict, List, Tuple
from ruamel.yaml import YAML
import srsly

from utils.fetch_features import *


class data_utils:
    @staticmethod
    def size_format(file_path):
        byte_size = os.path.getsize(file_path)
        if byte_size < 1000:
            return "%i" % byte_size + "size"
        elif 1000 <= byte_size < 1000000:
            return "%.1f" % float(byte_size / 1000) + "KB"
        elif 1000000 <= byte_size < 1000000000:
            return "%.1f" % float(byte_size / 1000000) + "MB"
        elif 1000000000 <= byte_size < 1000000000000:
            return "%.1f" % float(byte_size / 1000000000) + "GB"
        elif 1000000000000 <= byte_size:
            return "%.1f" % float(byte_size / 1000000000000) + "TB"


    @staticmethod
    def fast_read_yaml_data(file_path: str, file_type: str) -> list:
        """
        @description  :  fast read fetch data from a yaml joint/state file
        ---------
        @param  :  
        ---------
        @Returns  :  
        ---------
        """
        if os.path.isfile(file_path):
            with open(file_path, "r") as f:
                raw_data = f.read()
        else:
            raise FileExistsError

        data_yaml = yaml.load_all(raw_data, Loader=yaml.FullLoader)
        file_data = []
        count = 9
        for data in data_yaml:
            if data != None and data != '':
                row = []
                for k, v in data.items():
                    if file_type == "joint_state":
                        raise ValueError

                    elif file_type == "object_pose":
                        
                        if count == 10:
                            count = 0
                            if k == "pose":
                                for data in v:
                                    row.append(data["position"]["x"])
                                    row.append(data["position"]["y"])
                                    row.append(data["position"]["z"])
                                    row.append(data["orientation"]["x"])
                                    row.append(data["orientation"]["y"])
                                    row.append(data["orientation"]["z"])
                                    row.append(data["orientation"]["w"])
                            elif k == "twist":
                                for data in v:
                                    row.append(data["linear"]["x"])
                                    row.append(data["linear"]["y"])
                                    row.append(data["linear"]["z"])
                                    row.append(data["angular"]["x"])
                                    row.append(data["angular"]["y"])
                                    row.append(data["angular"]["z"])
                        count += 1
                if row != []:
                    file_data.append(np.array(row).flatten().tolist())

        return file_data                
        

    @staticmethod
    def read_yaml_data(file_path: str, file_type: str) -> list:
        """
        @description  :  read fetch data from a yaml joint/state file
        ---------
        @param  :  file_path -> string
                   file_type -> joint_state/object_pose
        -------
        @Returns  : list[, []]
        -------
        """

        if os.path.isfile(file_path):
            with open(file_path, "r") as f:
                raw_data = f.read()
        else:
            raise FileExistsError

        # data_yaml = yaml.load_all(raw_data, Loader=yaml.BaseLoader)
        # data_yaml = yaml.load_all(raw_data, Loader=yaml.FullLoader)
        # data_yaml = srsly.yaml_loads(raw_data)
        # data_yaml = yaml.safe_load_all(raw_data)
        
        ryaml=YAML(typ='safe')   # default, if not specfied, is 'rt' (round-trip)
        data_yaml = ryaml.load_all(raw_data)

        file_data = []
        for data in data_yaml:
            if data != None and data != '':
                # print(type(data))
                row = []
                for k, v in data.items():
                    if file_type == "joint_state":
                        if (k == "position") or (k == "velocity") or (k == "effort"):
                            # v = list([float(d) for d in v])
                            row.append(v)

                    elif file_type == "object_pose":
                        if k == "pose":
                            for data in v:
                                row.append(data["position"]["x"])
                                row.append(data["position"]["y"])
                                row.append(data["position"]["z"])
                                row.append(data["orientation"]["x"])
                                row.append(data["orientation"]["y"])
                                row.append(data["orientation"]["z"])
                                row.append(data["orientation"]["w"])
                        elif k == "twist":
                            for data in v:
                                row.append(data["linear"]["x"])
                                row.append(data["linear"]["y"])
                                row.append(data["linear"]["z"])
                                row.append(data["angular"]["x"])
                                row.append(data["angular"]["y"])
                                row.append(data["angular"]["z"])

                        # row = list([float(d) for d in row])
                file_data.append(np.array(row).flatten().tolist())

        return file_data

    @staticmethod
    def store_in_pickle(data: Any, save_path: str, protocol: int = pickle.HIGHEST_PROTOCOL):
        with open(save_path, "wb") as f:
            pickle.dump(data, f, protocol)

    @staticmethod
    def read_from_pickle(read_path: str) -> Any:
        with open(read_path, "rb") as f:
            data = pickle.load(f)
        return data

    @staticmethod
    def _last_folder(path: str) -> str:
        return os.path.split(path)[1]

    @staticmethod
    def _last2_folder(path: str) -> str:
        return os.path.split(os.path.split(path)[0])[1]

    @staticmethod
    def if_readme(file_name):
        if file_name == "readme.txt" or file_name == "README.txt" or file_name == "readme.md" or file_name == "README.md":
            print("skip readme")
            print()
            return True
        else:
            return False

    # @staticmethod
    # def rename_objec10_2_object10(path):
    #     for abspath, _, file_names in os.walk(path):
    #         if "objec10.txt" in file_names:
    #             os.rename(os.path.join(abspath, "objec10.txt"), os.path.join(abspath, "object10.txt"))

    # @staticmethod
    # def del_old_joint6object6(data_path):
    #     """
    #     @description  :  删除采集的arm_data数据中只有6个joint、object文件中的joint6.txt和object6.txt
    #     ---------
    #     @param  :
    #     -------
    #     @Returns  :
    #     -------
    #     """

    #     for abspath, _, file_names in os.walk(data_path):
    #         if os.path.split(os.path.split(abspath)[0])[1] == "datas_arm":
    #             if "joint6.txt" in file_names and "joint7.txt" not in file_names:
    #                 file_path = os.path.join(abspath, "joint6.txt")
    #                 if os.path.exists(file_path):
    #                     print(file_path)
    #                     os.remove(file_path)

    #             if "object6.txt" in file_names and "object7.txt" not in file_names:
    #                 file_path = os.path.join(abspath, "object6.txt")
    #                 if os.path.exists(file_path):
    #                     print(file_path)
    #                     os.remove(file_path)

    # @staticmethod
    # def _file_size_check_remove(file_path, threshold_mb):
    #     file_size = data_utility.size_format(file_path)
    #     if file_size == "0size":
    #         return True
    #     elif file_size[-2:] == "KB":
    #         return False
    #     elif file_size[-2:] == "GB":
    #         return True
    #     elif file_size[-2:] == "MB":
    #         if float(file_size[:-2]) > threshold_mb:
    #             return True
    #         else:
    #             return False
    #     else:
    #         raise ValueError("--- file size error ---")

    # @staticmethod
    # def del_overlarge_file(data_path):
    #     """
    #     @description  :
    #     ---------
    #     @param  :
    #     -------
    #     @Returns  :
    #     -------
    #     """

    #     del_file_list = []
    #     for abspath, _, file_names in os.walk(data_path):
    #         if len(file_names) > 0:
    #             print("current abspath: ", abspath)
    #             marked_files = []
    #             for file_name in file_names:
    #                 if (
    #                     file_name == "readme.txt"
    #                     or file_name == "README.txt"
    #                     or file_name == "readme.md"
    #                     or file_name == "README.md"
    #                 ):
    #                     continue

    #                 if file_name not in marked_files:
    #                     if re.findall("\d*j.*_.*to.*_.*\.txt", file_name) != []:

    #                         correspond_file_name = file_name.replace("j", "o", 1)
    #                         file_path = os.path.join(abspath, file_name)
    #                         correspond_file_path = os.path.join(abspath, correspond_file_name)

    #                         if os.path.exists(correspond_file_path):
    #                             # 检查大于80.0mb的objects文件
    #                             if data_utility._file_size_check_remove(correspond_file_path, threshold_mb=80.0):
    #                                 del_file_list.append(correspond_file_path)
    #                                 del_file_list.append(file_path)
    #                             if data_utility._file_size_check_remove(file_path, threshold_mb=80.0):
    #                                 del_file_list.append(correspond_file_path)
    #                                 del_file_list.append(file_path)
    #                             marked_files.append(file_name)
    #                             marked_files.append(correspond_file_name)
    #                         else:
    #                             if data_utility._file_size_check_remove(file_path, threshold_mb=80.0):
    #                                 del_file_list.append(file_path)
    #                             marked_files.append(file_name)

    #                     elif re.findall("\d*o.*_.*to.*_.*\.txt", file_name) != []:
    #                         # fine correspond file
    #                         correspond_file_name = file_name.replace("o", "j", 1)
    #                         file_path = os.path.join(abspath, file_name)
    #                         correspond_file_path = os.path.join(abspath, correspond_file_name)

    #                         if os.path.exists(correspond_file_path):
    #                             # 检查大于80.0mb的objects文件
    #                             if data_utility._file_size_check_remove(file_path, threshold_mb=80.0):
    #                                 del_file_list.append(file_path)
    #                                 del_file_list.append(correspond_file_path)
    #                             if data_utility._file_size_check_remove(correspond_file_path, threshold_mb=80.0):
    #                                 del_file_list.append(correspond_file_path)
    #                                 del_file_list.append(file_path)
    #                             marked_files.append(file_name)
    #                             marked_files.append(correspond_file_name)
    #                         else:
    #                             if data_utility._file_size_check_remove(file_path, threshold_mb=80.0):
    #                                 del_file_list.append(file_path)
    #                             marked_files.append(file_name)

    #                     elif re.findall("7motion_joints_\d+\.pickle", file_name) != []:
    #                         # find correspond file
    #                         correspond_file_name = file_name.replace("joints", "objects", 1)
    #                         file_path = os.path.join(abspath, file_name)
    #                         correspond_file_path = os.path.join(abspath, correspond_file_name)

    #                         if os.path.exists(correspond_file_path):
    #                             # 检查大于23.0mb的objects文件
    #                             if data_utility._file_size_check_remove(correspond_file_path, threshold_mb=23.0):
    #                                 del_file_list.append(correspond_file_path)
    #                                 del_file_list.append(file_path)
    #                             if data_utility._file_size_check_remove(file_path, threshold_mb=80.0):
    #                                 del_file_list.append(correspond_file_path)
    #                                 del_file_list.append(file_path)

    #                         marked_files.append(file_name)
    #                         marked_files.append(correspond_file_name)

    #                     elif re.findall("7motion_objects_\d+\.pickle", file_name) != []:
    #                         # find correspond file
    #                         correspond_file_name = file_name.replace("objects", "joints", 1)
    #                         file_path = os.path.join(abspath, file_name)
    #                         correspond_file_path = os.path.join(abspath, correspond_file_name)

    #                         if os.path.exists(correspond_file_path):
    #                             # 检查大于23.0mb的objects文件
    #                             if data_utility._file_size_check_remove(file_path, threshold_mb=23.0):
    #                                 del_file_list.append(file_path)
    #                                 del_file_list.append(correspond_file_path)
    #                             if data_utility._file_size_check_remove(correspond_file_path, threshold_mb=80.0):
    #                                 del_file_list.append(correspond_file_path)
    #                                 del_file_list.append(file_path)

    #                         marked_files.append(file_name)
    #                         marked_files.append(correspond_file_name)

    #     print(del_file_list)
    #     print("del_file_list length: ", len(del_file_list))

    #     # del files
    #     for file in del_file_list:
    #         print("remove file: ", file)
    #         print("file size: ", data_utility.size_format(file))
    #         print()
    #         os.remove(file)

    @staticmethod
    def breakfilepath_convert(file_path):
        with open(file_path, "r") as f:
            content = f.readlines()
        paths = [c.strip().split(",")[1] for c in content]
        file_names = [c.strip().split(",")[0] for c in content]
        if len(file_names) != len(paths):
            raise RuntimeError("file index error")

        convert_paths = []
        for path in paths:
            index = 0
            new_path = ""
            if len(path.split("/")) > 1:
                names = path.split("/")
                for i, name in enumerate(names):
                    if name == "data_archive":
                        index = i
                        break

                if platform.system() == "Windows":
                    new_path = os.path.join(os.getcwd(), "\\".join(names[index:]))
                elif platform.system() == "Linux":
                    new_path = os.path.join(os.getcwd(), "/".join(names[index:]))

            elif len(path.split("\\")) > 1:
                names = path.split("\\")
                for i, name in enumerate(names):
                    if name == "data_archive":
                        index = i
                        break

                if platform.system() == "Windows":
                    new_path = os.path.join(os.getcwd(), "\\".join(names[index:]))
                elif platform.system() == "Linux":
                    new_path = os.path.join(os.getcwd(), "/".join(names[index:]))

            convert_paths.append(new_path)

        if len(file_names) != len(paths):
            raise RuntimeError("file index error")

        res = []
        for file_name, convert_path in zip(file_names, convert_paths):
            s = file_name + "," + convert_path
            res.append(s)

        with open(file_path, "w") as f:
            f.write("\n".join(res))

    # @staticmethod
    # def add_filename_2_break_file(file_path):
    #     with open(file_path, "r") as f:
    #         content = f.readlines()
    #     res = []
    #     paths = [c.strip() for c in content]
    #     file_names = os.listdir(RAW_PICKLE_TRAJECTORY_PATH)
    #     file_names.sort(key=lambda e: int(os.path.splitext(e)[0].split("_")[-1]))

    #     for index, path in enumerate(paths):
    #         s = file_names[index] + "," + path
    #         res.append(s)

    #     print(res)
    #     with open(file_path, "w") as f:
    #         f.write("\n".join(res))


class data_process:
    def __init__(self):
        pass

    def _cal_diff(self, np_arr1, np_arr2):
        return np.linalg.norm(np_arr1 - np_arr2)

    @staticmethod
    def tukeys_test(self, np_arr):
        rate = 50
        percentile = np.percentile(np_arr, [0, 25, 50, 75, 100], axis=0)
        iqr = percentile[3, :] - percentile[1, :]
        up_threshold = percentile[3, :] + iqr * rate
        low_threshold = percentile[1, :] - iqr * rate

        res = []
        for row in np_arr:
            if (row > up_threshold).any() or (row < low_threshold).any():
                continue
            res.append(row)
        res = np.array(res)
        return res

    def _valid_data_filter(self, motion_arr, threshold=6):
        row, col = motion_arr.shape
        last_row = None
        start_step = None
        end_step = None
        cur_row = None
        i = 0
        for i in range(row):
            cur_row = motion_arr[i]
            if i == 0:
                last_row = cur_row
                continue

            diff = self._cal_diff(cur_row, last_row)
            #         print('diff: ',diff)
            if abs(diff) > threshold and start_step == None and end_step == None:
                start_step = i - 1
                break

            last_row = cur_row

        cur_row = None
        last_row = None
        for j in range(row)[::-1]:
            cur_row = motion_arr[i]
            if j == row - 1:
                last_row = cur_row
                continue

            diff = self._cal_diff(cur_row, last_row)
            if abs(diff) > threshold and start_step != None and end_step == None:
                end_step = j + 1
                break

            last_row = row

        return motion_arr[start_step:end_step]

    @staticmethod
    def select_features(np_arr: np.ndarray, features: list) -> np.ndarray:
        """
        @description  :  特征过滤
        ---------
        @param  :
        ---------
        @Returns  :  过滤后的numpy array
        ---------
        """

        if np_arr.shape[1] == 123:
            df = pd.DataFrame(np_arr, columns=FETCH_FEATURE_123)
            return df[features].values
        elif np_arr.shape[1] == 136:
            df = pd.DataFrame(np_arr, columns=FETCH_FEATURE_136)
            return df[features].values
        elif np_arr.shape[1] == 97:
            df = pd.DataFrame(np_arr, columns=FETCH_FEATURE_97)
            return df[features].values
        elif np_arr.shape[1] == 65:
            df = pd.DataFrame(np_arr, columns=FETCH_FEATURE_65)
            return df[features].values
        # elif np_arr.shape[1] == 62:
        #     print("62")
        else:
            raise RuntimeError("feature number error.")

    @classmethod
    def data_filter(cls, _2d_list: list, *func_filters, **kwargs) -> list:
        """
        @description  :
        ---------
        @param  :
        ---------
        @Returns  :
        ---------
        """

        valid_data = np.array(copy.deepcopy(_2d_list))
        for func in func_filters:
            if func.__name__ == "select_features":
                valid_data = func(valid_data, kwargs["features"])
            else:
                valid_data = func(valid_data)
        return valid_data.tolist()

    @staticmethod
    def _list_align(input_2dlist, target_length):
        """
        @description  :  对齐长度不同的joint和object文件
        ---------
        @param  :  input_2dlist -> list[][]
        ---------
        @Returns  :
        ---------
        """

        diff = len(input_2dlist) - target_length
        step = target_length // (diff + 1)
        for i in range(diff):
            input_2dlist.pop(i + step)
        return input_2dlist

    @staticmethod
    def _down_sampling(input_2d_list, step=10) -> List:
        """
        @description  :  下采样1000hz的oject到与joint数据相同的100hz
        ---------
        @param  :
        -------
        @Returns  :  采样结果
        -------
        """
        res = []
        for i in range(0, len(input_2d_list), step):
            res.append(input_2d_list[i])
        return res

    @classmethod
    def data_preprocess(cls, joint_2d_list, object_2d_list) -> List:
        """
        @description  : down sampling object data, align joint and object data, merge
        ---------
        @param joint_2d_list : fetch joint data
        @param object_2d_list : fetch object data
        ---------
        @Returns _2d_list : preprocessed data
        ---------
        """

        # 下采样
        jf = joint_2d_list
        of = cls._down_sampling(object_2d_list)

        # 对齐数据
        jf_length = len(jf)
        of_length = len(of)
        if jf_length > of_length:
            jf = cls._list_align(jf, of_length)
        elif jf_length < of_length:
            of = cls._list_align(of, jf_length)

        # 合并数据
        arr = np.concatenate(
            (
                np.array(jf, dtype=float),
                np.array(of, dtype=float),
            ),
            axis=1,
        )
        return arr.tolist()

    @staticmethod
    def z_score_normalization(x, mean, std):
        """
        @description  :  normalization by mean and std
        ---------
        @param  :
        ---------
        @Returns  :
        ---------
        """

        return (x - mean) / (std + 1e-5)

    @staticmethod
    def compute_minibatch_mean_and_std(data_dict) -> Tuple:
        """
        @description  :  compute mean and std of data
        ---------
        @param  :
        ---------
        @Returns  :  mean, std
        ---------
        """

        mean = 0
        std = 0
        sample_num = 10
        for i in range(sample_num):
            temp_means = []
            temp_stds = []
            for k in data_dict:
                sample_index = np.random.randint(0, len(data_dict[k]), int(0.6 * len(data_dict[k])))
                for index in sample_index:
                    index_k_mean = np.mean(np.array(data_dict[k][index]), axis=0)
                    index_k_std = np.std(np.array(data_dict[k][index]), axis=0)
                    temp_means.append(index_k_mean)
                    temp_stds.append(index_k_std)

            mean = mean + np.mean(np.array(temp_means), axis=0)
            #         print('mean: ', np.mean(np.array(temp_means), axis=0))
            std = std + np.mean(np.array(temp_stds), axis=0)

        return (mean / sample_num, std / sample_num)