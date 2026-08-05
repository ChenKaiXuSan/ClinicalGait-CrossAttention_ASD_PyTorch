#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/skeleton/project/cross_validation.py
Project: /workspace/skeleton/project
Created Date: Friday March 22nd 2024
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Thursday May 1st 2025 8:34:05 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2024 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------

22-03-2024	Kaixu Chen	add different class number mapping, and add the cross validation process.
"""


import os, json, shutil, copy, random, time
from typing import Any, Dict, List, Tuple

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


from sklearn.model_selection import StratifiedGroupKFold, train_test_split, GroupKFold
from pathlib import Path

class_num_mapping_Dict: Dict = {
    2: {0: "ASD", 1: "non-ASD"},
    3: {0: "ASD", 1: "DHS", 2: "LCS_HipOA"},
    4: {0: "ASD", 1: "DHS", 2: "LCS_HipOA", 3: "normal"},
}


class DefineCrossValidation(object):
    """process:
    cross validation > over/under sampler > train/val split
    fold: [train/val]: [path]
    """

    def __init__(self, config) -> None:

        self.video_path: Path = Path(config.data.data_info_path)  # json file path
        self.gait_seg_idx_path: Path = Path(
            config.data.index_mapping
        )  # used for training path mapping

        self.K: int = config.train.fold
        self.sampler: str = config.data.sampling  # data balance, [over, under, none]

        # Clean-protocol switch: patient-grouped train/val/test with a TRUE
        # held-out test (no magic_move, holdout never over-sampled). Off by
        # default to preserve the original cache/behaviour.
        self.heldout: bool = bool(config.data.get("heldout_test", False))

        # cache dir name includes K (and the protocol), so changing train.fold or
        # the protocol never silently reuses a split built differently.
        self.cache_name: str = f"{self.sampler}_K{self.K}" + ("_heldout" if self.heldout else "")

        self.class_num: int = config.model.model_class_num
        self.clip_duration: int = config.train.clip_duration

        self.raw_video_path = config.data.video_path

    @staticmethod
    def random_sampler(X: list, y: list, train_idx: list, val_idx: list, sampler):
        # train
        train_mapped_path = []
        new_X_path = [X[i] for i in train_idx]

        sampled_X, sampled_y = sampler.fit_resample(
            [[i] for i in range(len(new_X_path))], [y[i] for i in train_idx]
        )

        # map sampled_X to new_X_path
        for i in sampled_X:
            train_mapped_path.append(new_X_path[i[0]])

        # val
        val_mapped_path = []
        new_X_path = [X[i] for i in val_idx]

        sampled_X, sampled_y = sampler.fit_resample(
            [[i] for i in range(len(new_X_path))], [y[i] for i in val_idx]
        )

        # map
        for i in sampled_X:
            val_mapped_path.append(new_X_path[i[0]])

        return train_mapped_path, val_mapped_path

    @staticmethod
    def _oversample_one(X, y, idx, sampler):
        """Resample a SINGLE index set (used for the held-out protocol so that
        only the inner-train split is balanced; val/test stay natural)."""
        paths = [X[i] for i in idx]
        sampled_X, _ = sampler.fit_resample(
            [[i] for i in range(len(paths))], [y[i] for i in idx]
        )
        return [paths[i[0]] for i in sampled_X]

    def process_cross_validation(self, video_dict: dict) -> Tuple[List, List, List]:

        _path = video_dict

        X = []  # patient index
        y = []  # patient class index
        groups = []  # different patient groups

        disease_to_num = {
            disease: idx
            for idx, disease in class_num_mapping_Dict[self.class_num].items()
        }
        element_to_num = {}

        name_map = set()

        # process one disease in one loop.
        for disease, path in _path.items():
            patient_list = sorted(list(path))

            for p in patient_list:
                name, _ = p.name.split("-")
                #  FIXME: 我觉得HipOA是造成数据不平衡的原因，所以我把HipOA的数据去掉了
                if "HipOA" not in name:
                    name_map.add(name)

        for idx, element in enumerate(name_map):
            element_to_num[element] = idx

        for disease, path in _path.items():
            patient_list = sorted(list(path))
            for i in range(len(patient_list)):

                name, _ = patient_list[i].name.split("-")

                label = disease_to_num[disease]

                # FIXME: 我举得HipOA是造成数据不平衡的原因，所以我把HipOA的数据去掉了
                if "HipOA" not in name:
                    X.append(patient_list[i])  # true path in Path
                    y.append(label)  # label, 0, 1, 2
                    groups.append(element_to_num[name])  # number of different patient

        return X, y, groups

    def make_dataset_with_video(self, val_dataset_idx: list, fold: int, flag: str):
        temp_path = (
            self.gait_seg_idx_path
            / str(self.class_num)
            / self.cache_name
            / str(fold)
            / str(flag)
        )
        val_idx = val_dataset_idx

        _class_map = class_num_mapping_Dict[self.class_num]
        _disease_to_num = {disease: idx for idx, disease in _class_map.items()}

        shutil.rmtree(temp_path, ignore_errors=True)

        for path in val_idx:
            with open(path) as f:
                file_info_dict = json.load(f)

            video_name = file_info_dict["video_name"]
            # * change the video path to fit the different server.
            file_info_dict["video_path"] = (
                self.raw_video_path
                + "/"
                + "/".join(file_info_dict["video_path"].split("/")[-4:])
            )
            video_path = file_info_dict["video_path"]
            video_disease = file_info_dict["disease"]

            if video_disease not in _disease_to_num.keys():
                video_disease = "non-ASD"

            if not (temp_path / video_disease).exists():
                (temp_path / video_disease).mkdir(parents=True, exist_ok=False)

            shutil.copy(video_path, temp_path / video_disease / (video_name + ".mp4"))

            # update the json file with the video path
            with open(path, "w") as f:
                json.dump(file_info_dict, f, indent=4)

        return temp_path

    @staticmethod
    def magic_move(train_mapped_path, val_mapped_path):

        new_train_mapped_path = copy.deepcopy(train_mapped_path)
        new_val_mapped_path = copy.deepcopy(val_mapped_path)

        # train magic
        train_tmp_dict = {}
        for i in train_mapped_path:
            # not move ASD
            if "ASD" in i.name:
                continue

            train_tmp_dict[i.name.split("-")[0]] = i

        val_tmp_dict = {}
        for i in val_mapped_path:
            # not move ASD
            if "ASD" in i.name:
                continue
            val_tmp_dict[i.name.split("-")[0]] = i

        for k, v in train_tmp_dict.items():
            new_val_mapped_path.append(v)

            rm_idx = new_train_mapped_path.index(v)
            new_train_mapped_path.pop(rm_idx)

        for k, v in val_tmp_dict.items():
            new_train_mapped_path.append(v)

            rm_idx = new_val_mapped_path.index(v)
            new_val_mapped_path.pop(rm_idx)

        return new_train_mapped_path, new_val_mapped_path

    @staticmethod
    def map_class_num(class_num: int, raw_video_path: Path) -> Dict:

        _class_num = class_num_mapping_Dict[class_num]

        res_dict = {v: [] for k, v in _class_num.items()}

        for disease in raw_video_path.iterdir():

            for one_json_file in disease.iterdir():

                if disease.name in res_dict.keys():
                    res_dict[disease.name].append(one_json_file)
                elif disease.name == "log":
                    continue
                else:
                    res_dict["non-ASD"].append(one_json_file)

        return res_dict

    def prepare(self):
        """define cross validation first, with the K.
        #! the 1 fold and K fold should return the same format.
        fold: [train/val]: [path]

        Args:
            video_path (str): the index of the video path, in .json format.
            K (int, optional): crossed number of validation. Defaults to 5, can be 1 or K.

        Returns:
            list: the format like upper.
        """
        K = self.K

        ans_fold = {}

        mapped_class_Dict = self.map_class_num(self.class_num, self.video_path)

        # define the cross validation
        # X: video path, in path.Path foramt. len = 1954
        # y: label, in list format. len = 1954, type defined by class_num_mapping_Dict.
        # groups: different patient, in list format. It means unique patient index. [54]
        X, y, groups = self.process_cross_validation(mapped_class_Dict)

        sgkf = StratifiedGroupKFold(n_splits=K)

        if self.heldout:
            # ---- clean protocol: patient-grouped train / val / test ----------
            # outer split -> patient-disjoint (train_index, test_index); test is a
            # TRUE held-out set (natural, never over-sampled, no magic_move). An
            # inner StratifiedGroupKFold carves a patient-grouped val set out of
            # train for early stopping. Only inner-train is over/under-sampled.
            for i, (train_index, test_index) in enumerate(
                sgkf.split(X=X, y=y, groups=groups)
            ):
                tr_y = [y[j] for j in train_index]
                tr_g = [groups[j] for j in train_index]
                inner = StratifiedGroupKFold(n_splits=K)
                rel_tr, rel_val = next(iter(inner.split(train_index, tr_y, tr_g)))
                inner_train_index = [train_index[j] for j in rel_tr]
                inner_val_index = [train_index[j] for j in rel_val]

                if self.sampler == "over":
                    train_paths = self._oversample_one(
                        X, y, inner_train_index, RandomOverSampler(random_state=42))
                elif self.sampler == "under":
                    train_paths = self._oversample_one(
                        X, y, inner_train_index, RandomUnderSampler(random_state=42))
                else:
                    train_paths = [X[j] for j in inner_train_index]
                val_paths = [X[j] for j in inner_val_index]     # natural
                test_paths = [X[j] for j in test_index]         # natural, held out
                # NB: no magic_move -> patient disjointness of the outer split is
                # preserved (train/val/test share no patient).
                ans_fold[i] = [train_paths, val_paths, test_paths]
            return ans_fold, X, y, groups

        for i, (train_index, test_index) in enumerate(
            sgkf.split(X=X, y=y, groups=groups)
        ):
            if self.sampler in ["over", "under"]:
                if self.sampler == "over":
                    ros = RandomOverSampler(random_state=42)
                elif self.sampler == "under":
                    ros = RandomUnderSampler(random_state=42)

                train_mapped_path, val_mapped_path = self.random_sampler(
                    X, y, train_index, test_index, ros
                )

            else:
                train_mapped_path = [X[i] for i in train_index]
                val_mapped_path = [X[i] for i in test_index]

            # FIXME: magic move
            train_mapped_path, val_mapped_path = self.magic_move(
                train_mapped_path, val_mapped_path
            )

            # TODO: here merge the multi info into one .pt file.

            # make the val data path
            train_video_path = self.make_dataset_with_video(
                train_mapped_path, i, "train"
            )
            val_video_path = self.make_dataset_with_video(val_mapped_path, i, "val")

            # * here used for gait labeled method, or load video from path
            ans_fold[i] = [
                train_mapped_path,
                val_mapped_path,
                train_video_path,
                val_video_path,
            ]

        return ans_fold, X, y, groups

    def __call__(self, *args: Any, **kwds: Any) -> Any:

        target_path = self.gait_seg_idx_path / str(self.class_num) / self.cache_name
        index_file = target_path / "index.json"

        # * when json file changed, need to reprocess the dataset
        # * (delete target_path to force a rebuild).
        if not index_file.exists():
            target_path.parent.mkdir(parents=True, exist_ok=True)

            # atomic mkdir acts as a build lock: with parallel per-fold PBS
            # array jobs, only the first one builds the cache, the others wait.
            try:
                target_path.mkdir(exist_ok=False)
                is_builder = True
            except FileExistsError:
                is_builder = False

            if is_builder:
                fold_dataset_idx, *_ = self.prepare()

                json_fold_dataset_idx = copy.deepcopy(fold_dataset_idx)

                for k, v in fold_dataset_idx.items():
                    if self.heldout:
                        # [0/1/2]: train/val/test mapping paths (gait cycle index)
                        json_fold_dataset_idx[k][0] = [str(i) for i in v[0]]
                        json_fold_dataset_idx[k][1] = [str(i) for i in v[1]]
                        json_fold_dataset_idx[k][2] = [str(i) for i in v[2]]
                    else:
                        # [0/1]: train/val mapping path (include the gait cycle index)
                        # [2/3]: train/val video path
                        json_fold_dataset_idx[k][0] = [str(i) for i in v[0]]
                        json_fold_dataset_idx[k][1] = [str(i) for i in v[1]]
                        json_fold_dataset_idx[k][2] = str(v[2])
                        json_fold_dataset_idx[k][3] = str(v[3])

                # write via tmp file + atomic rename, so waiters never load a
                # partially written index.json.
                tmp_index = target_path / "index.json.tmp"
                with open(tmp_index, "w") as f:
                    json.dump(json_fold_dataset_idx, f, sort_keys=True, indent=4)
                os.replace(tmp_index, index_file)
            else:
                waited = 0
                while not index_file.exists():
                    time.sleep(30)
                    waited += 30
                    if waited >= 4 * 3600:
                        raise RuntimeError(
                            f"Timed out waiting for fold cache: {index_file}. "
                            f"If a previous build crashed, delete {target_path} and rerun, "
                            "or pre-build once with: python -m project.prepare_folds"
                        )

        with open(index_file, "r") as f:
            fold_dataset_idx = json.load(f)

        # unpack: json strings back to Path. Keys are always str here.
        for k, v in fold_dataset_idx.items():
            if self.heldout:
                fold_dataset_idx[k][0] = [Path(i) for i in v[0]]
                fold_dataset_idx[k][1] = [Path(i) for i in v[1]]
                fold_dataset_idx[k][2] = [Path(i) for i in v[2]]
            else:
                fold_dataset_idx[k][0] = [Path(i) for i in v[0]]
                fold_dataset_idx[k][1] = [Path(i) for i in v[1]]
                fold_dataset_idx[k][2] = Path(v[2])
                fold_dataset_idx[k][3] = Path(v[3])

        return fold_dataset_idx
