import os

import numpy as np
import pandas as pd
import torch.utils.data

import config
import dataset.utils

from abc import ABCMeta, abstractmethod
from typing import Sequence, List, Union, Tuple
from PIL import Image

from dataset.transform import AdultIncomeTransform


class FairnessDataset(torch.utils.data.Dataset, metaclass=ABCMeta):
    train_default: str
    eval_default: str
    test_default: str

    def __init__(self, ds_path, ds_type, transform, weights=0.0):
        """
        Use weights = None to ignore weights, all weights are adjusted to be 1/n, where n = #groups.
        Use weights = 0.0 to get fair weights such that each group gets equal representation.
        Use weights > 0.0 to create weight bias towards the majority group, each other group is adjusted accordingly.
        Use weights = [x_1,...,x_n] to create custom weights, where n = #groups.
        :param ds_path:
        :param ds_type:
        :param transform:
        :param weights:
        """
        self.ds_path = ds_path
        self.ds_type = ds_type
        self.transform = transform
        self._data = None
        self.fairness_groups, self.fairness_groups_count = self._group_and_count(self.fairness_groups_column)
        self.label_groups, self.label_groups_count = self._group_and_count(self.label_column)
        self.groups, self.groups_count = self._group_and_count(self.groups_columns)
        self._weights = self.compute_weights(weights)

    @property
    def data(self) -> pd.DataFrame:
        """
        Data after preproccessing
        :return: data in pandas DataFrame format
        """
        if self._data is None:
            self._data = self._initialize_data()
        return self._data

    @property
    def feature_columns(self) -> Sequence[str]:
        return self.data.columns.drop(self.label_column).tolist()

    @property
    @abstractmethod
    def label_column(self) -> str:
        pass

    @property
    def label_count(self) -> int:
        return self.data[self.label_column].nunique()

    @property
    @abstractmethod
    def fairness_groups_column(self) -> str:
        pass

    @property
    def groups_columns(self):
        """
        Groups categories as list of attributes with the label attribute. e.g. ['gender', 'income'].
        :return: list of groups columns (strings)
        """
        return np.unique([self.fairness_groups_column, self.label_column]).tolist()

    @property
    def weights(self) -> np.ndarray:
        return self._weights

    def compute_weights(self, weights: Union[float, Sequence[float], None]) -> np.ndarray:
        """
        Compute weights per group (fairness group + label).
        Use weights = None to ignore weights, all weights are adjusted to be 1.
        Use weights = 0.0 to get fair weights such that each group gets equal representation.
        Use weights > 0.0 to create weight bias towards the majority group, each other group is adjusted accordingly.
        Use weights = [x_1,...,x_n] to create custom weights, where n = #groups.
        :param weights:
        :return:
        """
        groups_num = self.groups.shape[0]
        if weights is None:
            weights = np.ones(groups_num, dtype=np.float)
        # adds bias
        elif isinstance(weights, float) or len(weights) == 1:
            fairness_weights = self._compute_fair_group_weights()
            epsilon = weights if isinstance(weights, float) else weights[0]
            bias = np.zeros(groups_num, dtype=np.float)
            mask = np.ones(groups_num, dtype=bool)
            mask[np.argmax(fairness_weights)] = False
            bias[~mask] = epsilon
            bias[mask] -= epsilon / np.sum(mask)
            fairness_weights += bias
            fairness_weights = np.clip(fairness_weights, 0., 1.)
            weights = fairness_weights
        return np.array([weights[np.where(np.all(self.groups == row, axis=1))[0][0]] for row in
                         self.data[self.groups_columns].values])

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx) -> (torch.FloatTensor, torch.LongTensor, torch.Tensor, torch.Tensor):
        row = self.data.iloc[idx]
        features = torch.from_numpy(row[self.feature_columns].values).float()
        label = torch.tensor(row[self.label_column]).long()
        weight = torch.tensor(self.weights[idx])
        fairness_group = torch.tensor(row[self.fairness_groups_column])
        return features, label, weight, fairness_group

    @abstractmethod
    def _initialize_data(self) -> pd.DataFrame:
        pass

    def _compute_fair_group_weights(self) -> Sequence[float]:
        count_arr = self.groups_count / len(self)
        # begin a linear equations system, which will compute the weight of each group to have equal representation
        # e.g. for 2 fairness groups and 2 labels, the linear equations would be:
        # w0*x-c = 0, w1*y - c = 0, w2*z - c = 0, w_3(1-x-y-z) -c = 0
        lineq = np.diag(count_arr)
        lineq[-1] -= count_arr[-1]
        lineq[:, -1] = -1
        b = np.zeros(count_arr.shape)
        b[-1] = lineq[-1, 0]
        sol = np.linalg.solve(lineq, b)
        sol[-1] = 1 - np.sum(sol[:-1])
        sol = sol / np.sum(sol)
        return sol.tolist()

    def _group_and_count(self, columns) -> Tuple[np.ndarray, np.ndarray]:
        filtered_data = self.data[columns].values.astype(int)
        unique_val, val_counts = np.unique(filtered_data, axis=0, return_counts=True)
        return unique_val, val_counts


class ImageFairnessDataset(FairnessDataset, metaclass=ABCMeta):
    @abstractmethod
    def get_image_path(self, data_row) -> str:
        pass

    def __getitem__(self, idx) -> (torch.FloatTensor, torch.LongTensor, torch.Tensor, torch.Tensor):
        row = self.data.iloc[idx]
        img_path = self.get_image_path(row)
        img_tensor = self.transform(Image.open(img_path).convert('RGB'))
        label = torch.tensor(row[self.label_column]).long()
        weight = torch.tensor(self.weights[idx])
        fair_group = torch.tensor(row[self.fairness_groups_column]).long()
        return img_tensor, label, weight, fair_group


class UTKFaceDataset(ImageFairnessDataset):
    train_default = "train_set"
    eval_default = "eval_set"
    test_default = "eval_set"

    valid_extension = '.jpg'
    races = {0: "white", 1: "black", 2: "asian", 3: "indian", 4: "other"}
    genders = {0: "male", 1: "female"}
    _df_cols = {"image_name": str, "date": str, "age": int, "gender": int, "race": int}

    @property
    def label_column(self) -> str:
        return "gender"

    @property
    def fairness_groups_column(self) -> str:
        return "race"

    def _initialize_data(self) -> pd.DataFrame:
        utk_file_strs = os.listdir(self.ds_path)
        utk_lst = []
        for utk_str in utk_file_strs:
            utk_file, utk_extension = os.path.splitext(utk_str)
            face_pic_prop = utk_file.split('_')
            if utk_extension != self.valid_extension:
                config.log.info(f"file: {utk_str} is ignored due to invalid extension")
                continue
            if len(face_pic_prop) != 4:
                config.log.info(f"file: {utk_str} is ignored due to invalid format")
                continue
            age, gender, race, date = face_pic_prop
            date = date[:date.index('.')]
            utk_lst.append([utk_str, date, age, gender, race])
        return pd.DataFrame(utk_lst, columns=list(self._df_cols.keys())).astype(self._df_cols)

    def get_image_path(self, data_row) -> str:
        return os.path.join(self.ds_path, data_row.image_name)


class AdultIncomeDataset(FairnessDataset):
    train_default = "adult.data"
    eval_default = "adult.test"  # Non-exists, test=eval
    test_default = "adult.test"
    instance_weight = "fnlwgt"

    def __init__(self, ds_path, ds_type, transform: AdultIncomeTransform, weights):
        self.df_raw = None
        super().__init__(ds_path, ds_type, transform, weights)

    @property
    def label_column(self) -> str:
        return "income"

    @property
    def fairness_groups_column(self) -> str:
        return "sex"

    @property
    def feature_columns(self) -> Sequence[str]:
        return self.data.columns.drop([self.label_column, self.instance_weight]).tolist()

    def _initialize_data(self) -> pd.DataFrame:
        self.df_raw = dataset.utils.get_adult_income_raw_dataset(self.ds_path)
        data_arr = self.transform.transform(self.df_raw).toarray()
        return pd.DataFrame(data_arr, columns=self.transform.get_feature_names())


class SyntheticDataset(FairnessDataset):
    fairness_group = "group_feature3"
    label_col = "label"

    train_default = "fair_train_data.npy"
    eval_default = "fair_test_data.npy"  # Non-exists, test=eval
    test_default = "fair_test_data.npy"

    @property
    def label_column(self) -> str:
        return self.label_col

    @property
    def fairness_groups_column(self) -> str:
        return self.fairness_group

    @property
    def feature_columns(self) -> Sequence[str]:
        return self.data.columns.drop([self.label_column, self.fairness_groups_column]).tolist()

    def _initialize_data(self) -> pd.DataFrame:
        data_arr = np.load(self.ds_path)
        return pd.DataFrame(data_arr, columns=["feature1", "feature2", self.fairness_group, self.label_col])


class LFWDataset(ImageFairnessDataset):
    attributes_file = 'lfw_attributes.txt'
    images_subdir = "lfw"
    train_default = "train_indices.txt"
    eval_default = "test_indices.txt"  # Non-exists, test=eval
    test_default = "test_indices.txt"

    def __init__(self, ds_path, ds_type, transform, weights):
        self.indices_arr = np.loadtxt(ds_path, dtype=int)
        self.indices_dict = {i: v for i, v in enumerate(self.indices_arr.tolist())}
        ds_root_path, _ = os.path.split(ds_path)
        super().__init__(ds_root_path, ds_type, transform, weights)

    @property
    def label_column(self) -> str:
        return "Male"

    @property
    def fairness_groups_column(self) -> str:
        return "Black"

    def _initialize_data(self) -> pd.DataFrame:
        # load attribute file
        df = pd.read_csv(os.path.join(self.ds_path, self.attributes_file), delimiter='\t', skiprows=1)
        # binarize all attributes
        df[df.columns[2:]] = np.clip(np.sign(df[df.columns[2:]]), 0, 1)
        # fix extra column problem
        df_formatted = df.iloc[:, : -1]
        df_formatted.columns = df.columns[1:]
        # create image name column
        df_formatted["person"] = df_formatted.person.str.replace(' ', '_')
        df_formatted["image_name"] = df_formatted.person + '_' + df_formatted.imagenum.astype(str).str.zfill(4) + ".jpg"
        return df_formatted.iloc[self.indices_arr]

    def get_image_path(self, data_row) -> str:
        return os.path.join(self.ds_path, self.images_subdir, data_row.person, data_row.image_name)


class TimitDataset(FairnessDataset):
    train_default = "train"
    eval_default = "test"
    test_default = "test"

    sample_loc_column = 'FileLoc'
    sample_split_column = 'SplitNum'

    metadata_processed_filename = 'timit_{timit_type}_{rows_slice}_metadata.csv'
    metadata_filename = 'SPKRINFO.TXT'
    metadata_id = 'ID'
    metadata_cols = [metadata_id, 'Sex', 'DR', 'Use', 'RecDate', ' BirthDate', 'Ht', 'Race', 'Edu', 'Comments']
    metadata_dr_exclude = [8]
    metadata_dr_mapping = {6: 1, 3: 2, 5: 4}

    row_count_in_features = 100

    @property
    def label_column(self) -> str:
        return "Sex"

    @property
    def fairness_groups_column(self) -> str:
        return "DR"

    def _initialize_data(self) -> pd.DataFrame:
        filename = self.metadata_processed_filename.format(timit_type=self.ds_type,
                                                           rows_slice=self.row_count_in_features)
        filepath = os.path.join(self.ds_path, filename)
        if not os.path.isfile(os.path.join(self.ds_path, filename)):
            metadata = pd.read_csv(os.path.join(self.ds_path, self.metadata_filename), comment=';', delimiter="  ",
                                   index_col=self.metadata_id, names=self.metadata_cols, engine="python")
            # add sample location and populate data
            columns = metadata.columns.to_list()
            columns.extend([self.sample_loc_column, self.sample_split_column])
            data = pd.DataFrame(columns=columns)
            for root, subdirs, filenames in os.walk(self.ds_path):
                for filename in filenames:
                    if os.path.splitext(filename)[1] != ".scores":
                        continue
                    features_row_num = 1
                    file_path = os.path.join(root, filename)
                    with open(file_path, 'r') as f:
                        line = f.readline()
                        features_row_num = int(line.split()[0]) // self.row_count_in_features
                    idx = os.path.split(root)[1][-4:].upper()
                    for i in range(features_row_num):
                        metadata_row: pd.Series = metadata.loc[idx].copy()
                        metadata_row[self.sample_loc_column] = file_path
                        metadata_row[self.sample_split_column] = i
                        metadata_row = metadata_row.rename(f"{metadata_row.name}_{os.path.splitext(filename)[0]}_{i}")
                        data = data.append(metadata_row)
            # alter DR mapping
            data = data[~data["DR"].isin(self.metadata_dr_exclude)]
            for k, v in self.metadata_dr_mapping.items():
                data.loc[data["DR"] == k, "DR"] = v
            if self.ds_type == config.Timit2Groups:
                data = data[data["DR"].isin([1, 2])]
            data[self.label_column] = pd.factorize(data[self.label_column])[0]
            # save file
            data.to_csv(filepath, index=False)
        data = pd.read_csv(filepath)
        return data

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        split_idx = row[self.sample_split_column]
        file_path = row[self.sample_loc_column]
        features_row = self.transform(np.loadtxt(file_path, skiprows=1))
        features = features_row[split_idx * self.row_count_in_features:(split_idx + 1) * self.row_count_in_features]
        label = torch.tensor(row[self.label_column]).long()
        weight = torch.tensor(self.weights[idx])
        fairness_group = torch.tensor(row[self.fairness_groups_column]).long()
        return features.unsqueeze(0), label, weight, fairness_group


class CelebADataset(ImageFairnessDataset):
    metadata_file = 'celeba_metadata.csv'
    images_subdir = "img_align_celeba"
    train_default = "train"
    eval_default = "test"  # Non-exists, test=eval
    test_default = "test"

    caucasian_id = 0

    def __init__(self, ds_path, ds_type, transform, weights):
        self.root_dir, self.mode = os.path.split(ds_path)
        self.metadata = pd.read_csv(os.path.join(self.root_dir, self.metadata_file))
        super().__init__(ds_path, ds_type, transform, weights)

    @property
    def label_column(self) -> str:
        return "Male"

    @property
    def fairness_groups_column(self) -> str:
        return "label"

    def _initialize_data(self) -> pd.DataFrame:
        # get train or test rows
        df = self.metadata[self.metadata['train'] == int(self.mode == self.train_default)].reset_index(drop=True)
        if self.ds_type == config.CelebA2Groups:
            # unify non-white groups
            df.loc[df['label'] != 0, 'label'] = 1
        return df

    def get_image_path(self, data_row) -> str:
        return os.path.join(self.root_dir, self.images_subdir, data_row.img_id)


class ColoredMNISTDataset(ImageFairnessDataset):
    train_default = "train"
    eval_default = "test"
    test_default = "test"

    fairness_group_metadata = "colors_idx.csv"
    colors = {0: "white", 1: "red"}
    _df_cols = {"color": int, "label": int}

    @property
    def label_column(self) -> str:
        return "label"

    @property
    def fairness_groups_column(self) -> str:
        return "color"

    def _initialize_data(self) -> pd.DataFrame:
        df = pd.read_csv(os.path.join(self.ds_path, self.fairness_group_metadata)).astype(self._df_cols)
        if self.ds_type == config.ColoredMNIST2Groups:
            mask = df[self.label_column] <= 4
            df.loc[mask, 'label'] = 0
            df.loc[~mask, 'label'] = 1
        return df

    def get_image_path(self, data_row) -> str:
        return os.path.join(self.ds_path, f"{data_row.name}.png")
