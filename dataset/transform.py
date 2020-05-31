import numpy as np
import sklearn.preprocessing
import sklearn.compose
import sklearn.pipeline
import sklearn.impute
import torch
import torchvision.transforms as transforms

import config
import dataset.utils

from sklearn.utils.validation import check_is_fitted


def get_transform(ds_type, train_dataset_path, train_mode=False, eval_mode=True):
    if ds_type == config.AdultIncome:
        return adult_income_transform(train_dataset_path)
    elif ds_type in config.UTKFace_lst:
        return utkface_transform(ds_type, train_mode, eval_mode)
    elif ds_type == config.LFW:
        return ImageTransform(train_mode, eval_mode, [0.49793946, 0.43451509, 0.38834113],
                              [0.11312092, 0.09601101, 0.09236771])
    elif ds_type in config.CelebA_lst:
        return ImageTransform(train_mode, eval_mode, [0.57372573, 0.48243514, 0.43426916],
                              [0.352059, 0.32908693, 0.32839223])
    elif ds_type in config.ColoredMNIST_lst:
        return ImageTransform(train_mode, eval_mode, [0.14808187, 0.07396729, 0.07396729],
                              [0.34918884, 0.2576412, 0.2576412])
    elif ds_type in config.Timit_lst:
        return lambda x: torch.Tensor(x)

    else:
        return None

def adult_income_transform(train_dataset_path):
    df_raw = dataset.utils.get_adult_income_raw_dataset(train_dataset_path)
    preprocessor = AdultIncomeTransform()
    preprocessor.fit(df_raw)
    return preprocessor


def utkface_transform(ds_type, train_mode, eval_mode):
    if ds_type == config.UTKFace_Full:
        return ImageTransform(train_mode, eval_mode,
                              [0.67552375, 0.517558, 0.443263], [0.2939831, 0.26247376, 0.25761761])
    elif ds_type == config.UTKFace_BW:
        return ImageTransform(train_mode, eval_mode,
                              [0.66550432, 0.50883236, 0.43741927], [0.28910463, 0.25760034, 0.25436434])
    elif ds_type == config.UTKFace_BW_balanced:
        return ImageTransform(train_mode, eval_mode,
                              [0.66710732, 0.50925283, 0.43707259], [0.28807441, 0.256671, 0.2530223])


class ImageTransform:
    def __init__(self, train_mode, eval_mode, img_mean, img_std):
        self.train_mode = train_mode
        self.eval_mode = eval_mode
        self.img_mean = img_mean
        self.img_std = img_std
        self.transform = None

    def __call__(self, img):
        width, height = img.size
        if self.transform is None:
            self.transform = self.create_transforms(width, height)
        return self.transform(img)

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.transform) + ')'

    def create_transforms(self, img_width, img_height):
        pil_trans = []
        tensor_trans = []
        if self.train_mode and config.args.activate_transformations:
            if config.args.transform_train_rotate_prob != 0:
                tran = transforms.RandomRotation(config.args.transform_train_rotate_angle)
                self.add_random_apply_transformation(pil_trans, tran, config.args.transform_train_rotate_prob)
            if config.args.transform_train_crop_prob != 0:
                tran = transforms.RandomCrop(config.args.transform_train_crop_size)
                self.add_random_apply_transformation(pil_trans, tran, config.args.transform_train_crop_prob)
            if config.args.transform_train_rectangle_erasing_prob != 0:
                tensor_trans.append(transforms.RandomErasing(config.args.transform_train_rectangle_erasing_prob))
            if config.args.transform_train_gaussian_noise_std != 0:
                noise_std = config.args.transform_train_gaussian_noise_std
                tensor_trans.append(transforms.Lambda(lambda x: x + torch.normal(0.0, noise_std, size=x.shape)))
        elif self.eval_mode and config.args.activate_transformations:
            if config.args.transform_test_rotate_prob != 0:
                tran = transforms.RandomRotation(config.args.transform_test_rotate_angle)
                self.add_random_apply_transformation(pil_trans, tran, config.args.transform_test_rotate_prob)
            if config.args.transform_test_crop_prob != 0:
                tran = transforms.RandomCrop(config.args.transform_test_crop_size)
                self.add_random_apply_transformation(pil_trans, tran, config.args.transform_test_crop_prob)
            if config.args.transform_test_rectangle_erasing_prob != 0:
                tensor_trans.append(transforms.RandomErasing(config.args.transform_test_rectangle_erasing_prob))
            if config.args.transform_test_gaussian_noise_std != 0:
                noise_std = config.args.transform_test_gaussian_noise_std
                tensor_trans.append(transforms.Lambda(lambda x: x + torch.normal(0.0, noise_std, size=x.shape)))
        pil_trans.append(transforms.Resize((img_width, img_height)))

        return transforms.Compose([*pil_trans,
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=self.img_mean, std=self.img_std),
                                   *tensor_trans])

    @staticmethod
    def add_random_apply_transformation(transformation_list, transformation, p):
        transformation_list.append(transforms.RandomApply([transformation], p=p))


class AdultIncomeTransform(sklearn.compose.ColumnTransformer):
    def __init__(self):
        numeric_cols = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
        categorial_cols = ['workclass', 'education', 'marital-status', 'occupation',
                           'relationship', 'race', 'native-country']
        categorial_binary_cols = ['sex', 'income']
        passthrough_cols = ['fnlwgt']

        numeric_transformer = sklearn.pipeline.Pipeline(steps=[
            ('imputer', sklearn.impute.SimpleImputer(strategy='median')),
            ('scaler', sklearn.preprocessing.StandardScaler())])

        categorical_encode_transformer = sklearn.pipeline.Pipeline(steps=[
            ('imputer', sklearn.impute.SimpleImputer(strategy='most_frequent')),
            ('encoder', sklearn.preprocessing.OrdinalEncoder())])

        categorical_onehot_transformer = sklearn.pipeline.Pipeline(steps=[
            ('imputer', sklearn.impute.SimpleImputer(strategy='most_frequent')),
            ('onehot', sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore'))])

        super().__init__(transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('passthrough', 'passthrough', passthrough_cols),
            ('cat_encode', categorical_encode_transformer, categorial_binary_cols),
            ('cat_onehot', categorical_onehot_transformer, categorial_cols)])

    def get_feature_names(self):
        check_is_fitted(self, 'transformers_')

        categorial_cols = self.named_transformers_.cat_onehot.named_steps.onehot.get_feature_names()
        categorial_cols = np.array([s[3:] for s in categorial_cols])
        return np.concatenate((self._columns[0], self._columns[1], self._columns[2], categorial_cols))
