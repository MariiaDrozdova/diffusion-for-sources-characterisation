import os
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, Subset, DataLoader, ConcatDataset
from torchvision import transforms

class SkaDataset(Dataset):
    DIRTY_NOISY_WO_PROCESSING = "dirty"
    TRUE_WO_PROCESSING = "true"

    def __init__(self, folder_images, image_size, power, from_uv):
        self.path_images = folder_images
        self.from_uv = from_uv
        self.power = power
        self.file_list = sorted(os.listdir(os.path.join(folder_images, self.DIRTY_NOISY_WO_PROCESSING)))
        self._set_folders(folder_images)
        self.test_data = False
        self.im_size = image_size
        self.augment = self._get_transforms(image_size)
        self.augment_test = self._get_transforms(image_size)
        self.train = True

    def _set_folders(self, folder_images):
        self.dirty_noisy_wo_processing_folder = os.path.join(folder_images, self.DIRTY_NOISY_WO_PROCESSING)
        self.true_wo_processing_folder = os.path.join(folder_images, self.TRUE_WO_PROCESSING)

    def _get_transforms(self, image_size):
        return transforms.Compose([
            transforms.Resize((image_size,) * 2),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)  # images normalized to [-1, 1]
        ])

    def __len__(self):
        return len(self.file_list)

    def set_train(self, train=False):
        self.train = train

    def __getitem__(self, index: int):
        file_name = self.file_list[index]

        # loading true sources distribution
        # if not available, loads zeros
        true_file = os.path.join(self.true_wo_processing_folder, file_name)

        if os.path.exists(true_file):
            true = np.load(true_file)[np.newaxis, ...]
        else:
            true = np.zeros((1, 512, 512))

        # cutting the circle, the same one we have in dirty images from CASA
        #mask = np.load("dirty_mask.npy").reshape(-1, 512, 512)
        #true = true * mask
        const = 0.00002960064
        true=true/const
        true = np.abs(true)
        true = (true) ** (1. / self.power)
        true = (true - 0.5) / 0.5

        data = {}
        data["filename"] = index

        sky_model = torch.from_numpy(true).float()

        def open_dirty_noisy(file_name):
            dirty_noisy_file = os.path.join(self.dirty_noisy_wo_processing_folder, file_name)
            dirty_noisy = np.load(dirty_noisy_file)[np.newaxis, ...]
            dirty_noisy = np.nan_to_num(dirty_noisy)
            return dirty_noisy

        dirty_noisy = open_dirty_noisy(file_name)

        dirty_noisy = dirty_noisy / 5e-4#3.0e-5

        dirty_noisy = (dirty_noisy - 0.5) / 0.5
        dirty_noisy = torch.from_numpy(dirty_noisy).float()

        data["true"] = torch.nn.functional.interpolate(
            sky_model.unsqueeze(0),
            mode="bicubic",
            size=self.im_size)[0]

        data["dirty_noisy"] = torch.nn.functional.interpolate(
            dirty_noisy.unsqueeze(0),
            mode="bicubic",
            size=self.im_size)[0]

        return data


class MakeDataLoader:
    """Class that creates train, valid and test datasets/dataloaders"""
    def __init__(self, folder_images, image_size, test_size=0.5,
                random_state=2, augmented=True, real_data=False, power=10,
                from_uv=False, use_zeros=False):

        print("dataset ", power, from_uv, use_zeros)
        self.dataset = SkaDataset(folder_images, image_size, power=power, from_uv=from_uv)
        if not augmented:
            self.dataset.test_data = True
        if real_data:
            train_idx = []
            valid_idx = []
            test_idx = list(range(len(self.dataset)))

        else:
            train_idx, test_idx = train_test_split(list(range(len(self.dataset))), test_size=test_size,
                                                   random_state=random_state)
            valid_idx, test_idx = train_test_split(test_idx, test_size=0.5, random_state=random_state + 1)

        self.dataset_train = Subset(self.dataset, np.array(train_idx))

        # in our dataset observations of the sky with zero sources are in the separate folder
        # just put use_zeros=False if it is not the case for you
        if not use_zeros:
            dataset_zero = SkaDataset(f"{folder_images}/zero", image_size, power=power, from_uv=from_uv)

            print(len(self.dataset_train))
            self.dataset_train = ConcatDataset([self.dataset_train, dataset_zero])
        print(len(self.dataset_train))
        set1 = set(train_idx)
        set2 = set(valid_idx)

        intersection = set1.intersection(set2)

        if intersection:
            print("The lists intersect.")
            print("Common elements:", intersection)
            raise KeyError("Training and validation sets are intersecting.")
        else:
            print("The lists do not intersect.")
        self.dataset_valid = Subset(self.dataset, valid_idx)
        self.dataset_test = Subset(self.dataset, test_idx)

    def get_data_loader_full(self,
                             batch_size: int = 64,
                             shuffle: bool = True, **kwargs) -> DataLoader:
        return DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True,
                          pin_memory=False, **kwargs)

    def get_data_loader_train(self,
                              batch_size: int = 64,
                              shuffle: bool = True, **kwargs) -> DataLoader:
        return DataLoader(self.dataset_train, batch_size=batch_size, shuffle=shuffle, drop_last=True,
                          pin_memory=False, **kwargs)

    def get_data_loader_test(self,
                             batch_size: int = 64,
                             shuffle: bool = False, **kwargs) -> DataLoader:
        return DataLoader(self.dataset_test, batch_size=batch_size, shuffle=shuffle, drop_last=True,
                          pin_memory=False, **kwargs)

    def get_data_loader_valid(self, batch_size: int = 64,
                              shuffle: bool = False,
                              **kwargs) -> DataLoader:
        return DataLoader(self.dataset_valid, batch_size=batch_size, shuffle=shuffle, drop_last=True,
                          pin_memory=False, **kwargs)
