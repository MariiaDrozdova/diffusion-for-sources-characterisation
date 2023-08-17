from typing import Tuple
import argparse

import numpy as np

from sklearn.manifold import TSNE

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from chamferdist import ChamferDistance
from geomloss import SamplesLoss

from src.model import ResNetSimCLR
from src.data import MakeDataLoader
from src.metrics import get_fid_between_datasets, inception_score
from src.metrics.fid import load_patched_inception_v3
from src.utils import calculate_frechet_distance


# encoder parameters
SIMCLR_PATH = './models/galaxy_zoo_simclr.pth'
ENCODER_DIM = 128
BASE_MODEL = 'resnet50'


class DatasetFromNumpy(Dataset):
    def __init__(self, data: np.ndarray, labels: np.ndarray, return_labels: bool = True):
        self.data = data
        self.labels = labels
        self.return_labels = return_labels

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)
        ])

    def __getitem__(self, index):
        lbl = self.labels[index]
        img = self.data[index]
        img = self.transform(img)

        if self.return_labels:
            return img, lbl

        return img

    def __len__(self):
        return len(self.data)


class Evaluator:

    def __init__(self, path_original_images: str, path_original_labels: str,
                 imgs_gen: np.ndarray, labels_gen: np.ndarray,
                 device: str, batch_size: int = 64):
        self.imgs_gen = imgs_gen
        self.labels_gen = labels_gen

        n_channels = imgs_gen.shape[-1]
        img_size = imgs_gen.shape[1]

        self.device = device
        self.batch_size = batch_size

        self.make_dl = MakeDataLoader(path_original_images, path_original_labels, img_size)

        # load SimCLR encoder
        self.encoder = ResNetSimCLR(BASE_MODEL, n_channels, ENCODER_DIM).to(self.device)
        self.encoder.load_state_dict(torch.load(SIMCLR_PATH, map_location=self.device))
        self.encoder.eval()

    @torch.no_grad()
    def evaluate(self) -> None:
        fid_orig = self._compute_fid_score()
        print(f'FID score for original images: {fid_orig}')

        is_mean, is_std = self._compute_inception_score()
        print(f'Inception score for original images: {is_mean} +- {is_std}')

        chamfer_dist = self._compute_chamfer_distance()
        print(f'Chamfer distance for generated images: {chamfer_dist}')

        fid_ssl = self._compute_ssl_fid()
        print(f'FID score with SSL encoder: {fid_ssl}')

        kid = self._compute_inception_kid()
        print(f'KID score for generated images: {kid}')

        kid_ssl = self._compute_ssl_kid()
        print(f'KID score with SSL encoder: {kid_ssl}')

        wasserstein = self._compute_geometric_distance()
        print(f'Wasserstein distance for generated images: {wasserstein}')

    @torch.no_grad()
    def _compute_fid_score(self) -> float:
        """Computes original FID score with the Inception v3 model

        Returns:
            float: FID score
        """

        ds_gen = DatasetFromNumpy(self.imgs_gen, self.labels_gen)
        fid = get_fid_between_datasets(ds_gen, self.make_dl.dataset_test, self.device, self.batch_size, len(ds_gen))
        return fid

    @torch.no_grad()
    def _compute_inception_score(self) -> Tuple[float, float]:
        """Computes inception score

        Returns:
            float: inception score
        """

        ds = DatasetFromNumpy(self.imgs_gen, self.labels_gen, return_labels=False)
        score_mean, score_std = inception_score(ds, True, self.batch_size, True, 3)
        return score_mean, score_std

    @torch.no_grad()
    def _compute_chamfer_distance(self) -> float:
        """Computes chamfer distance

        Returns:
            float: chamfer distance
        """

        ds_gen = DatasetFromNumpy(self.imgs_gen, self.labels_gen, return_labels=False)
        dl_gen = DataLoader(ds_gen, batch_size=self.batch_size, shuffle=False, num_workers=4, drop_last=True)
        dl_real = self.make_dl.get_data_loader_test(self.batch_size, shuffle=False)
        n_batches = len(dl_gen)

        embeddings = []
        i = 0
        for (img, _) in dl_real:
            img = img.to(self.device)
            with torch.no_grad():
                h, _ = self.encoder(img)
            embeddings.append(h.cpu().numpy())

            i += 1
            if i == n_batches:
                break

        for img in dl_gen:
            img = img.to(self.device)
            with torch.no_grad():
                h, _ = self.encoder(img)
            embeddings.append(h.cpu().numpy())

        embeddings = np.concatenate(embeddings, axis=0)
        tsne_emb = TSNE(n_components=3).fit_transform(embeddings)
        n = len(tsne_emb)
        tsne_real = tsne_emb[:n // 2]
        tsne_gen = tsne_emb[n // 2:]

        tsne_real = torch.from_numpy(tsne_real).unsqueeze(0)
        tsne_fake = torch.from_numpy(tsne_gen).unsqueeze(0)

        chamfer_dist = ChamferDistance()
        return chamfer_dist(tsne_real, tsne_fake).detach().item()

    @torch.no_grad()
    def _compute_ssl_fid(self) -> float:
        """Computes FID score with the SimCLR encoder

        Returns:
            float: FID score
        """

        dl_real = self.make_dl.get_data_loader_test(self.batch_size, shuffle=False)
        dl_gen = DataLoader(DatasetFromNumpy(self.imgs_gen, self.labels_gen, return_labels=False),
                            batch_size=self.batch_size, shuffle=False, num_workers=4, drop_last=True)

        features_real = []
        features_gen = []

        for img, _ in dl_real:
            img = img.to(self.device)
            with torch.no_grad():
                h, _ = self.encoder(img)
            features_real.append(h.cpu().numpy())

        for img in dl_gen:
            img = img.to(self.device)
            with torch.no_grad():
                h, _ = self.encoder(img)
            features_gen.append(h.cpu().numpy())

        features_real = np.concatenate(features_real, axis=0)
        features_gen = np.concatenate(features_gen, axis=0)

        mu_real, sigma_real = np.mean(features_real, axis=0), np.cov(features_real, rowvar=False)
        mu_gen, sigma_gen = np.mean(features_gen, axis=0), np.cov(features_gen, rowvar=False)

        fid = calculate_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)
        return fid

    @torch.no_grad()
    def _compute_inception_kid(self) -> float:
        """Computes Kernel Inception Distance using features computed using pretrained InceptionV3

        Returns:
            float: inception score
        """

        encoder = load_patched_inception_v3().to(self.device).eval()

        dl_real = self.make_dl.get_data_loader_test(self.batch_size, shuffle=False)
        dl_gen = DataLoader(DatasetFromNumpy(self.imgs_gen, self.labels_gen, return_labels=False),
                            batch_size=self.batch_size, shuffle=False, num_workers=4, drop_last=True)

        feat_real = []
        for img, _ in dl_real:
            img = img.to(self.device)
            img = torch.nn.functional.interpolate(img, size=(299, 299), mode='bicubic')
            with torch.no_grad():
                h = encoder(img)[0].flatten(start_dim=1)
            feat_real.append(h.cpu().numpy())
        feat_real = np.concatenate(feat_real, axis=0)

        feat_gen = []
        for img in dl_gen:
            img = img.to(self.device)
            img = torch.nn.functional.interpolate(img, size=(299, 299), mode='bicubic')
            with torch.no_grad():
                h = encoder(img)[0].flatten(start_dim=1)
            feat_gen.append(h.cpu().numpy())
        feat_gen = np.concatenate(feat_gen, axis=0)

        m = 1000  # max subset size
        num_subsets = 100

        n = feat_real.shape[1]
        t = 0
        for _ in range(num_subsets):
            x = feat_gen[np.random.choice(feat_gen.shape[0], m, replace=False)]
            y = feat_real[np.random.choice(feat_real.shape[0], m, replace=False)]
            a = (x @ x.T / n + 1) ** 3 + (y @ y.T / n + 1) ** 3
            b = (x @ y.T / n + 1) ** 3
            t += (a.sum() - np.diag(a).sum()) / (m - 1) - b.sum() * 2 / m
        kid = t / num_subsets / m
        return float(kid)

    @torch.no_grad()
    def _compute_ssl_kid(self) -> float:
        """Computes Kernel Inception Distance using features computed using SimCLR encoder

        Returns:
            float: inception score
        """

        dl_real = self.make_dl.get_data_loader_test(self.batch_size, shuffle=False)
        dl_gen = DataLoader(DatasetFromNumpy(self.imgs_gen, self.labels_gen, return_labels=False),
                            batch_size=self.batch_size, shuffle=False, num_workers=4, drop_last=True)

        feat_real = []
        for img, _ in dl_real:
            img = img.to(self.device)
            with torch.no_grad():
                h, _ = self.encoder(img)
            feat_real.append(h.cpu().numpy())
        feat_real = np.concatenate(feat_real, axis=0)

        feat_gen = []
        for img in dl_gen:
            img = img.to(self.device)
            with torch.no_grad():
                h, _ = self.encoder(img)
            feat_gen.append(h.cpu().numpy())
        feat_gen = np.concatenate(feat_gen, axis=0)

        m = 1000
        num_subsets = 100

        n = feat_real.shape[1]
        t = 0
        for _ in range(num_subsets):
            x = feat_gen[np.random.choice(feat_gen.shape[0], m, replace=False)]
            y = feat_real[np.random.choice(feat_real.shape[0], m, replace=False)]
            a = (x @ x.T / n + 1) ** 3 + (y @ y.T / n + 1) ** 3
            b = (x @ y.T / n + 1) ** 3
            t += (a.sum() - np.diag(a).sum()) / (m - 1) - b.sum() * 2 / m
        kid = t / num_subsets / m
        return float(kid)

    @torch.no_grad()
    def _compute_geometric_distance(self) -> float:
        """Computes Geometric Distance using features computed using SimCLR encoder

        Returns:
            float: geometric distance
        """

        loss = SamplesLoss("sinkhorn", p=2, blur=0.05, scaling=0.8, backend="tensorized")

        dl_real = self.make_dl.get_data_loader_test(self.batch_size, shuffle=False)
        dl_gen = DataLoader(DatasetFromNumpy(self.imgs_gen, self.labels_gen, return_labels=False),
                            batch_size=self.batch_size, shuffle=False, num_workers=4, drop_last=True)

        feat_real = []
        for img, _ in dl_real:
            img = img.to(self.device)
            with torch.no_grad():
                h, _ = self.encoder(img)
            feat_real.append(h.cpu().numpy())
        feat_real = np.concatenate(feat_real, axis=0)
        feat_real = torch.from_numpy(feat_real)

        feat_gen = []
        for img in dl_gen:
            img = img.to(self.device)
            with torch.no_grad():
                h, _ = self.encoder(img)
            feat_gen.append(h.cpu().numpy())
        feat_gen = np.concatenate(feat_gen, axis=0)
        feat_gen = torch.from_numpy(feat_gen)

        distance = loss(feat_real, feat_gen)
        return distance.item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_data', type=str, required=True)
    parser.add_argument('--path_labels', type=str, required=True)
    parser.add_argument('--path_gen_images', type=str, required=True)
    parser.add_argument('--path_gen_labels', type=str, required=True)

    args = parser.parse_args()
    gen_images = np.load(args.path_gen_images)
    labels = np.load(args.path_gen_labels)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    evaluator = Evaluator(args.path_data, args.path_labels, gen_images, labels, device)
    evaluator.evaluate()
