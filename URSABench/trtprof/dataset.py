import numpy as np
import torch
from skimage import io
from skimage.transform import resize
from torch.utils.data import Dataset
from torchvision.transforms import Normalize


class MLPDataset(Dataset):
    def __init__(self, n_samples, n_feats):
        super(MLPDataset, self).__init__()
        self.n_samples = n_samples
        self.n_feats = n_feats
        self.labels = np.random.randint(1, size=self.n_samples)
        self.feats = np.random.randn(self.n_samples, self.n_feats).astype("f")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        return self.feats[index], self.labels[index]


class DummyDataset(Dataset):
    def __init__(self, img_hw, n_samples, dtype):
        self.n_samples = n_samples
        self.dtype = dtype

        url = "https://images.dog.ceo/breeds/retriever-golden/n02099601_3004.jpg"
        self.img = self.transform(resize(io.imread(url), (img_hw, img_hw)))
        self.img_labels = np.random.randint(0, 2, size=self.n_samples)

    def __len__(self):
        return self.n_samples

    def transform(self, img):
        norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        result = norm(torch.from_numpy(img).transpose(0, 2).transpose(1, 2))
        return np.array(result, dtype=self.dtype)

    def __getitem__(self, idx):
        label = self.img_labels[idx]
        return self.img, label
