import torch

import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
import tifffile as tiff
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from PIL import Image

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import tifffile as tiff

# Cached version
class UnlabeledTIFFDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [
            os.path.join(root_dir, fname)
            for fname in os.listdir(root_dir)
            if fname.lower().endswith(('tif', 'tiff'))
        ]

        self.data = []
        for img_path in self.image_paths:
            try:
                image = tiff.imread(img_path).astype(np.float32)
                image = image * 240
                image[image > 254] = 254
                if image.ndim == 3:
                    image = image[0]
                image = image / 255

                if self.transform:
                    image = self.transform(image)
                image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # shape: (1, H, W)
                self.data.append(image)
            except Exception as e:
                print(f"Warning: Skipping unreadable image {img_path} - {e}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], 0  # Returning dummy label 0


# Load Data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from torch.utils.data import ConcatDataset

dir1 = '/mnt/d/lding/FA/analysis_results/FA_ML_Annabel_20250217/031125/ctrl_ch1_major/ctrl_ch1_patches_gridonly_pslocation00/tiff_patches32_65p_20250909_1521'
dir2 = '/mnt/d/lding/FA/analysis_results/FA_ML_Annabel_20250217/031125/y_ch1_major/y_ch1_patches_gridonly_pslocation00/tiff_patches32_65p_20250909_1532'

transform = transforms.Compose([transforms.ToTensor()])

ds1 = UnlabeledTIFFDataset(root_dir=dir1, transform=transform)
ds2 = UnlabeledTIFFDataset(root_dir=dir2, transform=transform)

dataset = ConcatDataset([ds1, ds2])


# Split dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

# Train AE
ae = AE().to(device)
ae = train_ae(ae, train_loader, val_loader)

torch.save(ae, 'pax_ch1_ps32_ctrl_Y_grid_00_bestmodel_1000epoch.pt')


latents, labels, images = cluster_latents(ae, train_loader, num_clusters=6)


cluster_3Dplot(latents, 0,1,2,labels)

cluster_3Dplot(latents, 3,4,5,labels)


pax_ctrl_loader = DataLoader(ds1, batch_size=128, shuffle=False)
pax_y_loader = DataLoader(ds2, batch_size=128, shuffle=False)

latents_2d = visualize_clusters(latents, labels, images, ae)


cluster_and_display_images(latents, labels, images, ae, num_samples_per_cluster=10)

