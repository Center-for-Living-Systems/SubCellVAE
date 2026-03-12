from torch.utils.data import DataLoader, Dataset, random_split
import os
import torch
import numpy as np
import tifffile as tiff


# Cached version
class TIFFDataset(Dataset):
    def __init__(self, root_dir, label=0, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.label = label

        self.image_paths = [
            os.path.join(root_dir, fname)
            for fname in os.listdir(root_dir)
            if fname.lower().endswith(('tif', 'tiff'))
        ]

        self.data = []
        for img_path in self.image_paths:
            try:
                image = tiff.imread(img_path).astype(np.float32)

                # Image normalization is usually done in the transform pipeline,
                # which is provided when initializing the Dataset.  
                if self.transform:
                    image = self.transform(image)

                image = torch.tensor(image, dtype=torch.float32)  # shape: (1, H, W)
                self.data.append(image)
            except Exception as e:
                print(f"Warning: Skipping unreadable image {img_path} - {e}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]

        # Ensure it's a torch tensor
        if not isinstance(image, torch.Tensor):
            image = torch.tensor(image, dtype=torch.float32)

        return image, self.label, self.image_paths[idx]

