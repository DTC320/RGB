import os
import torch
import scipy.io
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
from augmentation import augment_image


def pad_image(image, target_shape=(512, 512)):
    if len(image.shape) == 3 and image.shape[0] in [3, 31]:  # (channels, height, width)
        image = np.transpose(image, (1, 2, 0))  # Convert to (height, width, channels)
    
    h, w, c = image.shape
    pad_h = target_shape[0] - h
    pad_w = target_shape[1] - w

    # Ensure even padding on both sides
    pad_h_before = pad_h // 2
    pad_h_after = pad_h - pad_h_before
    pad_w_before = pad_w // 2
    pad_w_after = pad_w - pad_w_before

    padding = ((pad_h_before, pad_h_after), (pad_w_before, pad_w_after), (0, 0))
    return np.pad(image, padding, mode='constant', constant_values=0)




def split_into_patches(image, patch_size=(128, 128)):
    h, w = image.shape[:2]
    patches = []
    for i in range(0, h, patch_size[0]):
        for j in range(0, w, patch_size[1]):
            patch = image[i:i+patch_size[0], j:j+patch_size[1]]
            patches.append(patch)
    return patches

class MyDataset(Dataset):
    def __init__(self, rgb_folder, mat_folder, patch_size=128, augment=None):
        self.rgb_folder = rgb_folder
        self.mat_folder = mat_folder
        self.patch_size = patch_size
        self.augment = augment
        self.rgb_files = sorted([f for f in os.listdir(rgb_folder) if f.startswith('rgb_') and f.endswith('.pt')])
        self.all_patches = []
        
        for rgb_filename in self.rgb_files:
            # Load the RGB image from .pt file
            rgb_path = os.path.join(self.rgb_folder, rgb_filename)
            rgb_image_np = torch.load(rgb_path).numpy()
            rgb_image_np = np.squeeze(rgb_image_np)
            
            # Pad and split the RGB image
            rgb_image_np = pad_image(rgb_image_np, target_shape=(512, 512))
            rgb_patches = split_into_patches(rgb_image_np, patch_size=(self.patch_size, self.patch_size))
            
            # Load the HS data from .pt file
            hs_filename = rgb_filename.replace("rgb_", "hyperspectral_")
            hs_path = os.path.join(self.mat_folder, hs_filename)
            hyperspectral_data = torch.load(hs_path).numpy()
            hyperspectral_data = np.squeeze(hyperspectral_data)
            
            # Pad and split the hyperspectral data
            hyperspectral_data = pad_image(hyperspectral_data, target_shape=(512, 512))
            hs_patches = split_into_patches(hyperspectral_data, patch_size=(self.patch_size, self.patch_size))
            
            for rgb_patch, hs_patch in zip(rgb_patches, hs_patches):
                self.all_patches.append((rgb_patch, hs_patch))

    def __len__(self):
        return len(self.all_patches)

    def __getitem__(self, idx):
        rgb_data, hyperspectral_data = self.all_patches[idx]

        if self.augment:
            rgb_data = augment_image(rgb_data)
        else:
            rgb_data = np.transpose(rgb_data, (2, 0, 1))
        #print("rgb shape: ", rgb_data.shape)   

        hyperspectral_data = np.transpose(hyperspectral_data, (2, 0, 1))
        #print("HS shape: ", hyperspectral_data.shape)   

        return torch.tensor(rgb_data, dtype=torch.float32), torch.tensor(hyperspectral_data, dtype=torch.float32)



if __name__ == "__main__":
    
    dataset = MyDataset(rgb_folder='/root/autodl-tmp/ZERO/RGB_train', mat_folder='/root/autodl-tmp/ZERO/HS_train', patch_size=128, augment=True)
    #dataset = MyDataset(rgb_folder='/root/autodl-tmp/ZERO/RGB_train', mat_folder='/root/autodl-tmp/ZERO/HS_train', patch_size=128)
    print("len:", len(dataset))
    
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
        
    
    for batch_idx, (rgb_patches, hs_patches) in enumerate(train_loader):
        print(f"Batch {batch_idx + 1}: RGB size: {rgb_patches.shape}, HS size: {hs_patches.shape}")

        
        assert rgb_patches.shape[1:] == (3, 128, 128), f"Unexpected size for RGB patch in batch {batch_idx + 1}"
        assert hs_patches.shape[1:] == (31, 128, 128), f"Unexpected size for HS patch in batch {batch_idx + 1}"

    print("All checks passed!")

