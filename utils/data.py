import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import v2
from config.constants import crop_dict, lung_seg_dict, arch_seg_dict

class SideBySideDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        for label, subdir in enumerate(['normal', 'nodule']):
            subdir_path = os.path.join(root_dir, subdir)
            for image_name in os.listdir(subdir_path):
                self.image_paths.append(os.path.join(subdir_path, image_name))
                self.labels.append(label)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        width, height = image.size
        left_image = image.crop((0, 0, width // 2, height))
        right_image = image.crop((width // 2, 0, width, height))
        
        if self.transform:
            left_image = self.transform(left_image)
            right_image = self.transform(right_image)
        
        return left_image, right_image, label

def get_cxr_train_transforms(crop_size, normalise):
    return [
        v2.ToImage(),
        v2.RandomRotation(15),
        v2.RandomHorizontalFlip(),
        v2.RandomApply([
            v2.ColorJitter(0.4, 0.2, 0.2, 0)
        ], p=0.8),
        v2.RandomResizedCrop(size=crop_size, scale=(0.6, 1.), antialias=True),
        v2.ToDtype(torch.float32, scale=False),
        normalise
    ]

def get_cxr_eval_transforms(crop_size, normalise):
    return [
        v2.ToImage(),
        v2.Resize(size=crop_size, antialias=True),
        v2.ToDtype(torch.float32, scale=False),
        normalise
    ]

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

def set_loader(opt, rank, world_size):
    if opt.processing == "crop":
        mean, std = crop_dict[opt.dataset]
    elif opt.processing == "lung_seg":
        mean, std = lung_seg_dict[opt.dataset]
    elif opt.processing == "arch_seg":
        mean, std = arch_seg_dict[opt.dataset]
    else:
        raise ValueError('Dataset processing type not supported: {}'.format(opt.dataset))

    v2Normalise = transforms.Normalize(mean=mean, std=std)

    cxr_v2_train_transform = transforms.Compose(get_cxr_train_transforms(opt.size, v2Normalise))
    cxr_v2_val_transform = transforms.Compose(get_cxr_eval_transforms(opt.size, v2Normalise))

    train_dataset = SideBySideDataset(
        root_dir=os.path.join(opt.data_folder, opt.processing, 
                              "flat_std_1024" if opt.processing == "arch_seg" else "std_1024", "train"),
        transform=cxr_v2_train_transform
    )
    val_dataset = SideBySideDataset(
        root_dir=os.path.join(opt.data_folder, opt.processing, 
                              "flat_std_1024" if opt.processing == "arch_seg" else "std_1024", "test"),
        transform=cxr_v2_val_transform
    )

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)

    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers, pin_memory=True, sampler=val_sampler)

    return train_loader, val_loader

