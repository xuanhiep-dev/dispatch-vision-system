import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


class AlbumentationsDataset(Dataset):
    def __init__(self, subset, transform=None, albumentations_transform=None):
        self.subset = subset
        self.dataset = subset.dataset
        self.indices = subset.indices
        self.transform = transform
        self.albumentations_transform = albumentations_transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        path, label = self.dataset.samples[real_idx]
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.albumentations_transform:
            augmented = self.albumentations_transform(image=image)
            image = augmented['image']
        elif self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        return image, label


class ClsDataset:
    def __init__(self, dataset_path, batch_size=64):
        self.dataset_path = dataset_path
        self.batch_size = batch_size

        # Augmentations for train dataset
        self.train_alb_transform = A.Compose([
            A.RandomResizedCrop(size=(224, 224), scale=(0.9, 1.0), p=1.0),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(
                hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.4),
            A.MotionBlur(blur_limit=3, p=0.2),
            A.GaussNoise(p=0.2),
            A.ImageCompression(p=0.3),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

        # Augmentations for valid dataset
        self.val_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))
        ])
        self._prepare()

    def _prepare(self):
        # Load full dataset
        full_dataset = datasets.ImageFolder(root=self.dataset_path)
        self.classes = full_dataset.classes

        # Split train/val
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size])

        # Apply augmentations
        self.train_dataset = AlbumentationsDataset(
            train_dataset,
            albumentations_transform=self.train_alb_transform
        )

        self.val_dataset = AlbumentationsDataset(
            val_dataset,
            transform=self.val_transform
        )

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False)
