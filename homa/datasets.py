import torch
from torchvision import transforms


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


class AugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform_probability=0.6):
        self.original_dataset = dataset
        self.transform_probability = transform_probability
        self.augmented_data = []
        self.transformations = [
            transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(
                        brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2
                    ),
                    transforms.RandomRotation(45),
                ]
            ),
            transforms.Compose(
                [
                    transforms.RandomVerticalFlip(),
                    transforms.RandomAffine(
                        degrees=30, translate=(0.1, 0.1), scale=(0.8, 1.2)
                    ),
                    transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),
                ]
            ),
            transforms.Compose(
                [
                    transforms.RandomResizedCrop(size=(224, 224), scale=(0.5, 1.0)),
                    transforms.RandomPerspective(distortion_scale=0.5, p=1.0),
                    transforms.RandomGrayscale(p=0.3),
                ]
            ),
        ]
        self._augment_dataset()

    def _augment_dataset(self):
        for image, label in self.original_dataset:
            if torch.rand(1).item() < self.transform_probability:
                for transform in self.transformations:
                    augmented_image = transform(image)
                    self.augmented_data.append((augmented_image, label))
            else:
                self.augmented_data.append((image, label))

    def __len__(self):
        return len(self.augmented_data)

    def __getitem__(self, idx):
        return self.augmented_data[idx]
