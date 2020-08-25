import os
from PIL import Image
from torch.utils.data.dataset import Dataset


class DatasetProcessingCIFAR_10(Dataset):
    def __init__(self, data_path, img_filename, transform=None):
        self.img_path = data_path
        self.transform = transform
        img_filepath = os.path.join(data_path, img_filename)
        fp = open(img_filepath, 'r')
        self.img_filename = [x.strip() for x in fp]
        fp.close()

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_path, self.img_filename[index]))
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, index

    def __len__(self):
        return len(self.img_filename)


class DatasetProcessingNUS_WIDE(Dataset):
    def __init__(self, data_path, img_filename, transform=None):
        self.img_path = data_path
        self.transform = transform
        img_filepath = os.path.join(data_path, img_filename)
        fp = open(img_filepath, 'r')
        self.img_filename = [x.strip() for x in fp]
        fp.close()

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_path, self.img_filename[index]))
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, index

    def __len__(self):
        return len(self.img_filename)


class DatasetProcessingPorject(Dataset):
    def __init__(self, data_path, img_filename, transform=None):
        self.img_path = data_path
        self.transform = transform
        img_filepath = os.path.join(data_path, img_filename)
        fp = open(img_filepath, 'r', encoding='utf-8')
        self.img_filename = [x.strip() for x in fp]
        fp.close()

    def __getitem__(self, index):
        Image.MAX_IMAGE_PIXELS = 1000000000
        img = Image.open(os.path.join(self.img_path, self.img_filename[index]))
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, index

    def __len__(self):
        return len(self.img_filename)
