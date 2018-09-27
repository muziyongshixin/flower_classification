from torch.utils.data import Dataset
import torch
from skimage import io
from PIL import Image
import os

class filelist_DataSet(Dataset):
    def __init__(self, image_list_file, transform=None):
        images = []
        labels = []
        with open(image_list_file, "r") as f:
            for line in f:
                items = line.strip("\n").split(" ")
                image_path = items[0]
                image_label=items[1]

                images.append(image_path)
                labels.append(image_label)

        self.images = images
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        """
        :param index: The ind/ex of item.
        :return: Image and its label.
        """
        image_path = self.images[index]
        image = Image.open(image_path).convert('RGB')
        label = torch.tensor(int(self.labels[index])).type(torch.long)  # 图片数据不需要label

        if self.transform is not None:
            image = self.transform(image)



        return image, label

    def __len__(self):
        return len(self.images)
