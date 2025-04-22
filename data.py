from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os

class FlowerDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path = self.data.iloc[index]["image_path"]
        label = self.data.iloc[index]["label"]
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return image, label


def Data(dataset_name, dataset_root):
    if dataset_name == "Flowers":
        path = str(dataset_root) + "flowers"
        labels = ["astilbe", "bellflower", "black_eyed_susan", "calendula", "california_poppy", "carnation",
                  "common_daisy", "coreopsis", "daffodil", "dandelion", "iris", "magnolia", "rose", "sunflower",
                  "tulip", "water_lily"]
        data = []
        for label in labels:
            img_dir = os.path.join(path, label)
            image_names = os.listdir(img_dir)
            for image_name in image_names:
                img_path = os.path.join(img_dir, image_name)
                data.append({"image_path": img_path, "label": label})
        data = pd.DataFrame(data)
        data["label"] = data["label"].map({label: i for i, label in enumerate(labels)})
        data.head()
        return data
    else: raise ValueError("Invalid dataset name")