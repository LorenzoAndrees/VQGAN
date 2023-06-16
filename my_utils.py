import os, albumentations
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from my_config import DATASET_PATH, IMAGE_SIZE, BATCH_SIZE

def loadSource(path, dataset):
    source_path = os.path.join(path, dataset)
    source = pd.read_csv(source_path, sep = ";", index_col = False)
    source.fillna(0, inplace = True)
    df = source[source.selected == 1].copy()
    df['imgs'] = df.filename.apply(
        lambda f: os.path.join(path, f)
    )
    df['v'] = df['v'].astype('int')
    df['a'] = df['a'].astype('int')
    df['d'] = df['d'].astype('int')
    images = df.imgs.to_numpy()
    emotions = df[['v', 'a', 'd']].to_numpy()
    return {'path': images, 'emotion': emotions}

class ImagePaths(Dataset):
    def __init__(self, path, size = None):
        self.size = size
        source = loadSource(path, 'data.csv')
        self.images = source['path']
        self.emotions = source['emotion']
        self._length = len(self.images)
        self.rescaler =  albumentations.SmallestMaxSize(
            max_size = self.size
        )
        self.cropper = albumentations.CenterCrop(
            height = self.size,
            width = self.size
        )
        self.preprocessor = albumentations.Compose(
            [self.rescaler, self.cropper]
        )

    def __len__(self):
        return self._length
    
    def preprocessImage(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image / 127.5 - 1.0).astype(np.float32)
        image = image.transpose(2, 0 , 1)
        return image
    
    def __getitem__(self, i):
        return self.preprocessImage(self.images[i]), self.emotions[i]

def loadData():
    train_data = ImagePaths(DATASET_PATH, size = IMAGE_SIZE)
    train_loader = DataLoader(
        train_data,
        batch_size = BATCH_SIZE,
        shuffle = True
    )
    return train_loader