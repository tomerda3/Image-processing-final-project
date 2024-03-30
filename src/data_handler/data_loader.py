import os
import cv2
from pathlib import Path
from tqdm import tqdm


class DataLoader:

    def __init__(self, limiter: int = None):
        self.limiter = limiter
        self.mapping_labels = {'male': 0, 'female': 1}
        self.path = Path.cwd() / "HHD_gender"
        self.train_images, self.train_labels = self.load_images('train')
        self.test_images, self.test_labels = self.load_images('test')
        self.val_images, self.val_labels = self.load_images('val')

    @property
    def train(self):
        return self.train_images, self.train_labels

    @property
    def test(self):
        return self.test_images, self.test_labels

    @property
    def val(self):
        return self.val_images, self.val_labels

    def load_images(self, dtype: str):
        images = []
        labels = []
        path = self.path / dtype
        for gender in ['male', 'female']:
            image_files = os.listdir(path / gender)
            print(f"load->{dtype} gender->{gender}\n")
            for file_name in tqdm(image_files):
                image_path = os.path.join(path / gender, file_name)
                image = cv2.imread(image_path)
                images.append(image)
                labels.append(self.mapping_labels[gender])
        return images, labels
