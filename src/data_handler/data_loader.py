import os
import sys

import cv2
from pathlib import Path
from tqdm import tqdm


class DataLoader:

    def __init__(self, limiter: int = None):
        path_args = sys.argv[1:]
        print(path_args)
        self.limiter = limiter
        self.mapping_labels = {'male': 0, 'female': 1}
        self.path = Path.cwd() / "HHD_gender"
        self.train_images, self.train_labels = self.load_images('train',path_args[0])
        self.test_images, self.test_labels = self.load_images('test',path_args[1])
        self.val_images, self.val_labels = self.load_images('val',path_args[2])

    @property
    def train(self):
        return self.train_images, self.train_labels

    @property
    def test(self):
        return self.test_images, self.test_labels

    @property
    def val(self):
        return self.val_images, self.val_labels

    def load_images(self, dtype: str,data_path: str):
        images = []
        labels = []

        images_path = Path(data_path)  # Create a Path object once for efficiency

        for gender in ["male", "female"]:
            print(f"load->{dtype} gender->{gender}\n")

            gender_path = images_path / gender  # Create subdirectory Path for clarity
            image_files = os.listdir(gender_path)  # Use Pathlib for path handling

            for file_name in tqdm(image_files):
                image_path = gender_path / file_name  # Use Pathlib for joining
                image = cv2.imread(str(image_path))  # Convert Path to string for cv2.imread
                images.append(image)
                labels.append(self.mapping_labels[gender])

        return images, labels
