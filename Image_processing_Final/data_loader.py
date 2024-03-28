import os
import pandas as pd
from pathlib import Path
import cv2
from typing import List
from typing import Literal
from tqdm import tqdm

SHORT_RUN = False
IMAGE_LIMIT = 100

class DataLoader:
    def __init__(self, dataframe, data_type: Literal["test", "train"], data_path: str, name_col: str, label_col: str):
        self.df = dataframe
        self.type = data_type
        self.path = data_path
        self.name_col = name_col
        self.label_col = label_col

    def load_data(self, clean_method: Literal["HHD"]="HHD"):

        images = []
        labels = []
        orginal_path = self.path
        # read file for man
        self.path = os.path.join(self.path, "male")
        files = self._get_files_name()
        if SHORT_RUN:
            cnt = 0

        for file_name in tqdm(files):

            if SHORT_RUN:
                cnt += 1
                if cnt == IMAGE_LIMIT:
                    return images, labels

            image = cv2.imread(str(Path(self.path) / file_name))  # TODO: Read as greyscale -> remove from preprocess
            gender = "male"
            images.append(image)
            labels.append(gender)

        # read files from female
        self.path = orginal_path
        self.path = os.path.join(self.path, "female")
        files = self._get_files_name()
        for file_name in tqdm(files):

            if SHORT_RUN:
                cnt += 1
                if cnt == IMAGE_LIMIT:
                    return images, labels

            image = cv2.imread(str(Path(self.path) / file_name))  # TODO: Read as greyscale -> remove from preprocess
            gender = "female"
            images.append(image)
            labels.append(gender)

        return images, labels

    def _get_files_name(self) -> List[str]:
        file_names = []
        for file in os.listdir(self.path):
            if os.path.isfile(os.path.join(self.path, file)):
                file_names.append(file)
        return file_names
