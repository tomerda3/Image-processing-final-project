import pickle
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from typing import Tuple, Literal
import numpy as np
from data_loader import DataLoader
from label_splitter import *
from pre_process import PreProcess
from collections import Counter
from sklearn.metrics import accuracy_score,f1_score

class Engine:

    def __init__(self, image_shape: Tuple):
        self.model = None
        self.image_shape = image_shape
        self.train_data_path = None
        self.test_data_path = None
        self.train_labels = None
        self.eval_labels = None
        self.test_labels = None
        self.train_images = None
        self.test_images = None
        self.eval_images = None

    def set_train_labels(self, df):
        self.train_labels = df

    def set_test_labels(self, df):
        self.test_labels = df

    def set_train_data_path(self, path: str):
        self.train_data_path = path

    def set_test_data_path(self, path: str):
        self.test_data_path = path

    def preprocess_data(self, images, labels, data_type):
        print(f"\nPreprocessing {data_type} images...")

        preprocessor = PreProcess(self.image_shape)
        proc_labels = preprocessor.arrange_labels_indexing_from_0(labels)
        reverse_binarize_images = preprocessor.preprocess_and_binarize_images(images)

        cropped_images = preprocessor.crop_text_from_reversed_binary_images(reverse_binarize_images)
        proc_images = cropped_images
        if data_type == "train":
            proc_images, proc_labels = preprocessor.patch_images(cropped_images, proc_labels, self.image_shape)
        # proc_images = preprocessor.resize_images(proc_images)

        return proc_images, proc_labels

    def load_images(self, data_type: Literal["test", "train","val"], image_filename_col: str, label_col: str,
                    clean_method: Literal["HHD"]="HHD"):

        data_path, dataframe = "", ""

        print(f"Loading {data_type} images...")
        if data_type == "test":
            data_path = self.test_data_path
            dataframe = self.test_labels
        elif data_type == "train":
            data_path = self.train_data_path
            dataframe = self.train_labels
        elif data_type == "val":
            data_path = self.eval_data_path
            dataframe = self.eval_labels

        data_loader = DataLoader(dataframe, data_type, data_path, image_filename_col, label_col)
        images, labels = data_loader.load_data(clean_method)
        print(f"Number of {data_type} images: {len(images)} labels: {len(labels)}")
        # Preprocessing:
        proc_images, proc_labels = self.preprocess_data(images, labels, data_type)

        if data_type == "train":
            self.train_images, self.train_labels = proc_images, proc_labels
        elif data_type == "test":
            self.test_images, self.test_labels = proc_images, proc_labels
        elif data_type == "validation":
            self.eval_images, self.eval_labels = proc_images, proc_labels

    def train_model(self, features, labels):
        param_grid = {'C': [0.1, 1, 10, 100],
                      'gamma': [1, 0.1, 0.01, 0.001, 0.00001, 10]}
        for kernel in ['linear', 'rbf']:
            svm = SVC(kernel=kernel)
            grid_search = GridSearchCV(svm, param_grid, cv=5)
            grid_search.fit(features, labels)

            print("Best parameters for", kernel, "kernel:", grid_search.best_params_)

        # Move the best_model definition and assignment outside the loop
        best_model = grid_search.best_estimator_
        self.model = best_model

    def validation_model(self, images, labels):
        predictions = self.model.predict(images)
        accuracy = accuracy_score(labels, predictions)
        print("Validation accuracy: {}%".format(accuracy * 100))

    def save_model(self, filename="saved_model.pkl"):
        if self.load_model() is None:
            print("Saving model...")
            with open(filename, "wb") as f:
                pickle.dump(self.model, f)
            print(f"Model saved to: {filename}")
        else:
            print("Model already exists. Please delete the file or load the model.")

    def load_model(self, filename="saved_model.pkl"):
        if self.model is not None:
            print("Model already loaded.")
            return self.model
        with open(filename, "rb") as f:
            self.model = pickle.load(f)
        print(f"Model loaded from: {filename}")
        return self.model

    def predict(self, images):
        predicted_labels = self.model.predict(images)
        print("Accuracy: {}%".format(self.model.score(images) * 100))
        return predicted_labels


def construct_HHD_engine(base_dir, image_shape):

    # Setting file system
    train_path = base_dir / "train"
    test_path = base_dir / "test"
    csv_label_path = str(base_dir / "AgeSplit.csv")

    # Initializing engine
    engine = Engine(image_shape)

    # Setting engine labels & paths
    HHD_labels = LabelSplitter(csv_label_path)  # returns object with 'train', 'test', 'val' attributes
    engine.set_train_labels(HHD_labels.train)
    engine.set_test_labels(HHD_labels.test)
    engine.set_test_data_path(str(test_path))
    engine.set_train_data_path(str(train_path))
    engine.load_images(data_type='train', image_filename_col='File', label_col='Age')
    engine.load_images(data_type='val', image_filename_col='File', label_col='Age')
    engine.load_images(data_type='test', image_filename_col='File', label_col='Age')

    return engine
