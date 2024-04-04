import cv2
from model.svm_model import SVM
from pre_process.preprocess import PreProcess
from data_handler.data_loader import DataLoader
from feature_extraction.feature_extraction import *
import sys
from pathlib import Path


class App:

    def __init__(self, args):
        self.data_loader = DataLoader(limiter=300)
        self.pre_processor = PreProcess()
        self.svm_model = None

    def regular_preprocess(self):
        train_images = []
        for image in self.data_loader.train_images:
            words = self.pre_processor.regular_preprocess(image)
            train_images += words
        train_labels = self.data_loader.train_labels

        val_images = []
        for image in self.data_loader.val_images:
            words = self.pre_processor.regular_preprocess(image)
            val_images += words
        val_labels = self.data_loader.val_labels

        test_images = []
        for image in self.data_loader.test_images:
            words = self.pre_processor.regular_preprocess(image)
            test_images += words
        test_labels = self.data_loader.test_labels

        return train_images, train_labels, val_images, val_labels, test_images, test_labels

    def get_features(self, images, labels):
        features = extract_lbp_features(images)
        total_labels = labels
        return features, total_labels

    def run_model(self):
        x_train1, y_train1, x_val1, y_val1, x_test1, y_test1 = self.regular_preprocess()

        x_train, y_train = self.get_features(x_train1, y_train1)
        x_val, y_val = self.get_features(x_val1, y_val1)
        x_test, y_test = self.get_features(x_test1, y_test1)

        del x_train1, y_train1,x_val1, y_val1, x_test1, y_test1

        model = SVM(x_train, y_train, x_val, y_val, x_test, y_test)
        print("start grid search...")
        model.run_model()
        print("found the best parameters for model...")
        print("start predictions...")
        model.predict()
        print("model predictions done...")
        model.make_confusion_matrix()


if __name__ == '__main__':
    app = App(sys.argv[1:])
    app.run_model()
