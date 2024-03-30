from src.model.svm_model import SVM
from src.pre_process.preprocess import PreProcess
from src.data_handler.data_loader import DataLoader
from src.feature_extraction.feature_extraction import *


class App:

    def __init__(self):
        self.data_loader = DataLoader(limiter=150)
        self.pre_processor = PreProcess()
        self.svm_model = None

    def pre_process_data(self):
        train_images = []
        train_labels = []
        for index, image in enumerate(self.data_loader.train_images):
            words = self.pre_processor.extract_words(image)
            train_images += words
            train_labels += [self.data_loader.train_labels[index]] * len(words)

        test_images = []
        test_labels = []
        for index, image in enumerate(self.data_loader.train_images):
            words = self.pre_processor.extract_words(image)
            test_images += words
            test_labels += [self.data_loader.train_labels[index]] * len(words)

        return train_images, train_labels, test_images, test_labels

    def get_features(self, train_images, train_labels):
        features = features_extract_combine(train_images)
        total_labels = train_labels
        return features, total_labels

    def run_model(self):
        x_train1, y_train1, x_test1, y_test1 = self.pre_process_data()

        x_train, y_train = self.get_features(x_train1, y_train1)
        x_test, y_test = self.get_features(x_test1, y_test1)

        del x_train1, y_train1, x_test1, y_test1

        model = SVM(x_train, y_train, x_test, y_test)
        print("start grid search...")
        model.run_model()
        print("found the best parameters for model...")
        print("start predictions...")
        model.predict()


if __name__ == '__main__':
    app = App()
    app.run_model()
