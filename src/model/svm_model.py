import pickle

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


class SVM:

    def __init__(self, train_images, train_labels, test_images, test_labels):
        self.train_images = train_images
        self.train_labels = train_labels
        self.test_images = test_images
        self.test_labels = test_labels
        self.model = None

    def grid_search(self):
        param_grid = {'C': [0.1, 1, 10, 100],
                      'gamma': [1, 0.1, 0.01, 0.001, 0.00001, 10]}

        for kernel in ['linear', 'rbf']:
            svm = SVC(kernel=kernel)
            grid_search = GridSearchCV(svm, param_grid, cv=2, verbose=2, n_jobs=10)
            print(len(self.train_images))
            print(len(self.train_labels))
            grid_search.fit(X=self.train_images, y=self.train_labels)

        best_model = grid_search.best_estimator_
        self.model = best_model

    def run_model(self):
        self.model.fit(self.train_images, self.test_labels)

    def predict(self):
        predicted_labels = self.model.predict(self.test_images)
        accuracy = accuracy_score(self.test_labels, predicted_labels)
        print("Accuracy:", accuracy)

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
            return self.model
        with open(filename, "rb") as f:
            self.model = pickle.load(f)
        print(f"Model loaded from: {filename}")
        return self.model
