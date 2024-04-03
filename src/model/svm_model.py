import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

class SVM:

    def __init__(self, train_images, train_labels, val_images, val_labels, test_images, test_labels):
        self.train_images = train_images
        self.train_labels = train_labels
        self.test_images = test_images
        self.test_labels = test_labels
        self.val_images = val_images
        self.val_labels = val_labels
        self.model = None

    def run_model(self):
        param_grid = {'C': [0.1, 1, 10, 100],
                      'gamma': [1, 0.1, 0.01, 0.001, 0.00001, 10]}

        best_models_array = []
        for kernel in ['linear', 'rbf']:
            svm = SVC(kernel=kernel)
            if kernel == 'rbf':
                grid_search = GridSearchCV(svm, param_grid, cv=5, verbose=2, n_jobs=6)
                grid_search.fit(self.train_images, self.train_labels)
                best_model = grid_search.best_estimator_
                best_models_array.append(best_model)
            else:
                svm.fit(self.train_images, self.train_labels)
                best_models_array.append(svm)

        self.model = self.find_best_model(best_models_array, self.val_images, self.val_labels)

    def find_best_model(self,models, val_images, val_labels):

        best_model = None
        best_accuracy = 0
        index = 1
        for model in models:
            predictions = model.predict(val_images)
            accuracy = accuracy_score(val_labels, predictions)
            print(f"Model {index} accuracy: {accuracy}")
            print(f"Model Name {model.__str__()} parameters: {model.get_params()}")
            index += 1
            if accuracy > best_accuracy:
                best_model = model
                best_accuracy = accuracy
        # print the best model
        print(f"Best model accuracy: {best_accuracy}")
        print(f"Best model Name {best_model.__str__()} parameters: {best_model.get_params()}")
        return best_model

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
