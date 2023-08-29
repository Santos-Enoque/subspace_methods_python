import numpy as np
from sklearn.model_selection import train_test_split

class DataTransformer:
    def __init__(self, X: np.ndarray, y: np.ndarray, test_size=0.2, random_state=42, n_of_sets=2):
        self.X = X
        self.y = y
        self.test_size = test_size
        self.random_state = random_state
        # numver of features
        self.n_of_features = X.shape[-1]
        self.n_of_sets = n_of_sets
        self.X_train, self.X_test, self.y_train, self.y_test = self._split_data()

    def transform(self) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        # Group the train and test data
        n_of_classes = len(np.unique(self.y_train))
        X_train_grouped = np.array([self.X_train[self.y_train == i] for i in range(n_of_classes)])
        X_test_grouped = np.array([self.X_test[self.y_test == i] for i in range(n_of_classes)])
        X_test_grouped = X_test_grouped.reshape(n_of_classes, self.n_of_sets, -1, self.n_of_features)

        # Reshape and prepare labels for grouped data
        train_X = X_train_grouped
        train_y = np.arange(len(X_train_grouped))
        test_X = X_test_grouped.reshape(-1, X_test_grouped.shape[-2], self.n_of_features)
        test_y = np.array([[i] * self.n_of_sets for i in range(n_of_classes)]).flatten()

        return train_X, train_y, test_X, test_y

    def _split_data(self) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        return train_test_split(self.X, self.y, 
                                test_size=self.test_size, 
                                random_state=self.random_state, 
                                stratify=self.y)
