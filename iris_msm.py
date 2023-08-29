from sklearn.datasets import load_iris
from src.models import KernelMSM
from src.utils.data_transformer import DataTransformer


# Load data
iris_data = load_iris()
X, y = iris_data.data, iris_data.target
train_X, train_y, test_X, test_y = DataTransformer(X, y).transform()

model = KernelMSM(n_subdims=5, sigma=0.1, normalize=True, faster_mode=True)
model.fit(train_X, train_y)
pred = model.predict(test_X)
print(f"pred: {pred}\ntrue: {test_y}\naccuracy: {(pred == test_y).mean()}")