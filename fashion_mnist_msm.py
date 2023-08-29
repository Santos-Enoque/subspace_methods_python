from sklearn.datasets import fetch_openml
from src.models import KernelMSM
from src.utils.data_transformer import DataTransformer

# Load Fashion MNIST data
fashion_mnist = fetch_openml(name='Fashion-MNIST', version=1, parser='auto')

# Extract the data and labels
X, y = fashion_mnist["data"], fashion_mnist["target"]

# Convert labels to integers, since they come as strings
y = y.astype(int)


train_X, train_y, test_X, test_y = DataTransformer(X, y).transform()

model = KernelMSM(n_subdims=5, sigma=0.1, normalize=True, faster_mode=True)
model.fit(train_X, train_y)
pred = model.predict(test_X)
print(f"pred: {pred}\ntrue: {test_y}\naccuracy: {(pred == test_y).mean()}")
