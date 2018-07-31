from collect import load_mnist
from network import NeuralNetwork

training_data, validation_data, test_data = load_mnist()
net = NeuralNetwork([784, 80, 40, 16, 10])
net.fit(list(training_data), list(validation_data))

# champions:
# [784, 16, 10]: 92.31
# [784, 100, 16, 10]: 94.22
# [784, 80, 40, 16, 10]: 94.31
