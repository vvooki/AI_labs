import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

class CentroidGenerator:
    def __init__(self, num_centroids, input_dim, learning_rate=0.1, num_epochs=100):
        self.num_centroids = num_centroids
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.centroids = np.random.rand(num_centroids, input_dim)
        self.weights = np.random.rand(input_dim, num_centroids)
        
    def train(self, data):
        for epoch in range(self.num_epochs):
            for i in range(data.shape[0]):
                input_data = data[i]
                activation = np.dot(input_data, self.weights)
                winner_index = np.argmax(activation)
                self.centroids[winner_index] += self.learning_rate * (input_data - self.centroids[winner_index])
                self.weights[:, winner_index] += self.learning_rate * (input_data - self.centroids[winner_index])
                    
    def plot_data(self, data):
        # obliczenie etykiet dla każdego punktu danych
        activation = np.dot(data, self.weights)
        labels = np.argmax(activation, axis=1)
        
        # wykres punktów danych i centroidów
        plt.scatter(data[:, 0], data[:, 1], c=labels)
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], color='r', marker='x', s=100, label='Wygenerowane centroidy')
        
        # obliczenie rzeczywistych centroidów
        real_centroids = np.zeros((self.num_centroids, self.input_dim))
        for i in range(self.num_centroids):
            real_centroids[i] = np.mean(data[labels == i], axis=0)
        
        # wykres rzeczywistych centroidów
        plt.scatter(real_centroids[:, 0], real_centroids[:, 1], color='b', marker='o', s=100, label='Rzeczywiste centroidy')
        
        plt.legend()
        plt.show()

# liczba centroidów
num_centroids = 3

# wymiar danych wejściowych
input_dim = 2

# tempo uczenia
learning_rate = 0.1

# liczba epok
num_epochs = 10

# wygenerowanie danych
data, _ = make_blobs(n_samples=1000, centers=num_centroids, n_features=input_dim, random_state=42)

# utworzenie generatora centroidów
generator = CentroidGenerator(num_centroids, input_dim, learning_rate, num_epochs)

# trening generatora na danych
generator.train(data)

# wykres danych i centroidów
generator.plot_data(data)