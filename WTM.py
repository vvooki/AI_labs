import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

class CentroidGenerator:
    def __init__(self, num_centroids, input_dim, learning_rate=0.1, num_epochs=100):
        self.num_centroids = num_centroids  # liczba centroidów
        self.input_dim = input_dim  # wymiar wejściowy danych
        self.learning_rate = learning_rate  # współczynnik uczenia
        self.num_epochs = num_epochs  # liczba epok
        self.centroids = np.random.rand(num_centroids, input_dim)  # inicjalizacja centroidów
        self.weights = np.random.rand(input_dim, num_centroids)  # inicjalizacja wag
        
    def train(self, data):
        for epoch in range(self.num_epochs):
            for i in range(data.shape[0]):
                input_data = data[i]
                activation = np.dot(input_data, self.weights)
                winner_index = np.argmax(activation)  # indeks zwycięskiego neuronu
                self.centroids[winner_index] += self.learning_rate * (input_data - self.centroids[winner_index])  # aktualizacja pozycji zwycięskiego centroidu
                self.weights[:, winner_index] += self.learning_rate * (input_data - self.centroids[winner_index])  # aktualizacja wag
                
                # obliczenie etykiet dla każdego punktu
                activation = np.dot(data, self.weights)
                labels = np.argmax(activation, axis=1)

                # aktualizacja centroidów dla każdej grupy
                for j in range(self.num_centroids):
                    cluster_points = data[labels == j]
                    if len(cluster_points) > 0:
                        self.centroids[j] = np.mean(cluster_points, axis=0)
                    
    def plot_data(self, data):
        # obliczenie etykiet dla każdego punktu
        activation = np.dot(data, self.weights)
        labels = np.argmax(activation, axis=1)
        
        # rysowanie punktów danych i centroidów
        plt.scatter(data[:, 0], data[:, 1], c=labels)
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], color='r', marker='x', s=100, label='Wygenerowane centroidy')
        
        # obliczenie rzeczywistych centroidów
        real_centroids = np.zeros((self.num_centroids, self.input_dim))
        for i in range(self.num_centroids):
            real_centroids[i] = np.mean(data[labels == i], axis=0)
        
        # rysowanie rzeczywistych centroidów
        plt.scatter(real_centroids[:, 0], real_centroids[:, 1], color='b', marker='o', s=100, label='Rzeczywiste centroidy')
        
        plt.legend()
        plt.show()

# liczba centroidów
num_centroids = 3

# wymiar wejściowy danych
input_dim = 2

# współczynnik uczenia
learning_rate = 0.1

# liczba epok
num_epochs = 10

# generowanie danych
data, _ = make_blobs(n_samples=1000, centers=num_centroids, n_features=input_dim, random_state=42)

# tworzenie generatora centroidów
generator = CentroidGenerator(num_centroids, input_dim, learning_rate, num_epochs)

# uczenie generatora na danych
generator.train(data)

# rysowanie danych
generator.plot_data(data)
