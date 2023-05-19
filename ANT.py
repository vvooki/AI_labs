import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Funkcja Ackley
def ackley(x, y):
    return -20 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2))) - np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))) + np.exp(1) + 20

# Funkcja Rastrigin
def rastrigin(x, y):
    return 20 + (x**2 - 10 * np.cos(2 * np.pi * x)) + (y**2 - 10 * np.cos(2 * np.pi * y))

# Funkcja Eggholder
def eggholder(x, y):
    return -(y + 47) * np.sin(np.sqrt(np.abs(x/2 + (y + 47)))) - x * np.sin(np.sqrt(np.abs(x - (y + 47))))

# Implementacja algorytmu mrówkowego
def ant_colony_optimization(func):
    max_iterations = 500
    num_ants = 100
    alpha = 1.0
    beta = 5.0
    evaporation_rate = 0.5

    x_range = (-512, 512)
    y_range = (-512, 512)

    pheromone = np.ones((num_ants, 2))
    best_path = np.random.uniform(low=[x_range[0], y_range[0]], high=[x_range[1], y_range[1]], size=(num_ants, 2))
    best_score = np.inf

    best_solution = None  # Najlepsze znalezione rozwiązanie

    # Główna pętla ACO
    for iteration in range(max_iterations):
        for ant in range(num_ants):
            # Wybór kolejnego kroku na podstawie feromonów i heurystyki
            probabilities = pheromone[ant] ** alpha * (1.0 / np.maximum(func(best_path[ant, 0], best_path[ant, 1]), 1e-10)) ** beta
            probabilities[np.isinf(probabilities)] = 0  # Zastąpienie wartości NaN i Inf zerami
            probabilities /= np.sum(probabilities)
            next_step = np.random.choice(range(2), p=probabilities)

            # Aktualizacja ścieżki mrówki
            if next_step == 0:
                best_path[ant][0] += np.random.uniform(-1, 1)
                best_path[ant][0] = np.clip(best_path[ant][0], x_range[0], x_range[1])
            else:
                best_path[ant][1] += np.random.uniform(-1, 1)
                best_path[ant][1] = np.clip(best_path[ant][1], y_range[0], y_range[1])

            # Aktualizacja najlepszego rozwiązania
            score = func(best_path[ant][0], best_path[ant][1])
            if score < best_score:
                best_score = score
                best_solution = best_path[ant].copy()

        # Aktualizacja poziomu feromonów
        pheromone *= (1.0 - evaporation_rate)
        for ant in range(num_ants):
            pheromone[ant] += 1.0 / best_score

        # Wydrukowanie informacji o postępie
        print(f"Iteration: {iteration + 1}/{max_iterations} - Best Score: {best_score}")

    # Wydrukowanie wyników
    print("ACO finished:")
    print(f"Best position: {best_solution}")
    print(f"Best score: {best_score}")

    # Wykresy dla funkcji 3D
    x = np.linspace(x_range[0], x_range[1], 100)
    y = np.linspace(y_range[0], y_range[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = func(X, Y)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    ax.scatter(best_solution[0], best_solution[1], func(*best_solution), color='red', s=100, label='Best Solution')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.title("Algorytm mrówkowy " + func.__name__)
    plt.show()

# Testowanie algorytmu ACO dla funkcji Ackley
ant_colony_optimization(ackley)

# Testowanie algorytmu ACO dla funkcji Rastrigin
ant_colony_optimization(rastrigin)

# Testowanie algorytmu ACO dla funkcji Eggholder
ant_colony_optimization(eggholder)
