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

# Implementacja algorytmu PSO
def pso(func):
    max_iterations = 100
    num_particles = 50
    c1 = 2.0
    c2 = 2.0
    w = 0.7

    x_range = (-1, 1)
    y_range = (-1, 1)

    # Inicjalizacja cząstek
    particles = np.random.uniform(low=[x_range[0], y_range[0]], high=[x_range[1], y_range[1]], size=(num_particles, 2))
    velocities = np.zeros_like(particles)
    personal_best_positions = particles.copy()
    personal_best_scores = np.zeros(num_particles) + np.inf
    global_best_position = np.zeros(2)
    global_best_score = np.inf

    # Główna pętla PSO
    for iteration in range(max_iterations):
        # Obliczanie wartości funkcji celu dla każdej cząstki
        scores = func(particles[:, 0], particles[:, 1])

        # Aktualizacja najlepszych osobistych pozycji i wyników
        improved_personal_best = scores < personal_best_scores
        personal_best_scores[improved_personal_best] = scores[improved_personal_best]
        personal_best_positions[improved_personal_best] = particles[improved_personal_best]

        # Aktualizacja globalnej najlepszej pozycji i wyniku
        best_particle = np.argmin(personal_best_scores)
        if personal_best_scores[best_particle] < global_best_score:
            global_best_score = personal_best_scores[best_particle]
            global_best_position = personal_best_positions[best_particle]

        # Aktualizacja prędkości i pozycji cząstek
        r1 = np.random.rand(num_particles, 2)
        r2 = np.random.rand(num_particles, 2)
        velocities = w * velocities + c1 * r1 * (personal_best_positions - particles) + c2 * r2 * (global_best_position - particles)
        particles += velocities

                # Ograniczenie cząstek do obszaru poszukiwań
        particles = np.clip(particles, x_range[0], x_range[1])
        particles = np.clip(particles, y_range[0], y_range[1])

        # Wydrukowanie informacji o postępie
        print(f"Iteration: {iteration + 1}/{max_iterations} - Best Score: {global_best_score}")

    # Wydrukowanie wyników
    print("PSO finished:")
    print(f"Best position: {global_best_position}")
    print(f"Best score: {global_best_score}")

    # Wykresy dla funkcji 3D
    x = np.linspace(x_range[0], x_range[1], 100)
    y = np.linspace(y_range[0], y_range[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = func(X, Y)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    ax.scatter(global_best_position[0], global_best_position[1], global_best_score, color='red', s=100, label='Global Best')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.title("PSO | " + func.__name__)
    plt.show()

# Testowanie algorytmu PSO dla funkcji Ackley
pso(ackley)

# Testowanie algorytmu PSO dla funkcji Rastrigin
pso(rastrigin)

# Testowanie algorytmu PSO dla funkcji Eggholder
pso(eggholder)


       
