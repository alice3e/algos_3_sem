import random
from typing import List, Tuple
import matplotlib.pyplot as plt
from matplotlib import animation
from PIL import Image
import io
import tkinter as tk
from tkinter import ttk
import time  # Импортируем библиотеку для работы со временем
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

# Глобальные переменные для параметров
#mutation_rate = 0
tournament_size = 5
#population_size = 50
min_gene_value = -50
max_gene_value = 50
num_generations = 50
current_generation = 0
crossover_rate = 82
mutation_rate = 69.7

correct_answer = -12
correct_x_value = 1
correct_y_value = 2


class Chromosome:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    @classmethod
    def create_random(cls, min_range: float, max_range: float):
        """Создает новый объект Chromosome с случайными x и y в заданных пределах"""
        #Tuple[float, float])

        x = random.uniform(min_range, max_range)
        y = random.uniform(min_range, max_range)
        return cls(x, y)

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def __repr__(self):
        return f"Chromosome(x={self.x:.2f}, y={self.y:.2f})"


class Generation:
    def __init__(self, population_size: int, min_range: float, max_range: float, 
                 mutation_rate: float, crossover_rate: float, tournament_size: int):
        self.population_size = population_size
        self.min_range = min_range
        self.max_range = max_range
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size  # Количество групп в турнире
        self.population: List[Chromosome] = []

    def initialize_population(self):
        """Инициализирует нулевое поколение случайными хромосомами"""
        self.population = [Chromosome.create_random(self.min_range, self.max_range) for _ in range(self.population_size)]

    def fitness(self, chromosome: Chromosome) -> float:
        """Функция приспособленности: сюда нужно вставить целевую функцию"""
        # x1 = x , x2 = y
        # Пример функции приспособленности: минимизация функции f(x, y) = -12y + 4*(x^2) + 4*(y^2) - 4xy
        x = chromosome.get_x()
        y = chromosome.get_y()
        #
        return ( -12*y + (4*(x**2)) + (4*(y**2)) - 4*x*y)

    def evaluate_fitness(self) -> List[Tuple[Chromosome, float]]:
        """Оценивает приспособленность каждого хромосома в поколении"""
        return [(chromosome, self.fitness(chromosome)) for chromosome in self.population]

    def selection(self) -> List[Chromosome]:
        """Отбор по турнирному правилу"""
        if self.tournament_size > self.population_size:
            raise ValueError("Количество групп в турнире не должно превышать количество хромосом")

        # Разбиваем популяцию на N случайных групп
        random.shuffle(self.population)
        group_size = self.population_size // self.tournament_size
        groups = [self.population[i * group_size:(i + 1) * group_size] for i in range(self.tournament_size)]

        # Выбираем лучшую хромосому из каждой группы
        winners = []
        for group in groups:
            best_chromosome = min(group, key=self.fitness)  # Лучший по минимальному значению fitness
            winners.append(best_chromosome)
        #print(f'amount of winners = {len(winners)}')
        return winners

    # Однородный кроссинговер (Uniform Crossover)
    def crossover(self, parent1: Chromosome, parent2: Chromosome) -> Chromosome:
        """Скрещивает две хромосомы для создания потомка"""
        boarder = random.randint(0,100)
        if  boarder < self.crossover_rate:
            #print(f"crossover! {boarder,self.crossover_rate}")
            # Простое одноточечное скрещивание
            new_x = parent1.get_x() if random.random() < 0.5 else parent2.get_x()
            new_y = parent1.get_y() if random.random() < 0.5 else parent2.get_y()
            
            return Chromosome(new_x, new_y)
        # Если кроссовер не произошел, возвращаем копию одного из родителей
        return random.choice([parent1, parent2])

    def mutate(self, chromosome: Chromosome) -> Chromosome:
        """Мутирует хромосому с заданной вероятностью мутации"""
        boarder = random.randint(0,100)
        if  boarder < self.mutation_rate:
            #print(f"mutation! {boarder,self.mutation_rate}")
            # Изменяем x и/или y с учетом пределов
            new_x = chromosome.get_x() + random.uniform(-2, 2)
            new_y = chromosome.get_y() + random.uniform(-2, 2)
            # Ограничиваем значения в пределах x_range и y_range
            new_x = min(new_x, self.max_range)
            new_y = min(new_y, self.max_range)
            new_x = max(new_x, self.min_range)
            new_y = max(new_y, self.min_range)
            return Chromosome(new_x, new_y)
        return chromosome

    def generate_new_population(self):
        """Создает новое поколение на основе текущей популяции"""
        # Отбор лучших индивидов
        selected_parents = self.selection()
        #print(selected_parents)
        new_population = []

        # Создаем новое поколение с кроссинговером и мутацией
        while len(new_population) < self.population_size:
            parent1, parent2 = random.sample(selected_parents, 2)
            child = self.crossover(parent1, parent2)
            child = self.mutate(child)
            new_population.append(child)

        self.population = new_population

    def evolve(self, generations: int):
        """Запускает процесс эволюции на заданное количество поколений"""
        self.initialize_population()
        for _ in range(generations):
            self.generate_new_population()
        return self.population
    
    def get_best_chromosome(self) -> Tuple[float, float, float]:
        """Возвращает лучшую хромосому в поколении (fitness, x, y)"""
        best_chromosome = min(self.population, key=self.fitness)  # Нахождение хромосомы с минимальным fitness
        best_fitness = self.fitness(best_chromosome)
        best_x = best_chromosome.get_x()
        best_y = best_chromosome.get_y()
        return best_fitness, best_x, best_y
    
    def get_all_chromosomes(self) -> List[Tuple[float, float, float]]:
        """Возвращает все хромосомы в поколении в формате (fitness, x, y)"""
        all_chromosomes = []
        for chromosome in self.population:
            fitness = self.fitness(chromosome)
            x = chromosome.get_x()
            y = chromosome.get_y()
            all_chromosomes.append((fitness, x, y))
        return all_chromosomes

def test_generations(population_size: int, min_generations: int, max_generations: int, step: int, trials: int) -> List[Tuple[int, float]]:
    """Тестирует различные количества поколений и возвращает средний размер популяции и соответствующие ошибки"""
    results = []
    for generations in range(min_generations, max_generations + 1, step):
        total_error = 0.0  # Сумма ошибок для усреднения
        for _ in range(trials):  # Повторяем тест для каждого количества поколений
            generation = Generation(population_size, min_gene_value, max_gene_value,
                                    mutation_rate, crossover_rate, tournament_size)
            final_population = generation.evolve(generations)
            
            # Получаем лучшую хромосому и её ошибку
            best_fitness, best_x, best_y = generation.get_best_chromosome()
            # Ошибка = |correct_answer - best_fitness|
            error = abs(correct_answer - best_fitness)
            total_error += error  # Суммируем ошибку
            
        # Вычисляем среднюю ошибку
        average_error = total_error / trials
        results.append((generations, average_error))
    return results

def run_test(args):
    """Запускает тест для заданных аргументов и выводит прогресс в консоль."""
    population_size, min_generations, max_generations, step, trials = args
    error_data = test_generations(population_size, min_generations, max_generations, step, trials)
    print(f'Обработано данных для размера популяции: {population_size}')
    return error_data

if __name__ == '__main__':
    # Параметры теста
    min_population_size = 11
    max_population_size = 40
    step_size = 1
    min_generations = 1
    max_generations = 40
    trials = 25  # Количество повторений для каждого размера популяции

    # Сбор данных для 3D графика
    population_sizes = list(range(min_population_size, max_population_size + 1, step_size))
    generations = list(range(min_generations, max_generations + 1))

    errors_matrix = np.zeros((len(population_sizes), len(generations)))

    # Создаем список для аргументов для параллельного выполнения
    args = [(population_size, min_generations, max_generations, 1, trials) for population_size in population_sizes]

    with ProcessPoolExecutor() as executor:
        results = executor.map(run_test, args)
        
        # Обработка результатов
        for i, error_data in enumerate(results):
            for j, (gen, error) in enumerate(error_data):
                errors_matrix[i, j] = error

    # Построение 3D графика
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    X, Y = np.meshgrid(generations, population_sizes)
    ax.plot_surface(X, Y, errors_matrix, cmap='viridis')

    ax.set_title('Средняя ошибка в зависимости от количества поколений и размера популяции')
    ax.set_xlabel('Количество поколений')
    ax.set_ylabel('Размер популяции')
    ax.set_zlabel('Средняя ошибка (отклонение от correct_answer)')
    
    plt.show()
