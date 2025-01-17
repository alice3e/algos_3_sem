import random
from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import concurrent.futures
import time

# Глобальные параметры
current_speed_coeff = 0.26
personal_best_coeff = 1.5
global_best_coeff = 1.5
num_particles = 40
num_iterations = 40
executed_iterations = 0
velocity_limit = 15  # Максимальная скорость
correct_function_minimum = -12
correct_x_value = 1
correct_y_value = 2

# Глобальные переменные для роевого алгоритма
particles = []
global_best_position = None
global_best_value = float('inf')
best_solution_label = None
function_value_label = None
num_iterations_label = None
num_current_iterations_label = None

# Функция 
def fitness_function(position):
    x, y = position
    return -12*y + 4*(x**2) + 4*(y**2) - 4*x*y

# Класс для частиц
class Particle:
    def __init__(self):
        self.position = np.array([random.uniform(-500, 500), random.uniform(-500, 500)])
        self.velocity = np.array([random.uniform(-10, 10), random.uniform(-10, 10)])
        self.personal_best_position = self.position.copy()
        self.personal_best_value = fitness_function(self.position)

    def update_velocity(self):
        global global_best_position
        inertia = current_speed_coeff * self.velocity
        cognitive = personal_best_coeff * random.random() * (self.personal_best_position - self.position)

        # Проверка на наличие глобального лучшего положения
        if global_best_position is not None:
            social = global_best_coeff * random.random() * (global_best_position - self.position)
        else:
            social = np.array([0.0, 0.0])  # Если нет глобального лучшего положения, social равен нулю

        # Обновляем скорость частицы
        self.velocity = inertia + cognitive + social
        
        # Ограничиваем скорость
        speed = np.linalg.norm(self.velocity)  # Нормируем скорость
        if speed > velocity_limit:  # Если скорость превышает предел
            self.velocity = (self.velocity / speed) * velocity_limit  # Нормализуем скорость до предела

    def update_position(self):
        self.position += self.velocity
        # Ограничиваем положение частицы в пределах поиска
        self.position = np.clip(self.position, -500, 500)

    def update_personal_best(self):
        fitness = fitness_function(self.position)
        if fitness < self.personal_best_value:
            self.personal_best_position = self.position.copy()
            self.personal_best_value = fitness


def initialize_particles(num_particles):
    global global_best_position, global_best_value
    particles = [Particle() for _ in range(num_particles)]
    global_best_position = None
    global_best_value = float('inf')
    return particles

def run_iteration(particles):
    global global_best_position, global_best_value
    for particle in particles:
        particle.update_velocity()
        particle.update_position()
        particle.update_personal_best()

        if particle.personal_best_value < global_best_value:
            global_best_position = particle.personal_best_position.copy()
            global_best_value = particle.personal_best_value
    return particles

# Функция для одного запуска алгоритма
def single_run(num_iterations):
    global num_particles  # Зафиксированное количество частиц
    particles = initialize_particles(num_particles)
    
    for _ in range(num_iterations):  # Используем num_iterations для количества итераций
        particles = run_iteration(particles)
        
    error = (global_best_value - correct_function_minimum) ** 2
    return error


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

def plot_error_decay(num_iterations=40):
    # Инициализация данных для графика
    pso_errors = []
    ga_errors = []
    
    # Запуск Роевого Алгоритма
    particles = initialize_particles(num_particles)
    for i in range(num_iterations):
        run_iteration(particles)
        pso_error = (global_best_value - correct_function_minimum) ** 2
        pso_errors.append(pso_error)
        
    # Запуск Генетического Алгоритма
    generation = Generation(
        population_size=50, 
        min_range=-50, 
        max_range=50, 
        mutation_rate=69.7, 
        crossover_rate=82, 
        tournament_size=5
    )
    generation.initialize_population()
    for i in range(num_iterations):
        generation.generate_new_population()
        best_fitness, _, _ = generation.get_best_chromosome()
        ga_error = (best_fitness - correct_function_minimum) ** 2
        ga_errors.append(ga_error)
    
    print(ga_errors, pso_error)
    # Построение графика
    plt.figure(figsize=(10, 6))
    plt.plot(range(num_iterations), pso_errors, label='Роевой алгоритм (PSO)')
    plt.plot(range(num_iterations), ga_errors, label='Генетический алгоритм (GA)')
    plt.xlabel('Итерации')
    plt.ylabel('Ошибка (квадрат разности с правильным минимумом)')
    plt.title('Снижение ошибки минимума функции на итерациях')
    plt.ylim((0,100))
    plt.legend()
    plt.grid(True)
    plt.show()

# Вызов функции для построения графика
plot_error_decay(num_iterations=40)