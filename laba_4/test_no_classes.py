import random
from typing import List, Tuple
import matplotlib.pyplot as plt
from matplotlib import animation
from PIL import Image
import io
import tkinter as tk
from tkinter import ttk

# Глобальные переменные для параметров
mutation_rate = 0
population_size = 50
min_gene_value = -50
max_gene_value = 50
num_generations = 1
current_generation = 0
crossover_rate = 0
best_solution_label = None
function_value_label = None
num_generations_label = None
table = None

def print_global_values():
    print(f"mutation: {mutation_rate}")
    print(f"crossover: {crossover_rate}")
    print(f"population: {population_size}")
    print(f"number generations: {num_generations}")

# Функции для работы с хромосомами
def create_random_chromosome(min_range: float, max_range: float) -> Tuple[float, float]:
    """Создает случайную хромосому (x, y) в заданных пределах"""
    x = random.uniform(min_range, max_range)
    y = random.uniform(min_range, max_range)
    return (x, y)

def fitness(chromosome: Tuple[float, float]) -> float:
    """Функция приспособленности"""
    x, y = chromosome
    return -12 * y + 4 * (x ** 2) + 4 * (y ** 2) - 4 * x * y

def initialize_population(population_size: int, min_range: float, max_range: float) -> List[Tuple[float, float]]:
    """Инициализирует популяцию случайными хромосомами"""
    return [create_random_chromosome(min_range, max_range) for _ in range(population_size)]

def evaluate_fitness(population: List[Tuple[float, float]]) -> List[Tuple[Tuple[float, float], float]]:
    """Оценивает приспособленность каждой хромосомы в популяции"""
    return [(chromosome, fitness(chromosome)) for chromosome in population]

def selection(population: List[Tuple[float, float]], tournament_size: int) -> List[Tuple[float, float]]:
    """Отбор по турнирному правилу"""
    random.shuffle(population)
    group_size = len(population) // tournament_size
    groups = [population[i * group_size:(i + 1) * group_size] for i in range(tournament_size)]
    winners = [min(group, key=fitness) for group in groups]
    return winners

def crossover(parent1: Tuple[float, float], parent2: Tuple[float, float]) -> Tuple[float, float]:
    """Скрещивает две хромосомы"""
    if random.randint(0, 100) < crossover_rate:
        new_x = parent1[0] if random.random() < 0.5 else parent2[0]
        new_y = parent1[1] if random.random() < 0.5 else parent2[1]
        return (new_x, new_y)
    return random.choice([parent1, parent2])

def mutate(chromosome: Tuple[float, float], min_range: float, max_range: float) -> Tuple[float, float]:
    """Мутирует хромосому"""
    if random.randint(0, 100) < mutation_rate:
        new_x = chromosome[0] + random.uniform(-2, 2)
        new_y = chromosome[1] + random.uniform(-2, 2)
        new_x = max(min(new_x, max_range), min_range)
        new_y = max(min(new_y, max_range), min_range)
        return (new_x, new_y)
    return chromosome

def generate_new_population(population: List[Tuple[float, float]], min_range: float, max_range: float, tournament_size: int) -> List[Tuple[float, float]]:
    """Создает новое поколение"""
    selected_parents = selection(population, tournament_size)
    new_population = []
    while len(new_population) < len(population):
        parent1, parent2 = random.sample(selected_parents, 2)
        child = crossover(parent1, parent2)
        child = mutate(child, min_range, max_range)
        new_population.append(child)
    return new_population

def evolve(generations: int, population_size: int, min_range: float, max_range: float, tournament_size: int) -> List[Tuple[float, float]]:
    """Запускает процесс эволюции на заданное количество поколений"""
    population = initialize_population(population_size, min_range, max_range)
    for _ in range(generations):
        population = generate_new_population(population, min_range, max_range, tournament_size)
    return population

def get_best_chromosome(population: List[Tuple[float, float]]) -> Tuple[float, float, float]:
    """Возвращает лучшую хромосому в популяции (fitness, x, y)"""
    best_chromosome = min(population, key=fitness)
    best_fitness = fitness(best_chromosome)
    return best_fitness, best_chromosome[0], best_chromosome[1]

def get_all_chromosomes(population: List[Tuple[float, float]]) -> List[Tuple[float, float, float]]:
    """Возвращает все хромосомы в популяции в формате (fitness, x, y)"""
    return [(fitness(chromosome), chromosome[0], chromosome[1]) for chromosome in population]

# Графика
def best_chromosome_in_generation_graph(generations: int, population_size: int, min_range: float, max_range: float, tournament_size: int):
    """Рисует график лучшей приспособленности в зависимости от поколения"""
    best_fitness_values = []
    population = initialize_population(population_size, min_range, max_range)
    for _ in range(generations):
        best_fitness, _, _ = get_best_chromosome(population)
        best_fitness_values.append(best_fitness)
        population = generate_new_population(population, min_range, max_range, tournament_size)

    plt.figure(figsize=(10, 6))
    plt.plot(range(generations), best_fitness_values, marker='o', color='blue', linestyle='-')
    plt.title('Best Fitness Over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness')
    plt.grid()
    plt.show()

# GUI функции
def update_chromosome_table(all_chromo):
    global table
    for item in table.get_children():
        table.delete(item)
    for i, chromo in enumerate(all_chromo):
        table.insert("", "end", values=(i + 1, chromo[0], chromo[1], chromo[2]))

def display_best_solution(best_chromo):
    global best_solution_label, function_value_label, num_generations_label
    func_val, x_best_sol, y_best_sol = best_chromo
    best_solution_label.config(text=f"Лучшее решение: X[1] = {x_best_sol} \nX[2] = {y_best_sol}")
    function_value_label.config(text=f"Значение функции: {func_val}")
    num_generations_label.config(text=f'Количество поколений: {num_generations}')

def calculate_chromosomes():
    global mutation_rate, population_size, min_gene_value, max_gene_value, num_generations, crossover_rate
    print_global_values()
    tournament_size = population_size // 2
    population = evolve(num_generations, population_size, min_gene_value, max_gene_value, tournament_size)
    all_chromo = get_all_chromosomes(population)
    best_chromo = get_best_chromosome(population)
    update_chromosome_table(all_chromo)
    display_best_solution(best_chromo)

def update_global_variable(entry, variable_name):
    global mutation_rate, population_size, min_gene_value, max_gene_value, num_generations, crossover_rate
    try:
        value = int(entry.get())
        if variable_name == "mutation_rate":
            mutation_rate = value
        elif variable_name == "population_size":
            population_size = value
        elif variable_name == "min_gene_value":
            min_gene_value = value
        elif variable_name == "max_gene_value":
            max_gene_value = value
        elif variable_name == "num_generations":
            num_generations = value
        elif variable_name == "crossover_rate":
            crossover_rate = value
    except ValueError:
        pass
    
def add_to_num_generations(value: int):
    """Увеличивает количество поколений на заданное значение"""
    global num_generations
    num_generations += value
    num_generations_label.config(text=f'Количество поколений: {num_generations}')


def create_gui():
    global best_solution_label, function_value_label, num_generations_label, table
    root = tk.Tk()
    root.title("Genetic Algorithm")

    # Создание элементов GUI
    # Ваш код для создания GUI с использованием tk.Entry, tk.Label и других виджетов

    
    # Параметры
    frame_params = tk.LabelFrame(root, text="Параметры")
    frame_params.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

    tk.Label(frame_params, text="Функция").grid(row=0, column=0, sticky="w")
    tk.Label(frame_params, text="-12y + 4*(x^2) + 4*(y^2) - 4xy", width=30).grid(row=0, column=1)

    tk.Label(frame_params, text="Вероятность мутации, %:").grid(row=1, column=0, sticky="w")
    mutation_rate_entry = tk.Entry(frame_params, width=5)
    mutation_rate_entry.grid(row=1, column=1)
    mutation_rate_entry.insert(0, mutation_rate)  # Установка значения по умолчанию
    mutation_rate_entry.bind("<Leave>", lambda e: update_global_variable(mutation_rate_entry, 'mutation_rate'))
    
    tk.Label(frame_params, text="Вероятность crossover, %:").grid(row=2, column=0, sticky="w")
    crossover_rate_entry = tk.Entry(frame_params, width=5)
    crossover_rate_entry.grid(row=2, column=1)
    crossover_rate_entry.insert(0, crossover_rate)  # Установка значения по умолчанию
    crossover_rate_entry.bind("<Leave>", lambda e: update_global_variable(crossover_rate_entry, 'crossover_rate'))

    tk.Label(frame_params, text="Количество хромосом:").grid(row=3, column=0, sticky="w")
    population_size_entry = tk.Entry(frame_params, width=5)
    population_size_entry.grid(row=3, column=1)
    population_size_entry.insert(0, population_size)  # Установка значения по умолчанию
    population_size_entry.bind("<Leave>", lambda e: update_global_variable(population_size_entry, 'population_size'))

    tk.Label(frame_params, text="Минимальное значение гена:").grid(row=4, column=0, sticky="w")
    min_gene_value_entry = tk.Entry(frame_params, width=5)
    min_gene_value_entry.grid(row=4, column=1)
    min_gene_value_entry.insert(0, min_gene_value)  # Установка значения по умолчанию
    min_gene_value_entry.bind("<Leave>", lambda e: update_global_variable(min_gene_value_entry, 'min_gene_value'))

    tk.Label(frame_params, text="Максимальное значение гена:").grid(row=5, column=0, sticky="w")
    max_gene_value_entry = tk.Entry(frame_params, width=5)
    max_gene_value_entry.grid(row=5, column=1)
    max_gene_value_entry.insert(0, max_gene_value)  # Установка значения по умолчанию
    max_gene_value_entry.bind("<Leave>", lambda e: update_global_variable(max_gene_value_entry, 'max_gene_value'))

    # Управление
    frame_control = tk.LabelFrame(root, text="Управление")
    frame_control.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")


    tk.Button(frame_control, text="Рассчитать хромосомы", command=lambda: calculate_chromosomes()).grid(row=0, column=0, columnspan=3, pady=5)
    
    tk.Label(frame_control, text="Количество поколений:").grid(row=1, column=0, sticky="w")
    number_of_generations = tk.Entry(frame_control, width=5)
    number_of_generations.grid(row=1, column=1)
    number_of_generations.bind("<Leave>", lambda e: update_global_variable(number_of_generations, 'number_of_generations'))
    tk.Button(frame_control, text="1", command=lambda: add_to_num_generations(1)).grid(row=1, column=2)
    tk.Button(frame_control, text="10", command=lambda: add_to_num_generations(10)).grid(row=1, column=3)
    tk.Button(frame_control, text="100", command=lambda: add_to_num_generations(100)).grid(row=1, column=4)
    tk.Button(frame_control, text="1000", command=lambda: add_to_num_generations(1000)).grid(row=1, column=5)


    # Результаты
    frame_results = tk.LabelFrame(root, text="Результаты")
    frame_results.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")

    best_solution_label = tk.Label(frame_results, text="Лучшее решение:")
    best_solution_label.grid(row=0, column=0, sticky="w")

    num_generations_label = tk.Label(frame_results, text="Количество поколений:")
    num_generations_label.grid(row=1, column=0, sticky="w")
    
    function_value_label = tk.Label(frame_results, text="Значение функции:")
    function_value_label.grid(row=2, column=0, sticky="w")

    # Таблица хромосом
    frame_table = tk.LabelFrame(root, text="Хромосомы данного поколения")
    frame_table.grid(row=0, column=1, rowspan=3, padx=10, pady=10, sticky="nsew")

    table = ttk.Treeview(frame_table, columns=("Номер", "Результат", "Ген 1", "Ген 2"), show="headings")
    table.heading("Номер", text="Номер")
    table.heading("Результат", text="Результат")
    table.heading("Ген 1", text="Ген 1")
    table.heading("Ген 2", text="Ген 2")
    table.pack(fill="both", expand=True)

    root.mainloop()

# Запуск GUI
create_gui()
