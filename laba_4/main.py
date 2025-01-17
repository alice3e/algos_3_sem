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
    # print(min_gene_value)
    # print(max_gene_value)
    
    # print(current_generation)
    # print(best_solution_label)
    # print(function_value_label)

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
    
    def best_chromosome_in_generation_graph(self, generations: int, x_limits: Tuple[float, float] = None, y_limits: Tuple[float, float] = None):
        """Рисует график лучшей приспособленности в зависимости от номера поколения

        Args:
            generations (int): Количество поколений для эволюции.
            x_limits (Tuple[float, float], optional): Минимальные и максимальные пределы по оси X.
            y_limits (Tuple[float, float], optional): Минимальные и максимальные пределы по оси Y.
        """

        # Список для хранения значений лучшей приспособленности
        best_fitness_values = []

        # Запускаем эволюцию и собираем данные о лучшей приспособленности
        self.initialize_population()

        for _ in range(generations):
            # Получаем лучшую хромосому и ее приспособленность
            best_fitness, _, _ = self.get_best_chromosome()
            best_fitness_values.append(best_fitness)  # Сохраняем значение лучшей приспособленности
            
            # Генерируем новое поколение
            self.generate_new_population()

        # Рисуем график
        plt.figure(figsize=(10, 6))
        plt.plot(range(generations), best_fitness_values, marker='o', color='blue', linestyle='-')
        plt.title('Best Fitness Over Generations')
        plt.xlabel('Generation')
        plt.ylabel('Best Fitness')

        # Установка пределов по осям, если заданы
        if x_limits is not None:
            plt.xlim(x_limits)
        if y_limits is not None:
            plt.ylim(y_limits)

        plt.grid()
        plt.show()




# Функция для запуска генетического алгоритма
def calculate_chromosomes():
    global mutation_rate, population_size, min_gene_value, max_gene_value, num_generations, current_generation,crossover_rate
    # Логика генетического алгоритма будет здесь
    # После расчетов обновляем таблицу и выводим лучший результат
    print_global_values()
    tournament_size = population_size // 2
    genetic_alg = Generation(population_size, min_gene_value, max_gene_value, mutation_rate, crossover_rate, tournament_size)
    genetic_alg.evolve(num_generations)
    all_chromo = genetic_alg.get_all_chromosomes()
    best_chromo = genetic_alg.get_best_chromosome()
    #generation = Generation(population_size, min_gene_value, max_gene_value, mutation_rate, crossover_rate, tournament_size)
    #generation.best_chromosome_in_generation_graph(num_generations, x_limits=(0, num_generations), y_limits=(-12, 10))
    update_chromosome_table(all_chromo)
    display_best_solution(best_chromo)

def update_chromosome_table(all_chromo):
    global table
    for item in table.get_children():
        table.delete(item)
    # Обновляем таблицу хромосом новыми значениями
    for i in range(len(all_chromo)):
        table.insert("", "end", values=(i + 1, all_chromo[i][0], all_chromo[i][1], all_chromo[i][2]))

def display_best_solution(best_chromo):
    global best_solution_label, function_value_label, num_generations_label,num_generations
    # Пример вывода лучшего решения

    func_val, x_best_sol, y_best_sol  = best_chromo
    best_solution_label.config(text=f"Лучшее решение: X[1] = {x_best_sol} \nX[2] = {y_best_sol}")
    function_value_label.config(text=f"Значение функции: {func_val}")
    num_generations_label.config(text=f'Количество поколений: {num_generations}')

def update_global_variable(entry, variable):
    """Обновляет глобальную переменную на основе значения в Entry."""
    global mutation_rate, population_size, min_gene_value, max_gene_value, num_generations,crossover_rate
    try:
        value = int(entry.get())  # Пробуем получить значение как целое число
        if variable == 'mutation_rate':
            mutation_rate = value
        elif variable == 'population_size':
            population_size = value
        elif variable == 'min_gene_value':
            min_gene_value = value
        elif variable == 'max_gene_value':
            max_gene_value = value
        elif variable == 'crossover_rate':
            crossover_rate = value
        elif variable == 'number_of_generations':
            num_generations = value
            #print(f"num_generations теперь равно: {num_generations}")
    except ValueError:
        pass  # Игнорируем ошибки преобразования
    
def add_to_num_generations(value):
    global num_generations
    num_generations = value
    #print(f"num_generations теперь равно: {num_generations}")

def create_gui():
    global best_solution_label, function_value_label, num_generations_label, table, crossover_rate
    # Создание основного окна
    root = tk.Tk()
    root.title("Генетический алгоритм")

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


if __name__ == "__main__":
    create_gui()
