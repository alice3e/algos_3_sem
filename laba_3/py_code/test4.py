import hashlib
from itertools import product
from multiprocessing import Pool, cpu_count
import pandas as pd


charset = '0123456789'
salt = 116193795

unhashed_numbers_list = list()
file_path = 'data/my_var_2nd.xlsx'

# Читаем Excel файл, загружаем только первые и третьи столбцы
df = pd.read_excel(file_path, usecols=[0, 2],names=['Hash', 'phone_unhashed'], dtype={1: str})
hash_dict = df.set_index('Hash')['phone_unhashed'].to_dict()
for key in hash_dict:
    if not isinstance(hash_dict[key], float):
        unhashed_numbers_list.append( [key,hash_dict[key]] )


def run_md5(number):
    string = str(number + salt).encode()
    h = hashlib.md5(string).hexdigest()
    return h

def chunked(iterable, size):
    """Разделяет итератор на чанки заданного размера."""
    for i in range(0, len(iterable), size):
        yield iterable[i:i + size]

def generate_combinations_and_hashes():
    # Генерируем все возможные комбинации
    combinations = [''.join(combo) for combo in product(charset, repeat=9)]

    # Определяем размер чанка, основываясь на количестве доступных процессов
    num_processes = cpu_count()
    chunk_size = max(1, len(combinations) // num_processes)

    # Создаем чанки
    chunks = list(chunked(combinations, chunk_size))

    results = []
    
    # Параллельное выполнение для каждого чанка
    with Pool() as pool:
        for chunk in chunks:
            chunk_results = pool.map(run_md5, chunk)
            # Проверяем наличие хэша в словаре
            for hash_value, number in chunk_results:
                if hash_value in hash_dict:
                    # Записываем номер в значение словаря
                    hash_dict[hash_value] = number
                results.append(hash_value)

    return results







