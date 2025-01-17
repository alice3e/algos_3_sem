import hashlib
import logging
from typing import Callable, List
from itertools import product
from multiprocessing import Pool, cpu_count
import time

logging.basicConfig(
    filename='salt_checked.log',  # Имя лог-файла
    filemode='a',        # 'a' для добавления записей в файл, 'w' для перезаписи файла
    format='%(asctime)s - %(levelname)s - %(message)s',  # Формат логов
    level=logging.INFO   # Уровень логирования: DEBUG, INFO, WARNING, ERROR, CRITICAL
)

hash_functions = {
    'md5': hashlib.md5, 
}

def check_hash_function(input_data: str, output_hash: str, salt: str, hash_func_name: str):
    input_data_with_salt = salt + input_data 
    hash_func = hash_functions[hash_func_name]
    h = hash_func(input_data_with_salt.encode()).hexdigest()
    if h == output_hash:
        logging.info(f'верная соль: \"{salt}\" для пары {input_data} -> {h} и функции {hash_func_name}')
        return True
    else:
        if(int(salt) % 50_000_000 == 0):
            print(f'неверная соль: \"{salt}\" для пары {input_data} -> {h} и функции {hash_func_name}')
        return False

def salt_generator(min_length: int, max_length: int, type: int) -> List[str]:
    match type:
        case 0:
            charset = '0123456789'  # Только цифры
        case 1:
            charset = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'  # Только буквы
        case 2:
            charset = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'  # Цифры + буквы
        case _:
            raise ValueError("Недопустимый тип. Используйте 0 (цифры), 1 (буквы), или 2 (цифры и буквы).")

    out = []

    # Генерация всех возможных комбинаций от min_length до max_length
    for length in range(min_length, max_length + 1):
        out.extend(''.join(combo) for combo in product(charset, repeat=length))

    return out

def worker(args):
    input_data, output_hash, salt_chunk, hash_func_name = args
    for salt in salt_chunk:
        if check_hash_function(input_data, output_hash, salt, hash_func_name):
            return True
    return False

def chunkify(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def brute_force_salt_list(input_data: str, output_hash: str, salt_list: List[str], hash_func_name: str) -> bool:
    num_cpus = 10
    chunk_size = len(salt_list) // num_cpus
    salt_chunks = list(chunkify(salt_list, chunk_size))

    args_list = [(input_data, output_hash, chunk, hash_func_name) for chunk in salt_chunks]

    with Pool(processes=num_cpus) as pool:
        results = pool.map(worker, args_list)

    for result in results:
        if result:
            return True
    return False

if __name__ == '__main__':
    input_data = "89868999648"
    output_hash = "00f162fae89b11e04f94016fc752ccef"
    hash_func_name = 'md5'
    min_len = 7
    max_len = 8
    salt = '116193795'
    salt_list = salt_generator(min_length=min_len, max_length=max_len,type=0)

    print(f"Generated {len(salt_list)} salts.")
    print(f"First 10 salts: {salt_list[:10]}")
    print(f"Last 10 salts: {salt_list[-10:]}")
    print(f'output hash:                               {output_hash}')
    
    
    start_time = time.time()
    if brute_force_salt_list(input_data, output_hash, salt_list, hash_func_name):
        print(f"Correct salt found! for {hash_func_name}")
    else:
        print(f"No correct salt found for {hash_func_name}")
        logging.info(f'all possible salt vars with len from {min_len} to {max_len} for {hash_func_name}')
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds for {hash_func_name}")


    

    

