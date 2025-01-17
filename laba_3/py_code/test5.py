import os
import pandas as pd
import hashlib
import tkinter as tk
from tkinter import filedialog, messagebox

file_path = 'data/my_var_2nd.xlsx'
unhashed_5_phones = []
wrong_5_phones = []
hashes_5_nums = []
salt = 0 #116193795

def run_md5(string):
    h = hashlib.md5(string.encode()).hexdigest()
    return h

def read_file(file_path):
    # Читаем Excel файл
    df = pd.read_excel(file_path, dtype={2: str})

    # Получаем первый столбец
    first_column = df.iloc[:, 0]
    second_column = df.iloc[:, 2]
    
    with open('unhashed_5_numbers.txt', 'w') as f:
        for i in range(5):
            hash = first_column[i]
            unhashed_5_phones.append(second_column[i])
            f.write(f"{hash}\n")

    # Записываем данные в файл hashes.txt
    with open('hashes.txt', 'w') as f:
        for item in first_column:
            f.write(f"{item}\n")
            
def get_numbers_without_salt():
    os.system('rm -rf output.txt')
    os.system('hashcat -a 3 -m 0 --potfile-disable -O -o output.txt unhashed_5_numbers.txt 89?d?d?d?d?d?d?d?d?d')
    with open('output.txt', 'r') as f:
        for item in f.readlines():
            has,num = item.split(':')
            hashes_5_nums.append(has)
            wrong_5_phones.append(num[:-1])
            
# numbers - данные из таблицы
# phones - получившиеся после расшифровки без соли
def calc_salt():
    #unhashed_5_phones = [] - из файла
    #wrong_5_phones = [] - расшифрованные
    for i in range(5):
        k = 0
        for j  in range(5):
            cur_salt = abs(int(unhashed_5_phones[i]) - int(wrong_5_phones[j]))
            new_phone = str(int(unhashed_5_phones[i]) + cur_salt)
            hash = run_md5(new_phone)
            if(hash in hashes_5_nums):
                k += 1
        if(k==5):
            salt = cur_salt
            return salt
    return False

# def add_salt(input_file, output_file, salt="116193795"):
#     with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
#         for line in infile:
#             # Убираем возможные пробелы или символы новой строки
#             hash_value = line.strip()
#             # Добавляем соль и записываем в новый файл
#             outfile.write(f"{hash_value}:{salt}\n")

def unhash_all():
    os.system('rm -rf final_results.txt')
    os.system('hashcat -a 3 -m 0 --potfile-disable -O -o final_results_without_salt.txt hashes.txt 89?d?d?d?d?d?d?d?d?d')
    with open('final_results_without_salt.txt', 'r') as infile, open('final_results.txt', 'w') as outfile:
        for line in infile.readlines():
            hash,num = line.split(':')
            print(num)
            num_line = int(num.strip()) - salt
            outfile.write(f"{hash}:{num_line}\n")
            
            
read_file(file_path=file_path)
get_numbers_without_salt()
salt = calc_salt()
print(f'salt = {salt}')
#add_salt('hashes.txt','hashes_with_salt.txt',salt)
unhash_all()