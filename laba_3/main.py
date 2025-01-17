import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import hashlib
import os
import time
from passlib.hash import nthash

# Глобальные переменные
file_path = None
phones = None
numbers = None
salt = 0 # 58909967

def ntlm_hash(string):
    if isinstance(string, bytes):
        string = string.decode('utf-8')  # Если это байты, декодируем в строку
    return nthash.hash(string).replace('nthash$', '')  # Хэширование с passlib

def hash_phones(phones, hash_type):
    hash_functions = {
        'sha1': (hashlib.sha1, 'sha1.txt', 'output_sha1.txt', 100),
        'sha256': (hashlib.sha256, 'sha256.txt', 'output_sha256.txt', 1400),
        'sha512': (hashlib.sha512, 'sha512.txt', 'output_sha512.txt', 1700),
        'md5': (hashlib.md5, 'md5.txt', 'output_md5.txt', 0),
        #'ntlm': (ntlm_hash, 'ntlm.txt', 'output_ntlm.txt', 0),
        #'mysql323': (mysql323_hash, 'mysql323.txt', 'output_mysql323.txt', 0),x
    }

    if hash_type not in hash_functions:
        raise ValueError("Неверный тип хеширования.")

    hash_function, filename, output_file, mode = hash_functions[hash_type]

    # Хеширование номеров телефонов
    start_time = time.time()
    
    hashed_phones = [hash_function((str(int(phone)+salt)).encode()).hexdigest() for phone in phones]



    # Запись зашифрованных номеров в файлx
    with open(filename, 'w') as f:
        for hashed_phone in hashed_phones:
            f.write(hashed_phone + '\n')



    # Выполнение команды hashcat для расшифровки
    os.system(f"hashcat --potfile-disable -a 3 -m {mode} -o {output_file} {filename} 89?d?d?d?d?d?d?d?d?d")

    # Время окончания хеширования
    end_time = time.time()
    time_taken = end_time - start_time
    # Проверка успешности создания файла расшифровки
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            decrypted_lines = f.readlines()
            # Вывод расшифрованных данных
            print(f"Decrypted data: {decrypted_lines[:10]}")  # Отобразим только первые 10 строк
            messagebox.showinfo("Готово", f"Данные зашифрованы и расшифрованы алгоритмом {hash_type}. Время выполнения: {time_taken:.6f} секунд.")
    
    else:
        messagebox.showwarning("Ошибка", f"Файл {output_file} не был создан.")
    # Запись информации в лог
    with open("hash_time.log", "a") as log_file:
        log_file.write(f"Hash type: {hash_type}, Encrypted rows: {len(phones)}, Time taken: {time_taken:.6f} seconds\n")

    
def process_data():
    global file_path, phones, numbers
    df = pd.read_excel(file_path)
    hashes = df["Номер телефона"]
    numbers = [number[:-2] for number in df["Unnamed: 2"].astype(str).tolist()][:5]

    with open('hashed_phones.txt', 'w') as f:
        for HASH in hashes:
            f.write(HASH + "\n")
    os.system("hashcat --potfile-disable -a 3 -m 0 -o cracked_output.txt hashed_phones.txt 89?d?d?d?d?d?d?d?d?d")

    with open('cracked_output.txt') as r:
        phones = [line.strip()[33:] for line in r.readlines()]

    with open('phones.txt', 'w') as file:
        for phone in phones:
            file.write(phone + '\n')
    messagebox.showinfo("Готово", "Таблица успешно расшифрована. Данные сохранены в файле 'phones.txt'.")

def calculate_salt(decoded_phones, table_numbers):
    for decoded_phone in decoded_phones:
        computed_salt = int(decoded_phone) - int(table_numbers[0])
        if computed_salt < 0:
            continue
        idx = 1
        while (str(int(table_numbers[idx]) + computed_salt)) in decoded_phones:
            idx += 1
            if idx == 5:
                return computed_salt
    return 0

def discover_salt():
    global phones, numbers, salt
    # Запуск таймера
    start_time = time.time()
    # Расчет соли
    salt = calculate_salt(phones, numbers)
    # Время окончания
    end_time = time.time()
    time_taken = end_time - start_time
    # Запись информации в лог
    with open("decrypt_time.log", "a") as log_file:
        log_file.write(f"Decrypted salt: {salt}, Processed rows: {len(phones)}, Time taken: {time_taken:.6f} seconds\n")
    # Показ messagebox с результатом
    messagebox.showinfo("Готово", f"Значение соли: {salt}. Время выполнения: {time_taken:.6f} секунд.")

def load_file():
    global file_path
    file_path = filedialog.askopenfilename(title="Выберите файл", filetypes=[("Excel files", "*.xlsx")])
    if file_path:
        messagebox.showinfo("Файл загружен", f"Вы загружали файл: {file_path}")

def select_hash_type(hash_type):
    global phones
    if phones is not None:
        hash_phones(phones, hash_type)

    else:
        messagebox.showerror("Ошибка", "Сначала загрузите и обработайте данные.")

def create_gui():
    root = tk.Tk()
    root.title("Хеширование номеров телефонов")

    # Кнопка для загрузки файла
    load_button = tk.Button(root, text="Загрузить файл", command=load_file)
    load_button.pack(pady=10)

    # Кнопка для расшифровки данных
    decrypt_button = tk.Button(root, text="Расшифровать данные", command=process_data)
    decrypt_button.pack(pady=10)

    # Кнопка для вычисления соли
    salt_button = tk.Button(root, text="Вычислить соль", command=discover_salt)
    salt_button.pack(pady=10)

    # Метка для выбора кодировки
    tk.Label(root, text="Выберите тип хеширования:").pack(pady=5)

    # Кнопки для выбора кодировок
    hash_types = ['sha1', 'sha256', 'sha512', 'md5']
    for hash_type in hash_types:
        button = tk.Button(root, text=hash_type, command=lambda ht=hash_type: select_hash_type(ht))
        button.pack(pady=2)

    # Запуск главного цикла приложения
    root.mainloop()
    
if __name__ == '__main__':
    create_gui()