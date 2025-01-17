import os
import pandas as pd
import hashlib
import tkinter as tk
from tkinter import filedialog, messagebox

file_path = None
phones = None
numbers = None
is_file_loaded = False


# salt = 58644554
# numbers - данные из таблицы
# phones - получившиеся после расшифровки без соли
def compute_salt(phones, numbers):
    for phone in phones:
        salt = int(phone) - int(numbers[0])
        if salt < 0:
            continue
        i = 1
        while (str(int(numbers[i]) + salt)) in phones:
            i += 1
            if i == 5:
                return salt
    return 0


def sha1(phones):
    phones_sha1 = [hashlib.sha1(phone.encode()).hexdigest() for phone in phones]
    with open('sha1.txt', 'w') as f:
        for phone in phones_sha1:
            f.write(phone + '\n')

    os.remove('hashcat.potfile')
    os.system("hashcat -a 3 -m 100 -o output_sha1.txt sha1.txt ?d?d?d?d?d?d?d?d?d?d?d")


def sha256(phones):
    phones_sha256 = [hashlib.sha256(phone.encode()).hexdigest() for phone in phones]
    with open('sha256.txt', 'w') as f:
        for phone in phones_sha256:
            f.write(phone + '\n')

    os.remove('hashcat.potfile')
    os.system("hashcat -a 3 -m 1400 -o output_sha256.txt sha256.txt ?d?d?d?d?d?d?d?d?d?d?d")


def sha512(phones):
    phones_sha512 = [hashlib.sha512(phone.encode()).hexdigest() for phone in phones]
    with open('sha512.txt', 'w') as f:
        for phone in phones_sha512:
            f.write(phone + '\n')

    os.remove('hashcat.potfile')
    os.system("hashcat -a 3 -m 1700 -o output_sha512.txt sha512.txt ?d?d?d?d?d?d?d?d?d?d?d")


def load_file():
    global file_path, is_file_loaded
    file_path = filedialog.askopenfilename()
    if file_path:
        is_file_loaded = True
        button_deidentify["state"] = tk.NORMAL


def identify():
    global file_path, phones, numbers
    df = pd.read_excel(file_path)
    hashes = df["Номер телефона"]
    numbers = [number[:-2] for number in df["Unnamed: 2"].astype(str).tolist()][:5]
    print(numbers)
    with open('hashes.txt', 'w') as f:
        for HASH in hashes:
            f.write(HASH + "\n")
    os.system("hashcat -a 3 -m 0 -o output.txt hashes.txt ?d?d?d?d?d?d?d?d?d?d?d")

    with open(r'output.txt') as r:
        phones = [line.strip()[33:] for line in r.readlines()]

    with open('phones.txt', 'w') as file:
        for phone in phones:
            file.write(phone + '\n')
    messagebox.showinfo("Готово", "Таблица успешно расшифрована. Данные сохранены в файле 'phones.txt'.")


def find_salt():
    global phones, numbers
    print(numbers)
    salt = compute_salt(phones, numbers)
    messagebox.showinfo("Готово", f"Значение соли: {salt}")


def encrypt(algorithm):
    global is_file_loaded, phones
    if not is_file_loaded:
        return
    if algorithm == "sha1":
        sha1(phones)
        messagebox.showinfo("Готово", "Результат сохранен в файле output_sha1.")
    elif algorithm == "sha256":
        sha256(phones)
        messagebox.showinfo("Готово", "Результат сохранен в файле output_sha256.")
    else:
        sha512(phones)
        messagebox.showinfo("Готово", "Результат сохранен в файле output_sha512.")


root = tk.Tk()
root.title("Шифрование данных")

label_action = tk.Label(root, text="Выберите действие с таблицей:")
button_load = tk.Button(root, text="Загрузить", command=load_file)
button_deidentify = tk.Button(root, text="Деобезличить", command=identify, state=tk.DISABLED)
button_compute_salt = tk.Button(root, text="Вычислить соль", command=find_salt)

button_encrypt_sha1 = tk.Button(root, text="Зашифровать SHA-1", command=lambda: encrypt("sha1"))
button_encrypt_sha256 = tk.Button(root, text="Зашифровать SHA-256", command=lambda: encrypt("sha256"))
button_encrypt_sha512 = tk.Button(root, text="Зашифровать SHA-512", command=lambda: encrypt("sha512"))

label_action.grid(row=0, column=0, padx=10, pady=10, sticky="w")
button_load.grid(row=1, column=0, padx=10, pady=5, sticky="w")
button_deidentify.grid(row=2, column=0, padx=10, pady=5, sticky="w")
button_compute_salt.grid(row=3, column=0, padx=10, pady=5, sticky="w")

button_encrypt_sha1.grid(row=4, column=1, padx=10, pady=5, sticky="e")
button_encrypt_sha256.grid(row=5, column=1, padx=10, pady=5, sticky="e")
button_encrypt_sha512.grid(row=6, column=1, padx=10, pady=5, sticky="e")

root.mainloop()