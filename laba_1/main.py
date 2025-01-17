import random
import csv
from datetime import datetime, timedelta
import os
from tkinter import *
from tkinter import messagebox
from tkinter.ttk import Combobox


bank_names = ['GAZPROMBANK','MTS BANK','SBERBANK OF RUSSIA','TINKOFF BANK','VTB BANK']
painment_system_names = ['MIR','VISA','MASTERCARD']


def read_data(file_path: str, delimiter: str) -> list:
    data = []
    with open(file_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter=delimiter)
        for row in reader:
            if row:  # Проверка на пустую строку
                data.append(row)
    return data
    
def generate_random_datetime(min_time="09:00", max_time="21:00"):
    # Задаем диапазон дат: начальная и конечная
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 12, 31)
    
    # Вычисляем количество дней между начальной и конечной датами
    time_between_dates = end_date - start_date
    total_days = time_between_dates.days
    
    # Генерируем случайный день в пределах диапазона
    random_days = random.randint(0, total_days)
    random_date = start_date + timedelta(days=random_days)
    
    # Преобразуем минимальное и максимальное время в объекты time
    min_time_obj = datetime.strptime(min_time, "%H:%M").time()
    max_time_obj = datetime.strptime(max_time, "%H:%M").time()
    
    # Генерируем случайное время в пределах указанного диапазона
    min_seconds = min_time_obj.hour * 3600 + min_time_obj.minute * 60
    max_seconds = max_time_obj.hour * 3600 + max_time_obj.minute * 60
    random_seconds = random.randint(min_seconds, max_seconds)
    
    # Преобразуем случайное количество секунд в объект времени
    random_time = (datetime.min + timedelta(seconds=random_seconds)).time()
    
    # Соединяем случайную дату и время
    random_datetime = datetime.combine(random_date, random_time)
    
    # Возвращаем дату и время в формате ISO 8601: YYYY-MM-DDTHH:MM
    return random_datetime.strftime("%Y-%m-%dT%H:%M")

def generate_one_card_2(pay_system, bank):
    card_format = '{fig12} {fig3} {fig4}'
    #bank_names = ['SBERBANK OF RUSSIA','TINKOFF BANK','VTB BANK', 'GAZPROMBANK']
    if pay_system == 'MIR':
        if bank == 'SBERBANK OF RUSSIA':
            figures = '2202 20'
        elif bank == 'TINKOFF BANK':
            figures = '2200 70'
        elif bank == 'VTB BANK':
            figures = '2200 40' 
        else:
            figures = '2200 56' #GAZPROMBANK
    elif pay_system == 'MASTERCARD':
        if bank == 'SBERBANK OF RUSSIA':
            figures = '5228 60'
        elif bank == 'TINKOFF BANK':
            figures = '5389 94'
        elif bank == 'VTB BANK':
            figures = '5211 94'
        else:
            figures = '5112 23' #GAZPROMBANK
    else: # VISA
        if bank == 'SBERBANK OF RUSSIA':
            figures = '4039 33'
        elif bank == 'TINKOFF BANK':
            figures = '4377 73'
        elif bank == 'VTB BANK':
            figures = '4986 29'
        else:
            figures = '4306 43' #GAZPROMBANK
    argz = {'fig12': figures + str(random.randint(10, 99)), 
            'fig3': str(random.randint(1000, 9999)), 
            'fig4': str(random.randint(1000, 9999))}
    
    return card_format.format(**argz)
    
def possibility_generator(poss_vec, cat_vec):
    return random.choices(cat_vec, weights=poss_vec)[0]

def choose_one_row(data_table: list, type: str = 'clothes') -> list:
    #print(f'choose_one_row() : data_table - {data_table}, type - {type}')
    # Фильтруем строки по заданному типу
    if(type != 'all types'):
        filtered_data = [row for row in data_table if row[0].strip() == type]
    else:
        filtered_data = data_table
    #print(f'filtered - {filtered_data}')
    if not filtered_data:
        return None  # Если нет данных для заданного типа, возвращаем None
    # Выбираем случайный элемент из отфильтрованных данных
    random_item = random.choice(filtered_data)
    return random_item

def choose_item_from_row(row: list) -> list:
    #(f'Chosen row = {row}')
    if len(row) < 4:
        return None
    general_category = row[0].strip()
    brands = row[1].split(', ')
    items = row[2].split(', ')
    prices = list(map(float, row[3].split(', ')))
    
    if len(items) != len(prices):
        return None  # Если количество элементов не совпадает, возвращаем None

    random_brand = random.randint(0, len(brands) - 1)
    random_product = random.randint(0, len(items) - 1)
    price = prices[random_product]
    lower_bound = price * 0.75 * 65
    upper_bound = price * 1.25 * 65
    # Генерируем случайное значение в пределах диапазона
    random_price = int(round(random.uniform(lower_bound, upper_bound)))
    # Возвращаем информацию о случайном товаре
    return [general_category,brands[random_brand], items[random_product],random_price]


def generate_one_output(dataset: list,poss_bank_vec,poss_painment_vec,data_type='all types'):
    out = []
    # Магнит Онлайн,59.86261171 ; 30.3339711,2024-02-24T11:36,Орехи грецкие,Родник Приэльбрусья,2202 2066 7798 4241,6,2856
    random_row = choose_one_row(data_table=dataset, type=data_type)
    general_category, brand, item, price = choose_item_from_row(random_row)
    shop_data = read_data('data/shop_locations.csv',delimiter=';')
    general_category, shop_name,coordinates = choose_one_row(data_table=shop_data, type=general_category)
    min_time = '0'+str(random.randint(7,9))+':00' # time
    max_time =  str(random.randint(19,22))+':00' # time
    bank,painment_sys = possibility_generator(poss_bank_vec, bank_names),possibility_generator(poss_painment_vec, painment_system_names)
    amount = random.randint(1,7) # amount of items
 

    out.append(shop_name.strip()) # shop name
    out.append(coordinates.strip()) # shop coordinates
    out.append(generate_random_datetime(min_time=min_time,max_time=max_time)) # time
    out.append(item.strip()) 
    out.append(brand.strip())
    out.append(generate_one_card_2(bank=bank,pay_system=painment_sys)) # card generation
    out.append(amount)
    out.append((price) * amount)
    return out

def write_into_csv_file(sample):
    # Названия столбцов
    headers = ['Магазин', 'Координаты', 'Дата и время', 'Товар', 'Производитель', 'Номер карты', 'Количество', 'Цена']
    
    # Проверяем, существует ли файл result.csv
    file_exists = os.path.isfile('output/result.csv')
    
    # Открываем CSV файл для записи или создания
    with open('output/result.csv', mode='a', newline='', encoding='utf-8-sig') as out_csv:
        # Создаем объект writer
        writer = csv.writer(out_csv, delimiter=';', quoting=csv.QUOTE_MINIMAL)
        
        # Если файл не существует, записываем заголовки
        if not file_exists:
            writer.writerow(headers)
        
        # Записываем переданный массив данных
        writer.writerow(sample)

def generate_dataset(amount=10000, category_type='all types', possibility_banks=[0.5,0.1,0.1,0.1,0.1,0.1], possibility_painment_sys=[0.5,0.1,0.1,0.1,0.1,0.1]):
    dataset = read_data('data/general_table.csv', delimiter=';')
    for i in range(amount):
        if(i % (amount / 100) == 0):
            print(f'working, {round(((i/amount)*100),1)}%')
        out = generate_one_output(dataset=dataset,poss_bank_vec=possibility_banks,poss_painment_vec=possibility_painment_sys,data_type=category_type)
        write_into_csv_file(out)


def clicked(possibility_banks,possibility_painment_sys,category_type):
    try:
        amount = int(txt.get())
    except:
        amount = 1000
    if ((sum(possibility_banks)) != 1.0) or (sum(possibility_painment_sys) != 1.0):
        messagebox.showinfo('Ошибка', 'Сумма в процентах не равна 100')    
    elif(amount <= 0):
        messagebox.showinfo('Ошибка', 'Указано неправильное количество строк')    
    else:
        generate_dataset(amount=amount, category_type=category_type, possibility_banks=possibility_banks, possibility_painment_sys=possibility_painment_sys)
        messagebox.showinfo('Готово', 'Создание таблицы завершено')


if __name__ == '__main__':
    
    window = Tk()
    window.title("Конструктор синтетических данных")
    window.geometry('900x600')

    # Labels
    lbl = Label(window, text="Добро пожаловать!", font='Arial 16 bold')  
    lbl.place(relx=0.5, rely=0.1, anchor=CENTER)
    

    lbl2 = Label(window, text="Выберите необходимые параметры для генерации:", font='Arial 12')  
    lbl2.place(relx=0.5, rely=0.17, anchor=CENTER)

    # Payment system processing:
    lbl3 = Label(window, text="Платежная система:", font='Arial 11')
    lbl3.place(relx=0.5, rely=0.25, anchor=CENTER)

    #['MIR','VISA','MASTERCARD','MAESTRO','AMERICAN EXPRESS']
    t1 = Label(window, text="MIR, %")
    t1.place(relx=0.245, rely=0.3)

    t2 = Label(window, text="VISA, %")
    t2.place(relx=0.436, rely=0.3)

    t3 = Label(window, text="MASTERCARD, %")
    t3.place(relx=0.627, rely=0.3)
    
    combo1 = Combobox(window, width=10)  
    combo1['values'] = (10, 20, 30, 40, 50, 60, 70, 80)  
    combo1.current(5) 
    combo1.place(relx=0.245, rely=0.37)

    combo2 = Combobox(window, width=10)  
    combo2['values'] = (10, 20, 30, 40, 50, 60, 70, 80)  
    combo2.current(1) 
    combo2.place(relx=0.436, rely=0.37)

    combo3 = Combobox(window, width=10)  
    combo3['values'] = (10, 20, 30, 40, 50, 60, 70, 80)  
    combo3.current(1) 
    combo3.place(relx=0.627, rely=0.37)
        
    # Bank processing:
    lbl4 = Label(window, text="Банк:", font='Arial 11')
    lbl4.place(relx=0.5, rely=0.5, anchor=CENTER)

    #['GAZPROMBANK','MTS BANK','RENAISSANCE CREDIT','SBERBANK OF RUSSIA','TINKOFF BANK','VTB BANK']
    t6 = Label(window, text="GAZPROMBANK, %")
    t6.place(relx=0.045, rely=0.57)

    t7 = Label(window, text="MTS BANK, %")
    t7.place(relx=0.236, rely=0.57)

    t8 = Label(window, text="VTB, %")
    t8.place(relx=0.427, rely=0.57)

    t9 = Label(window, text="SBERBANK, %")
    t9.place(relx=0.618, rely=0.57)
    
    t10 = Label(window, text="TINKOFF, %")
    t10.place(relx=0.809, rely=0.57)
    
    combo6 = Combobox(window, width=10)  
    combo6['values'] = (10, 20, 30, 40, 50, 60, 70)  
    combo6.current(5) 
    combo6.place(relx=0.045, rely=0.65)

    combo7 = Combobox(window, width=10)  
    combo7['values'] = (10, 20, 30, 40, 50, 60, 70)  
    combo7.current(0) 
    combo7.place(relx=0.236, rely=0.65)

    combo8 = Combobox(window, width=10)  
    combo8['values'] = (10, 20, 30, 40, 50, 60, 70)  
    combo8.current(0) 
    combo8.place(relx=0.427, rely=0.65)

    combo9 = Combobox(window, width=10)  
    combo9['values'] = (10, 20, 30, 40, 50, 60, 70)  
    combo9.current(0) 
    combo9.place(relx=0.618, rely=0.65)
    
    combo10 = Combobox(window, width=10)  
    combo10['values'] = (10, 20, 30, 40, 50, 60, 70)  
    combo10.current(0) 
    combo10.place(relx=0.809, rely=0.65)

    txt = Entry(window,width=20)  
    txt.place(relx=0.2, rely=0.85)

    lbl9 = Label(window, text="Введите количество строк", font='Arial 12')
    lbl9.place(relx=0.3, rely=0.8, anchor=CENTER)

    lbl11 = Label(window, text="Значение по умолчанию: 1000", font='Arial 8')
    lbl11.place(relx=0.3, rely=0.95, anchor=CENTER)


    lbl12 = Label(window, text="Выберите категорию товаров", font='Arial 12')
    lbl12.place(relx=0.6, rely=0.8, anchor=CENTER)
    
    combo11 = Combobox(window, width=20)  
    combo11['values'] = ('electronics','clothes','food','all types')  
    combo11.current(3) 
    combo11.place(relx=0.5, rely=0.85)
        
    def on_button_click():
        possibility_painment_sys = [float(combo1.get())/100,float(combo2.get())/100,float(combo3.get())/100]
        possibility_banks = [float(combo6.get())/100,float(combo7.get())/100,float(combo8.get())/100,float(combo9.get())/100,float(combo10.get())/100]
        category_chosen = combo11.get()
        print(category_chosen)

        clicked(possibility_banks=possibility_banks, possibility_painment_sys=possibility_painment_sys, category_type=category_chosen)

    btn = Button(window, text="Создать таблицу", command=on_button_click)
    btn.place(relx=0.95, rely=0.95, anchor="se", width=150) 

    window.mainloop()
    