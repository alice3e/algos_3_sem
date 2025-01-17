import requests
import csv
import os

# Замените 'YOUR_API_KEY' на ваш реальный API ключ от Yandex
API_KEY = 'YOUR_API_KEY'

def get_places(location, radius, keyword):
    # Формируем URL для запроса к Places API Yandex
    url = "https://search-maps.yandex.ru/v1/"
    params = {
        'apikey': API_KEY,
        'text': keyword,
        'lang': 'ru_RU',
        'll': f'{location[1]},{location[0]}',  # Долгота, Широта
        'spn': f'{radius / 1000},{radius / 1000}',  # Радиус в километрах
        'type': 'biz',  # Только бизнесы (магазины)
        'results': 50  # Максимальное количество результатов
    }

    response = requests.get(url, params=params)
    data = response.json()

    places = []
    for place in data.get('features', []):
        place_info = place['properties']['CompanyMetaData']
        coordinates = place['geometry']['coordinates']
        places.append({
            'name': place_info['name'],
            'lat': coordinates[1],
            'lng': coordinates[0]
        })

    return places

def save_to_csv(data, filename):
    # Проверка существования директории
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Запись данных в CSV файл
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter=';')
        for row in data:
            writer.writerow(row)

def read_shop_names(filename):
    shops = []
    with open(filename, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter=';')
        for row in reader:
            category, shop_name = row
            shops.append((category, shop_name))
    return shops

def main():
    # Координаты центра поиска (например, центр города)
    location = (59.927153, 30.308394)
    radius = 5000  # Радиус поиска в метрах

    # Путь к файлу с названиями магазинов и категориями
    shop_names_file = 'data/shop_names.csv'
    
    # Путь для сохранения файла с геолокациями
    csv_file_path = 'data/shop_locations.csv'
    
    # Считываем названия магазинов и категории
    shop_data = read_shop_names(shop_names_file)
    
    all_places = []
    
    # Для каждого магазина ищем места и сохраняем
    for category, shop_name in shop_data:
        places = get_places(location, radius, shop_name)
        for place in places:
            # Добавляем строку: категория ; название магазина ; координаты
            all_places.append([category, place['name'], f"{place['lat']}, {place['lng']}"])
    
    # Сохраняем данные в CSV файл
    save_to_csv(all_places, csv_file_path)
    print(f"Данные сохранены в {csv_file_path}")

if __name__ == "__main__":
    main()
