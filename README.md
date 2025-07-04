# Приложение "Предсказание рисков сердечного приступа"
Приложение предсказывает риск сердечного приступа (1 - высокий, 0 - низкий) по входным признакам при помощи предобученной модели на базе `DecisionTreeClassifier`.

Метрики модели, полученные на валидационной выборке:
- `accuracy`: 0.409
- `f1`: 0.521
- `precision`: 0.362
- `recall`: 0.925
- `roc-auc`: 0.539

Пример входных признаков представлен в файле `templates/input_example.csv`.

# Архитектура приложения
Приложение состоит из двух файлов:
- `app.py` - веб-сервер.
- `predictor.py` - класс работы с моделью.

Класс `Predictor()` при инициализации загружает предобученную модель.

Методы класса:
- `predict` (публичный) - запускает процессы предобработки признаков и предсказания. Возвращает pd.DataFrame() с исходными индексами и результатами предсказаний.
- `get_feature_names_in` (публичный) - возвращает названия столбцов исходного датафрейма.
- `get_feature_names_out` (публичный) - возвращает названия столбцов датафрейма после предобработки.
- `__proc` (приватный) - предобрабатывает исходный датафрейм в формат, используемый моделью для предсказаний.
- `__drop_columns` (приватный) - в завершение предобработки датафрейма удаляет признаки, которые не будут использованы моделью.
- `__fill_missing` (приватный) - заполняет пустое значение признаковой строки на основании значений в другом признаке этой же строки.
- `__fill_missing_2d` (приватный)- заполняет пустое значение признаковой строки на основании значений в двух других признаках этой же строки.

# Запуск приложения
Веб-приложение запускается через файл `app.py`. Адрес по умолчанию `http://0.0.0.0:8000`.

На главной странице присутствуют два блока.
1. Предсказание по входным признакам. Первым делом необходимо файл с признаками. После чего необходимо нажать кнопку `Предсказать`. По завершению процесса предсказания будет предложен к сохранению файл с результатами. Формат результирующего файла `CSV`, разделитель - `запятая`, колонки - `id`, `prediction`.
2. Пример файла с входными признаками. При нажатии на кнопку `Получить` будет предложен к сохранению файл-пример для загрузки к предсказанию.

# Пример использования класса Predictor
Инициализация и заполнение:
```py
predictor = Predictor() # инициализируем модель
predictions = predictor.predict('input_features.csv') # предсказываем таргет и заполняем поля класса
predictions.to_csv('predict_result.csv') # сохраняем в файл
```
Использование доступных методов класса:
```py
print(predictor.get_feature_names_in()) # смотрим исходные названия столбцов
```
```py
print(predictor.get_feature_names_out()) # смотрим названия столбцов после предобработки
```