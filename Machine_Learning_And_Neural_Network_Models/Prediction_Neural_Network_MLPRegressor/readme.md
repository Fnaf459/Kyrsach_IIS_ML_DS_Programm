Общее задание:
Использовать нейронную сеть (четные варианты – MLPRegressor, нечетные – MLPClassifier) для данных из датасета выбранного для курсовой работы, самостоятельно сформулировав задачу. Интерпретировать результаты и оценить, насколько хорошо она подходит для решения сформулированной вами задачи.

Задание по вариантам:
Тема: Анализ данных игр Epic Games Store
Датасет: Epic Games Store Dataset
Ссылки:
https://www.kaggle.com/datasets/mexwell/epic-games-store-dataset?select=games.csv,
https://www.kaggle.com/datasets/mexwell/epic-games-store-dataset?select=open_critic.csv

Задача для нейронной сети:
Предсказание рейтинга игр на основе их (например: жанра (genres), цены (price), даты релиза (release_date), оценки критиков (rating) ).
Переменные: Жанр(ACTION,RPG; ACTION; INDIE,PUZZLE; SHOOTER,FPS; ACTION,FIGHTING,STEALTH), цена(1999, 1499, 999), даты релиза(2008-04-09T15:00:00.000Z, 2008-09-28T15:00:00.000Z, 2010-03-16T15:00:00.000Z), оценки критиков (90, 80, 100).

Запуск приложения осуществляется запуском файла app.py

Использованные технологии:
Среда программирования Pycharm
Версия языка python: 3.11

pandas: для работы с данными в виде DataFrame.

Flask: для создания веб-приложения.

render_template: для рендеринга HTML-шаблонов.

request: для работы с HTTP-запросами.

MLPRegressor: для создания многослойного персептрона (MLP) в задаче регрессии.

train_test_split: для разделения данных на обучающий и тестовый наборы.

StandardScaler: для стандартизации данных.

Краткое описание работы программы:
Загрузка данных: Программа загружает данные из CSV-файлов "games.csv" и "open_critic.csv"

Загрузка данных:
Программа загружает два набора данных из CSV-файлов (games.csv и open_critic.csv) и объединяет их по столбцам id и game_id.

Подготовка данных:
Удаляются строки с отсутствующими значениями в столбцах genres, price, release_date, и rating. Затем выбираются нужные столбцы для обучения (genres, price, release_date) и целевая переменная (rating).

Преобразование данных:
Категориальные признаки, такие как жанры игр (genres), преобразуются в числовые с использованием метода one-hot encoding. Дата релиза (release_date) преобразуется в числовой формат, в данном случае, используется только год релиза.

Разделение данных:
Данные разделяются на обучающий и тестовый наборы с использованием train_test_split.

Стандартизация данных:
Производится стандартизация данных с использованием StandardScaler.

Инициализация и обучение нейронной сети:
Инициализируется нейронная сеть с одним скрытым слоем (100 нейронов) и обучается на обучающем наборе данных.

Оценка точности:
Оценивается точность модели на тестовых данных и вычисляется стандартное отклонение целевой переменной.

Веб-интерфейс:
Flask-приложение предоставляет веб-интерфейс, где пользователь может видеть точность и стандартное отклонение модели. Есть также возможность предсказания рейтинга, введя данные через веб-форму.

Пример входных данных:
80% данных из файлов "games.csv" и "open_critic.csv", как обучающие и 20%, как тестовые для вывода точности и стандартного отклонения

Для предсказания: Жанры(писать только один основной): ACTION, Цена: 1999, Дата релиза: 2023-01-01

Пример выходных данных:
Predicted Rating: 59.96

Точность модели:
19.46889777598615% (on test data)

Стандартное отклонение рейтингов:
13.761271572601514% (on test data)