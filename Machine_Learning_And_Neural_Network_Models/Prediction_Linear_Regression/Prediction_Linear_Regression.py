# Импорт необходимых библиотек и создание Blueprint для линейной регрессии
from flask import request, render_template, Blueprint
import pandas as pd
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from datetime import datetime
import matplotlib.pyplot as plt

linear_regression_bp = Blueprint('Prediction_Linear_Regression', __name__)

# Загрузка данных и объединение двух наборов данных по общему идентификатору
games_data = pd.read_csv("games.csv")
open_critic_data = pd.read_csv("open_critic.csv")
data = games_data.merge(open_critic_data, left_on="id", right_on="game_id", how="inner")

# Преобразование даты релиза в числовой формат (в днях с минимальной даты)
data['release_date'] = (pd.to_datetime(data['release_date']) - pd.to_datetime(data['release_date']).min()).dt.days

# Удаление записей с отсутствующим рейтингом
data = data.dropna(subset=["rating"])

# Преобразование категориального признака "жанры" в числовой формат
label_encoder = LabelEncoder()
data["genres"] = label_encoder.fit_transform(data["genres"])

# Выделение признаков и целевой переменной
X = data[["genres", "price", "release_date"]]
y = data["rating"]

# Полиномиальное преобразование признаков
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X[['genres', 'price', 'release_date']])

# Стандартизация признаков
scaler_poly = StandardScaler()
X_poly[:, 3:] = scaler_poly.fit_transform(X_poly[:, 3:])

# Разделение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Создание и обучение модели линейной регрессии
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Определение минимальной даты релиза для дальнейшего использования
min_release_date = min(pd.to_datetime(games_data['release_date'])).replace(tzinfo=None)

# Определение маршрутов Flask для отображения формы и предсказания рейтинга
@linear_regression_bp.route('/', methods=['GET'])
def index():
    return render_template('Prediction_Linear_Regression.html')

@linear_regression_bp.route('/predict', methods=['POST'])
def predict():
    # Получение данных из формы ввода
    data = request.form
    genre = label_encoder.transform([data["genres"]])[0]
    price = float(data["price"])
    release_date = datetime.strptime(data["release_date"], "%Y-%m-%dT%H:%M:%S.%fZ")
    release_date = (release_date - min_release_date).days

    # Подготовка входных данных для предсказания
    input_data = poly.transform([[genre, price, release_date]])
    input_data[:, 3:] = scaler_poly.transform(input_data[:, 3:])

    # Предсказание рейтинга
    rating = regressor.predict(input_data)[0]

    # Оценка модели на тестовом наборе данных
    y_pred = regressor.predict(X_test)
    prediction_accuracy = mean_absolute_error(y_test, y_pred)
    model_accuracy = r2_score(y_test, y_pred)

    # Создание и сохранение графика рассеяния и гистограммы остатков
    plt.figure(figsize=(12, 6))
    plt.scatter(y_test, y_pred, color='blue', label='Фактический vs. Предсказанный')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red', label='Идеальное соответствие')
    plt.title('Фактический vs. Предсказанный рейтинг')
    plt.xlabel('Фактический рейтинг')
    plt.ylabel('Предсказанный рейтинг')
    plt.legend()
    plt.savefig('static/scatter_plot.png')
    plt.close()

    residuals = y_test - y_pred
    plt.figure(figsize=(12, 6))
    plt.hist(residuals, bins=30, edgecolor='black')
    plt.title('Распределение остатков')
    plt.xlabel('Остатки')
    plt.ylabel('Частота')
    plt.savefig('static/residuals_histogram.png')
    plt.close()

    # Отображение результатов в шаблоне HTML
    return render_template('Prediction_Linear_Regression.html', rating=rating,
                           prediction_accuracy=prediction_accuracy, model_accuracy=model_accuracy)
