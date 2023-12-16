import pandas as pd
from flask import render_template, request, Blueprint
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from io import BytesIO
import base64

mlp_regressor_bp = Blueprint('Prediction_Neural_Network_MLPRegressor', __name__)

# Загрузка данных из файлов
games_df = pd.read_csv('games.csv')
open_critic_df = pd.read_csv('open_critic.csv')

# Объединение данных по столбцам id и game_id
merged_df = pd.merge(games_df, open_critic_df, left_on='id', right_on='game_id')

# Проверка и удаление строк с NaN значениями в X и y
merged_df = merged_df.dropna(subset=['genres', 'price', 'release_date', 'rating'])

# Выбор нужных столбцов для обучения
X = merged_df[['genres', 'price', 'release_date']]
y = merged_df['rating']

# Преобразование категориальных признаков в числовые
X = pd.get_dummies(X, columns=['genres'])

# Преобразование даты релиза в числовой формат (в этом примере используется год релиза)
X['release_year'] = pd.to_datetime(X['release_date']).dt.year
X = X.drop(['release_date'], axis=1)

# Разделение данных на обучающий и тестовый набор
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Нормализация данных
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Инициализация и обучение MLPRegressor
mlp_regressor = MLPRegressor(
    hidden_layer_sizes=(200, 100),
    activation='relu',
    max_iter=1000,
    solver='adam',
    alpha=0.01,
    random_state=42
)

mlp_regressor.fit(X_train_scaled, y_train)

# Получение точности на тестовых данных
accuracy = mlp_regressor.score(X_test_scaled, y_test) * 100


# Маршрут для отображения HTML-страницы
@mlp_regressor_bp.route('/')
def index():
    # Получение стандартного отклонения на тестовых данных
    std_deviation = y_test.std()

    # Создание графика
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, mlp_regressor.predict(X_test_scaled), alpha=0.5, label='Фактический vs. Предсказанный')

    # Добавление идеальной диагональной линии
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--',
             label='Идеальное соответствие')

    plt.title('Фактический рейтинг vs. Предсказанный рейтинг')
    plt.xlabel('Фактический рейтинг')
    plt.ylabel('Предсказанный рейтинг')
    plt.legend()
    plt.grid(True)

    # Сохранение графика в байтовый объект
    image_stream = BytesIO()
    plt.savefig(image_stream, format='png')
    plt.close()

    # Кодирование изображения в формат base64 для вставки в HTML
    image_stream.seek(0)
    encoded_image = base64.b64encode(image_stream.read()).decode()

    return render_template('Prediction_Neural_Network_MLPRegressor.html', accuracy=accuracy, std_deviation=std_deviation, plot_image=encoded_image)

# Маршрут для предсказания рейтинга на основе входных данных
@mlp_regressor_bp.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()

    # Преобразование входных данных в формат, подходящий для модели
    input_data = pd.DataFrame(data, index=[0])
    input_data['price'] = int(input_data['price'])
    input_data['release_year'] = pd.to_datetime(input_data['releaseDate']).dt.year
    input_data = pd.get_dummies(input_data, columns=['genres'])

    # Удаление releaseDate, так как она больше не нужна после преобразования
    input_data = input_data.drop(['releaseDate'], axis=1)

    # Добавление отсутствующих дамми-переменных
    missing_cols = set(X.columns) - set(input_data.columns)
    for col in missing_cols:
        input_data[col] = 0

    # Упорядочивание столбцов в нужном порядке
    input_data = input_data[X.columns]

    input_data_scaled = scaler.transform(input_data)

    # Предсказание рейтинга
    prediction = mlp_regressor.predict(input_data_scaled)

    # Оценка точности на новых данных
    mae_on_prediction = mean_absolute_error([y_test.iloc[0]], [prediction[0]])
    mse_on_prediction = mean_squared_error([y_test.iloc[0]], [prediction[0]])

    return render_template('Prediction_Neural_Network_MLPRegressor.html', prediction_result=f'Predicted Rating: {prediction[0]:.2f}',
                           mae_on_prediction=f'MAE on Prediction: {mae_on_prediction:.2f}',
                           mse_on_prediction=f'MSE on Prediction: {mse_on_prediction:.2f}')
