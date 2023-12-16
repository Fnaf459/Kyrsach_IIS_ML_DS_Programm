from flask import render_template, request, Blueprint
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import io
import base64

decision_tree_bp = Blueprint('Decision_Tree_Classification', __name__)

# Загрузите данные из файлов
games_data = pd.read_csv("games.csv")
open_critic_data = pd.read_csv("open_critic.csv")

# Объедините данные
merged_data = games_data.merge(open_critic_data, left_on="id", right_on="game_id", how="inner")

# Создайте целевую переменную (успешность игры)
merged_data["Success"] = merged_data["rating"].apply(lambda x: 1 if x > 70 else 0)

# Преобразуйте столбец "release_date" в формат Unix-времени с корректной обработкой временных зон
merged_data["release_date"] = pd.to_datetime(merged_data["release_date"], utc=True).astype('int64') // 10**9

# Преобразуйте жанры в бинарные признаки (one-hot encoding)
genres_encoded = merged_data['genres'].str.get_dummies(sep=',')
merged_data = pd.concat([merged_data, genres_encoded], axis=1)
merged_data.drop('genres', axis=1, inplace=True)

# Определите признаки и целевую переменную
features = ["price", "release_date"] + list(genres_encoded.columns)
X = merged_data[features]
y = merged_data["Success"]

# Разделите данные на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=42)

# Создайте и обучите модель дерева решений
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Оцените модель на тестовом наборе данных
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Получите важности признаков
feature_importances = model.feature_importances_

# Определение индексов признаков, относящихся к категории "жанры", цена и дата релиза
genres_indices = [i for i, feature in enumerate(features) if feature in genres_encoded.columns]
price_index = features.index("price")
release_date_index = features.index("release_date")

# Получение общей важности для категории "жанры", цены и даты релиза
genres_importance = feature_importances[genres_indices].sum()
price_importance = feature_importances[price_index]
release_date_importance = feature_importances[release_date_index]

# Создание столбчатой диаграммы для общей важности признаков
plt.figure(figsize=(10, 3))
plt.barh(["Genres", "Price", "Release Date"], [genres_importance, price_importance, release_date_importance])
plt.xlabel("Importance")
plt.title("Feature Importance")

# Сохранение изображения в байтовом массиве
img_buf = io.BytesIO()
plt.savefig(img_buf, format='png')
img_buf.seek(0)
img_base64 = base64.b64encode(img_buf.read()).decode('utf-8')
plt.close()

# Передача закодированного изображения в шаблон HTML
img_tag = f'<img src="data:image/png;base64,{img_base64}" alt="Feature Importance">'

# Передача общей важности для категории "жанры", цены и даты релиза в шаблон HTML
genres_importance_html = f'<p>Общая важность для жанров: {genres_importance:.4f}</p>'
price_importance_html = f'<p>Важность цены: {price_importance:.4f}</p>'
release_date_importance_html = f'<p>Важность даты релиза: {release_date_importance:.4f}</p>'

# Передача данных в шаблон
@decision_tree_bp.route("/", methods=["GET", "POST"])
def index():
    img_tag = None
    genres_importance_html = None
    price_importance_html = None
    release_date_importance_html = None
    feature_importances = None

    if request.method == "POST":
        price = float(request.form["price"])
        release_date = pd.to_datetime(request.form["release_date"], utc=True)
        genres_input = request.form["genres"]
        genres_encoded_input = pd.DataFrame(genres_input.split(','), columns=["genre"])
        genres_encoded_input = genres_encoded_input['genre'].str.get_dummies()

        input_data = pd.DataFrame(columns=X_train.columns)
        input_data["price"] = [price]
        release_date_unix = release_date.timestamp()
        input_data["release_date"] = [release_date_unix]

        for genre in genres_encoded_input.columns:
            input_data[genre] = genres_encoded_input[genre].values

        prediction = model.predict(input_data)[0]
        result = "Успешная" if prediction == 1 else "Неуспешная"

        img_tag = f'<img src="data:image/png;base64,{img_base64}" alt="Feature Importance">'
        feature_importances = model.feature_importances_

        genres_importance_html = f'<p>Общая важность для жанров: {feature_importances[genres_indices].sum():.4f}</p>'
        price_importance_html = f'<p>Важность цены: {feature_importances[price_index]:.4f}</p>'
        release_date_importance_html = f'<p>Важность даты релиза: {feature_importances[release_date_index]:.4f}</p>'

        return render_template("Decision_Tree_Classification.html", accuracy=accuracy, success_count=sum(y_test),
                               failure_count=len(y_test) - sum(y_test), genres_importance_html=genres_importance_html,
                               price_importance_html=price_importance_html,
                               release_date_importance_html=release_date_importance_html, img_tag=img_tag,
                               prediction_result=result, feature_importances=feature_importances)

    if feature_importances is not None:
        genres_importance_html = f'<p>Общая важность для жанров: {feature_importances[genres_indices].sum():.4f}</p>'
        price_importance_html = f'<p>Важность цены: {feature_importances[price_index]:.4f}</p>'
        release_date_importance_html = f'<p>Важность даты релиза: {feature_importances[release_date_index]:.4f}</p>'

    return render_template("Decision_Tree_Classification.html", accuracy=accuracy, success_count=sum(y_test),
                           failure_count=len(y_test) - sum(y_test), genres_importance_html=genres_importance_html,
                           price_importance_html=price_importance_html,
                           release_date_importance_html=release_date_importance_html, img_tag=img_tag,
                           feature_importances=feature_importances)
