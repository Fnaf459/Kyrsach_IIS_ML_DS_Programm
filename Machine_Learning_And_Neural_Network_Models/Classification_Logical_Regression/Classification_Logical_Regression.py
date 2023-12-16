from flask import render_template, request, Blueprint
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import io
import base64

logical_regression_bp = Blueprint('Classification_Logical_Regression', __name__)

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

# Создайте и обучите модель логистической регрессии
model = LogisticRegression()
model.fit(X_train, y_train)

# Оцените модель на тестовом наборе данных
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Получите матрицу ошибок
conf_matrix = confusion_matrix(y_test, y_pred)

# Получите значения ROC-кривой
fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
roc_auc = auc(fpr, tpr)

# Создание графика ROC-кривой
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")

# Сохранение изображения в байтовом массиве
img_buf = io.BytesIO()
plt.savefig(img_buf, format='png')
img_buf.seek(0)
img_base64 = base64.b64encode(img_buf.read()).decode('utf-8')
plt.close()

# Передача закодированного изображения в шаблон HTML
img_tag = f'<img src="data:image/png;base64,{img_base64}" alt="ROC Curve">'

# Передача данных в шаблон
@logical_regression_bp.route("/", methods=["GET", "POST"])
def index():
    img_tag = None
    conf_matrix_html = None
    feature_coefficients = None

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

        # Заполнение пропущенных значений нулями
        input_data.fillna(0, inplace=True)

        prediction = model.predict(input_data)[0]
        result = "Успешная" if prediction == 1 else "Неуспешная"

        img_tag = f'<img src="data:image/png;base64,{img_base64}" alt="ROC Curve">'
        feature_coefficients = model.coef_[0]

        # Преобразование матрицы ошибок в HTML-строку
        conf_matrix_html = f'<p>Матрица ошибок:</p><pre>{conf_matrix.tolist()}</pre>'

        return render_template("Classification_Logical_Regression.html", accuracy=accuracy, success_count=sum(y_test),
                               failure_count=len(y_test) - sum(y_test), conf_matrix_html=conf_matrix_html,
                               img_tag=img_tag, prediction_result=result, feature_coefficients=feature_coefficients)

    return render_template("Classification_Logical_Regression.html", accuracy=accuracy, success_count=sum(y_test),
                           failure_count=len(y_test) - sum(y_test), conf_matrix_html=conf_matrix_html,
                           img_tag=img_tag, feature_coefficients=feature_coefficients)
