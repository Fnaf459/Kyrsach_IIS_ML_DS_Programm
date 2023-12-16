from flask import render_template, Blueprint
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
import seaborn as sns
import io
import base64
import matplotlib.pyplot as plt

dendrogram_bp = Blueprint('Clustering_Linkage_Dendrogram', __name__)

# Загрузка данных из файла games.csv
data = pd.read_csv('games.csv')

# Реализация кластеризации dendrogram
def generate_dendrogram_image():
    # Выбираем интересующие нас переменные для кластеризации
    features = data[['genres', 'price', 'platform', 'developer']]

    # Преобразовываем категориальные переменные в числовые (например, с помощью One-Hot Encoding)
    features = pd.get_dummies(features, columns=['genres', 'price', 'platform', 'developer'])

    # Вычисляем матрицу расстояний
    linkage_matrix = linkage(features, method='ward')
    dendrogram_data = dendrogram(linkage_matrix, no_plot=True)

    # Создаем изображение вручную с использованием Matplotlib
    fig, ax = plt.subplots(figsize=(24, 16))
    g = sns.heatmap(features.iloc[dendrogram_data['leaves'], :], cmap='viridis', ax=ax)
    plt.xticks(rotation=90)

    # Сохраняем изображение в буфер
    img_buffer = io.BytesIO()
    fig.savefig(img_buffer, format='png')
    img_buffer.seek(0)

    # Преобразуем изображение в base64 строку
    img_base64 = base64.b64encode(img_buffer.read()).decode()

    # Определяем количество кластеров
    num_clusters = len(set(dendrogram_data['color_list']))

    # Другие вычисления или данные, которые вы хотите добавить в отчет

    # Формируем текстовый отчет
    report = f"Количество кластеров: {num_clusters}\n"

    return img_base64, report

@dendrogram_bp.route('/')
def index():
    dendrogram_image, report = generate_dendrogram_image()
    return render_template('Clustering_Linkage_Dendrogram.html', dendrogram_image=dendrogram_image, report=report)
