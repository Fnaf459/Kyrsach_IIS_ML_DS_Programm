from flask import render_template, Blueprint
import pandas as pd
import io
import base64
import numpy as np
from minisom import MiniSom
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score

som_bp = Blueprint('Clustering_SOM_Neural_Network', __name__)

# Загрузка данных из файла games.csv
data = pd.read_csv('games.csv')

# Реализация кластеризации SOM
def generate_som_image():
    # Выбираем интересующие нас переменные для кластеризации
    features = data[['genres', 'price', 'platform', 'developer']]

    # Преобразовываем категориальные переменные в числовые (например, с помощью One-Hot Encoding)
    features = pd.get_dummies(features, columns=['genres', 'price', 'platform', 'developer'])

    # Нормализуем данные с использованием MinMaxScaler
    scaler = MinMaxScaler()
    features_normalized = scaler.fit_transform(features)

    # Определяем размер карты Кохонена
    som_size = (10, 10)  # Подберите подходящий размер

    # Создаем SOM
    som = MiniSom(som_size[0], som_size[1], features.shape[1], sigma=0.3, learning_rate=0.5)

    # Обучаем SOM
    som.train_random(features_normalized, 1000)

    # Получаем метки кластеров для каждого объекта
    cluster_labels = np.array([som.winner(x) for x in features_normalized])

    # Преобразуем координаты в одномерный массив
    cluster_labels_flat = cluster_labels[:, 0] * som_size[1] + cluster_labels[:, 1]

    # Создаем карту кластеров
    cluster_map = np.zeros((som_size[0], som_size[1]))

    for i in range(features.shape[0]):
        cluster_map[cluster_labels[i][0], cluster_labels[i][1]] += 1

    # Визуализируем карту кластеров
    plt.figure(figsize=(10, 8))
    plt.imshow(cluster_map, cmap='viridis', interpolation='nearest')
    plt.colorbar()

    # Оцениваем качество кластеризации с использованием метрики силуэта
    silhouette_avg = silhouette_score(features_normalized, cluster_labels_flat)

    # Сохраняем изображение в буфер
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)

    # Преобразуем изображение в base64 строку
    img_base64 = base64.b64encode(img_buffer.read()).decode()

    # Определяем количество уникальных кластеров и их размерность
    unique_clusters, cluster_sizes = np.unique(cluster_labels_flat, return_counts=True)
    non_empty_clusters = unique_clusters[cluster_sizes > 0]

    # Размеры кластеров
    cluster_sizes = cluster_sizes[cluster_sizes > 0]

    # Характеристики кластеров (средние значения)
    cluster_features_mean = [features[cluster_labels_flat == i].mean() for i in non_empty_clusters]

    # Формируем текстовый отчет
    report = f"Количество кластеров: {len(non_empty_clusters)}\n"
    report += f"Размеры кластеров: {cluster_sizes}\n"
    report += f"\nОценка качества кластеризации (силуэт): {silhouette_avg:.4f}"

    for i, cluster_mean in enumerate(cluster_features_mean, start=1):
        report += f"\nХарактеристики кластера {i} (средние значения):\n"
        report += f"{cluster_mean}\n"

    return img_base64, report

@som_bp.route('/')
def index():
    som_image, report = generate_som_image()
    return render_template('Clustering_SOM_Neural_Network.html', som_image=som_image, report=report)
