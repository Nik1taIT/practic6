import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

# 1. Завантаження даних
data_url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mall_customers.csv"
data = pd.read_csv(data_url)

# Попередній аналіз даних
print("Інформація про дані:")
print(data.info())
print("Описові статистики:")
print(data.describe())

# Візуалізація розподілів
sns.pairplot(data, hue='Gender')
plt.show()

# Підготовка даних: кодування та нормалізація
data['Gender'] = LabelEncoder().fit_transform(data['Gender'])
scaled_data = StandardScaler().fit_transform(data.drop(columns=['CustomerID']))

# 2. PCA (Principal Component Analysis)
pca = PCA(n_components=3)
pca_result = pca.fit_transform(scaled_data)
explained_variance = pca.explained_variance_ratio_

# Візуалізація дисперсії
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(explained_variance) + 1), explained_variance.cumsum(), marker='o')
plt.title("Кумулятивна частка дисперсії")
plt.xlabel("Кількість компонент")
plt.ylabel("Кумулятивна дисперсія")
plt.show()

# PCA: Візуалізація у 2D
plt.scatter(pca_result[:, 0], pca_result[:, 1])
plt.title("PCA: 2D Візуалізація")
plt.xlabel("Головна компонента 1")
plt.ylabel("Головна компонента 2")
plt.show()

# PCA: Візуалізація у 3D
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pca_result[:, 0], pca_result[:, 1], pca_result[:, 2])
ax.set_title("PCA: 3D Візуалізація")
plt.show()

# 3. t-SNE (t-Distributed Stochastic Neighbor Embedding)
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
tsne_result = tsne.fit_transform(scaled_data)

# t-SNE: Візуалізація у 2D
plt.scatter(tsne_result[:, 0], tsne_result[:, 1])
plt.title("t-SNE (perplexity=30, learning_rate=200)")
plt.xlabel("Компонента 1")
plt.ylabel("Компонента 2")
plt.show()

# Експеримент із параметрами t-SNE
alt_tsne = TSNE(n_components=2, perplexity=50, learning_rate=100, random_state=42)
alt_tsne_result = alt_tsne.fit_transform(scaled_data)

# Альтернативна t-SNE: Візуалізація у 2D
plt.scatter(alt_tsne_result[:, 0], alt_tsne_result[:, 1])
plt.title("t-SNE (perplexity=50, learning_rate=100)")
plt.xlabel("Компонента 1")
plt.ylabel("Компонента 2")
plt.show()

# 4. Кластеризація: PCA та t-SNE
# K-Means для PCA
kmeans_pca = KMeans(n_clusters=3, random_state=42).fit(pca_result)
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=kmeans_pca.labels_, cmap='viridis')
plt.title("Кластери на PCA-зменшених даних")
plt.xlabel("Головна компонента 1")
plt.ylabel("Головна компонента 2")
plt.show()

# K-Means для t-SNE
kmeans_tsne = KMeans(n_clusters=3, random_state=42).fit(tsne_result)
plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=kmeans_tsne.labels_, cmap='viridis')
plt.title("Кластери на t-SNE-зменшених даних")
plt.xlabel("Компонента 1")
plt.ylabel("Компонента 2")
plt.show()

# 5. Порівняння результатів
print("Результати PCA:")
print(f"Дисперсія першої компоненти: {explained_variance[0]:.2f}")
print("Результати t-SNE видно візуально.")
# Додавання кластерів до початкових даних
data['Cluster_PCA'] = kmeans_pca.labels_
data['Cluster_tSNE'] = kmeans_tsne.labels_

# Опис кожного кластера для PCA
print("Опис кластерів для PCA-зменшених даних:")
pca_clusters_description = data.groupby('Cluster_PCA').mean()
print(pca_clusters_description)

# Опис кожного кластера для t-SNE
print("Опис кластерів для t-SNE-зменшених даних:")
tsne_clusters_description = data.groupby('Cluster_tSNE').mean()
print(tsne_clusters_description)

# Генерація маркетингових стратегій на основі кластерів
for cluster_id, cluster_data in pca_clusters_description.iterrows():
    print(f"\nКластер {cluster_id}:")
    print("Характеристики:")
    print(cluster_data)
    print("Маркетингова стратегія:")
    if cluster_data['Annual Income (k$)'] > 80:
        print(" - Пропонуйте преміум-продукти або послуги.")
        print(" - Запустіть ексклюзивну програму лояльності.")
    elif cluster_data['Age'] < 30:
        print(" - Використовуйте соціальні мережі для реклами.")
        print(" - Зосередьтесь на модних продуктах та знижках для молоді.")
    elif cluster_data['Spending Score (1-100)'] > 60:
        print(" - Впровадьте програми заохочення для активних покупців.")
        print(" - Пропонуйте персоналізовані рекомендації.")
    else:
        print(" - Запустіть акції чи розстрочки, щоб підвищити активність покупців.")
