import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import tensorflow as tf
import umap
import matplotlib.pyplot as plt

# Load the CSV data
data = pd.read_csv('data.csv', low_memory=False)  # Set low_memory to False to avoid DtypeWarning

# Drop rows with missing Description values
data = data.dropna(subset=['Description'])

# Extract relevant columns
X_text = data['Description']

# Tokenize and pad the text data
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(X_text)
X_text_encoded = tokenizer.texts_to_sequences(X_text)
X_text_padded = tf.keras.preprocessing.sequence.pad_sequences(X_text_encoded, padding='post')

# Create a word embeddings layer
embedding_dim = 10  # You can choose the embedding dimension
vocab_size = len(tokenizer.word_index) + 1
embedding_layer = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim)(X_text_padded)

# Average the embeddings for each sample
X_text_embeddings = tf.reduce_mean(embedding_layer, axis=1)

# Combine embeddings with other features
X_other_features = data[['Quantity', 'UnitPrice']]
X_combined = np.concatenate((X_text_embeddings, X_other_features), axis=1)

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_combined)

# Perform UMAP dimensionality reduction
reducer = umap.UMAP()
X_umap = reducer.fit_transform(X_scaled)

# Perform KMeans clustering on reduced data
num_clusters = 5  # You can choose the number of clusters
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
data['Cluster'] = kmeans.fit_predict(X_umap)

# Visualize the clusters
plt.figure(figsize=(10, 6))
plt.scatter(X_umap[:, 0], X_umap[:, 1], c=data['Cluster'], cmap='viridis', s=50)
plt.title('Customer Segmentation')
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.show()

# Show the count of samples in each cluster
print("\nCount of Samples in Each Cluster:")
print(data['Cluster'].value_counts())

# Create cluster profiles by computing mean values for each feature
print("\nCluster Profiles (Mean):")
cluster_profiles_mean = data.groupby('Cluster').mean()
print(cluster_profiles_mean)

# Optionally: Create cluster profiles by computing median values for each feature
# print("\nCluster Profiles (Median):")
# cluster_profiles_median = data.groupby('Cluster').median()
# print(cluster_profiles_median)

# Calculate the centroids in the original feature space based on cluster assignments
original_feature_centroids = []
for i in range(num_clusters):
    cluster_data = X_scaled[data['Cluster'] == i]
    cluster_center = np.mean(cluster_data, axis=0)
    original_feature_centroids.append(cluster_center)

original_feature_centroids = np.array(original_feature_centroids)

# Inverse transform to go back to the original feature scales
original_feature_centroids_unscaled = scaler.inverse_transform(original_feature_centroids)
print("\nCluster Centroids in the Original Feature Space:")
print(original_feature_centroids_unscaled)
