# CustomerSegmentation
Customer Segmentation using Text Embedding, UMAP, and K-Means Clustering

This Python project demonstrates a technique for customer segmentation based on both text and numerical data. The code uses TensorFlow for text tokenization and embedding, UMAP for dimensionality reduction, and K-Means clustering for segmentation.

Overview
The project aims to segment customers based on various features including textual descriptions, quantities, and unit prices. It uses text embedding for textual data, UMAP for dimensionality reduction, and K-Means for clustering.

Code Structure
The code is organized as follows:

Data Import and Preprocessing

Importing necessary libraries.
Reading data from dataCopy.csv.
Dropping rows with missing 'Description' values.
Text Tokenization and Embedding

Using TensorFlow's Keras API for text tokenization.
Converting textual descriptions into fixed-size numerical vectors (text embeddings).
Feature Engineering

Combining the text embeddings with other numerical features like 'Quantity' and 'UnitPrice'.
Data Standardization

Standardizing features using sklearn.preprocessing.StandardScaler.
Dimensionality Reduction using UMAP

Reducing feature dimensions while preserving data structure.
K-Means Clustering

Using K-Means clustering to segment customers into groups.
Cluster Visualization

Using matplotlib to visualize the clusters in the UMAP feature space.
Interpretable Outputs

Printing the count of samples in each cluster.
Displaying the mean profile for each cluster based on original features.
Displaying the centroids of the clusters in the original feature space.
Functions and Libraries
pandas: For data manipulation and analysis.
numpy: For numerical operations.
sklearn.preprocessing.StandardScaler: For feature standardization.
sklearn.cluster.KMeans: For K-Means clustering.
tensorflow.keras.preprocessing.text.Tokenizer: For text tokenization.
umap.UMAP: For dimensionality reduction.
matplotlib.pyplot: For plotting.
Output
A UMAP scatter plot visualizing customer clusters.
Printed outputs including the count of samples per cluster and cluster profiles.



About Dataset
Context
Typically e-commerce datasets are proprietary and consequently hard to find among publicly available data. However, The UCI Machine Learning Repository has made this dataset containing actual transactions from 2010 and 2011. The dataset is maintained on their site, where it can be found by the title "Online Retail".

Content
"This is a transnational data set which contains all the transactions occurring between 01/12/2010 and 09/12/2011 for a UK-based and registered non-store online retail.The company mainly sells unique all-occasion gifts. Many customers of the company are wholesalers."

Per the UCI Machine Learning Repository, this data was made available by Dr Daqing Chen, Director: Public Analytics group. chend '@' lsbu.ac.uk, School of Engineering, London South Bank University, London SE1 0AA, UK.
The dataset can be downloaded in:
https://www.kaggle.com/datasets/carrie1/ecommerce-data

