import pandas as pd
import numpy as np
from random import sample
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

def custom_accuracy(cluster_labels, class_labels):
    unique_labels = np.unique(class_labels)
    num_labels = len(unique_labels)
    num_clusters = len(np.unique(cluster_labels))
    
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    
    for i in range(num_clusters):
        cluster_indices = (cluster_labels == i)
        cluster_class_labels = class_labels[cluster_indices]
        
        if len(cluster_class_labels) > 0:
            majority_class = np.argmax(np.bincount(cluster_class_labels))
            cluster_accuracy = accuracy_score(cluster_class_labels, majority_class * np.ones_like(cluster_class_labels))
            cluster_precision = precision_score(cluster_class_labels, majority_class * np.ones_like(cluster_class_labels), average='binary')
            cluster_recall = recall_score(cluster_class_labels, majority_class * np.ones_like(cluster_class_labels), average='binary')
            
            accuracy_scores.append(cluster_accuracy)
            precision_scores.append(cluster_precision)
            recall_scores.append(cluster_recall)
    
    avg_accuracy = np.mean(accuracy_scores) * 100
    avg_precision = np.mean(precision_scores) * 100
    avg_recall = np.mean(recall_scores) * 100
    
    return avg_accuracy, avg_precision, avg_recall

def kmeans_scikit(df, k):
    kmeans = KMeans(n_clusters=k, random_state=0, max_iter=MAX_ITER)
    cluster_labels = kmeans.fit_predict(df)
    cluster_centers = kmeans.cluster_centers_
    return cluster_labels, cluster_centers

def preprocess_data(df):
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return df_scaled

def main():
    k = 2
    MAX_ITER = 100 
    df_full = pd.read_csv('SPECTF_New.csv')
    columns = list(df_full.columns)
    features = columns[:len(columns)-1]
    class_labels = list(df_full[columns[-1]])
    df = df_full[features]
    
    df_scaled = preprocess_data(df)
    labels, centers = kmeans_scikit(df_scaled, k)
    a, p, r = custom_accuracy(labels, class_labels)

    print("Custom Accuracy = {:.2f}%".format(a))
    print("Custom Precision = {:.2f}%".format(p))
    print("Custom Recall = {:.2f}%".format(r))

if __name__ == "__main__":
    main()
