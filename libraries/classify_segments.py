import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics.cluster import rand_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_auc_score, matthews_corrcoef, log_loss

from libraries.process_data import merge_chunks



def calculate_statistics(y_true, y_pred, y_proba=None):
    stats = {}

    # Accuracy
    stats['Accuracy'] = accuracy_score(y_true, y_pred)

    # Precision
    stats['Precision'] = precision_score(y_true, y_pred, average='weighted')

    # Recall
    stats['Recall'] = recall_score(y_true, y_pred, average='weighted')

    # F1-Score
    stats['F1-Score'] = f1_score(y_true, y_pred, average='weighted')

    # Confusion Matrix
    stats['Confusion Matrix'] = confusion_matrix(y_true, y_pred)

    # ROC-AUC (wymaga prawdopodobieństw dla każdej klasy)
    if y_proba is not None:
        stats['ROC-AUC'] = roc_auc_score(y_true, y_proba, average='weighted', multi_class='ovo')

    # MCC
    stats['MCC'] = matthews_corrcoef(y_true, y_pred)

    # Log-Loss (wymaga prawdopodobieństw dla każdej klasy)
    if y_proba is not None:
        stats['Log-Loss'] = log_loss(y_true, y_proba)

    return stats


#################################################################################

                            ##Assign clusters##

#################################################################################

# Zwraca tablicę z przyporządkowanymi klasami dla każdego clustra.
def assign_clusters_to_classes_count_single_points(cluster_membership, centroids, y):
    n_classes = len(np.unique(y))
    n_clusters = centroids.shape[0]    

    count_points = np.zeros((n_clusters, n_classes))
    
    for i, label in enumerate(cluster_membership):
        count_points[label, y[i]] += 1

    # Zwracamy tablicę z przyporządkowanymi klasami dla każdego clustra.
    return np.argmax(count_points, axis=1)


# Zwraca tablicę z przyporządkowanymi klasami dla każdego clustra.
def assign_clusters_to_classes_count_summary_labels(fuzzy_labels, centroids, y, power=1):
    n_classes = len(np.unique(y))
    n_clusters = centroids.shape[0]

    count_points = np.zeros((n_clusters, n_classes))
    
    for i, fuzzy_row in enumerate(fuzzy_labels.T):
        for j in range(n_clusters):
            count_points[j, y[i]] += fuzzy_row[j] ** power
    
    # Zwracamy tablicę z przyporządkowanymi klasami dla każdego clustra.
    return np.argmax(count_points, axis=1)


#################################################################################

                            ##Exam segments##

#################################################################################


# Dla każdego segmentu podaje najczęściej występujący cluster
# Klasyfikujemy punkty przy pomocy modelu następnie
def get_segments_labels_count_single_points(chunks, centroids, cluster_membership):

    segment_clusters = []
    start_chunk = 0
    for i, chunk in enumerate(chunks):
        # Liczba klastrów
        num_clusters = len(centroids)
        chunk_size = chunk.shape[0]
        
        # Zmienna do zliczania punktów w segmentach przypisanych do każdego klastra
        cluster_counts = np.zeros(num_clusters)

        for x in cluster_membership[start_chunk:start_chunk+chunk_size]:
            cluster_counts[x] += 1

        start_chunk += chunk_size
        segment_clusters.append(np.argmax(cluster_counts))
        
    return segment_clusters


def get_segments_labels_count_summary_labels(chunks, centroids, fuzzy_labels):

    fuzzy_labels = fuzzy_labels.T
    segment_clusters = []
    start_chunk = 0
    n_clusters = len(centroids)
    
    for i, chunk in enumerate(chunks):
        # Liczba klastrów
        chunk_size = chunk.shape[0]
        
        # Zmienna do zliczania punktów w segmentach przypisanych do każdego klastra
        cluster_counts = np.zeros(n_clusters)

        for fuzzy_row in fuzzy_labels[start_chunk:start_chunk+chunk_size]:
            for j in range(n_clusters):
                cluster_counts[j] += fuzzy_row[j]

        start_chunk += chunk_size
        segment_clusters.append(np.argmax(cluster_counts))
        
    return segment_clusters

# Dla segmentu zwracamy liczbę punktów które należą do danego clustra
def get_segments_clusters_labels_count_single_points(chunks, centroids, cluster_membership):

    segment_clusters = []
    start_chunk = 0
    for i, chunk in enumerate(chunks):
        # Liczba klastrów
        num_clusters = len(centroids)
        chunk_size = chunk.shape[0]
        
        # Zmienna do zliczania punktów w segmentach przypisanych do każdego klastra
        cluster_counts = np.zeros(num_clusters)

        for x in cluster_membership[start_chunk:start_chunk+chunk_size]:
            cluster_counts[x] += 1

        start_chunk += chunk_size
        segment_clusters.append(cluster_counts)
        
    return segment_clusters

# Dla segmentu zwracamy sumę współczyników punktów które należą do danego clustra
def get_segments_clusters_labels_count_summary_labels(chunks, centroids, fuzzy_labels):

    fuzzy_labels = fuzzy_labels.T
    segment_clusters = []
    start_chunk = 0
    n_clusters = len(centroids)
    
    for i, chunk in enumerate(chunks):
        # Liczba klastrów
        chunk_size = chunk.shape[0]
        
        # Zmienna do zliczania punktów w segmentach przypisanych do każdego klastra
        cluster_counts = np.zeros(n_clusters)

        for fuzzy_row in fuzzy_labels[start_chunk:start_chunk+chunk_size]:
            for j in range(n_clusters):
                cluster_counts[j] += fuzzy_row[j]

        start_chunk += chunk_size
        segment_clusters.append(cluster_counts)
        
    return segment_clusters



#################################################################################

                            ##Exam segments##

#################################################################################


# Łączy powyższe 2 funkcje. Zwraca klasy dla segmentów
def validate_segments(chunks, chunks_y, centroids, fuzzy_labels):

    # Mergujemy chunki w dataset
    data, y = merge_chunks(chunks, chunks_y)
    
    # Przydzielenie punktów do danych klustrów. Przydzielamy cluster najczesciej wystepujacy u sasiadow
    cluster_membership = np.argmax(fuzzy_labels, axis=0)
    #cluster_membership = knn_classify_based_on_labels(X_train, y_train, data, cluster_membership, n_neighbors=5)
   
    # Znajudjemy do jakiego clustra przypisany jest dany segment.
    segment_clusters = get_segments_labels_count_single_points(chunks, centroids, cluster_membership)
    #segment_clusters = get_segments_labels_count_summary_labels(chunks, centroids, fuzzy_labels)

    # Klasy segmentów
    labels = [chunk_y[0] for chunk_y in chunks_y]

    # Przyporządkujemy clustry do klas na podstawie danych treningowych.
    cluster_to_class = assign_clusters_to_classes_count_single_points(cluster_membership, centroids, y)
    #cluster_to_class = assign_clusters_to_classes_count_summary_labels(fuzzy_labels, centroids, y)

    segment_labels = [cluster_to_class[cluster] for cluster in segment_clusters]
    
    return calculate_statistics(labels, segment_labels), cluster_to_class

# Łączy powyższe 2 funkcje. Zwraca klasy dla segmentów
def validate_segments_knn(chunks, chunks_y, cluster_membership):

    # Mergujemy chunki w dataset
    data, y = merge_chunks(chunks, chunks_y)
    
    # Znajudjemy do jakiego clustra przypisany jest dany segment.
    segment_clusters = get_segments_labels_count_single_points(chunks, centroids, cluster_membership)

    # Klasy segmentów
    labels = [chunk_y[0] for chunk_y in chunks_y]

    # Przyporządkujemy clustry do klas na podstawie danych treningowych.
    cluster_to_class = assign_clusters_to_classes_count_single_points(cluster_membership, centroids, y)

    segment_labels = [cluster_to_class[cluster] for cluster in segment_clusters]
    print('przynalznosc clustra do klasy', cluster_to_class)

    
    return calculate_statistics(labels, segment_labels)


def find_best_fitting_class(cluster_to_class, clusters_count, n_classes):
    
    # Przygotuj tablicę wynikową
    result = np.zeros(n_classes)
    
    # Sumowanie punktów w klastrach na podstawie klas
    for i in range(len(clusters_count)):
        class_idx = cluster_to_class[i]  # Przynależność klastra do klasy
        result[class_idx] += clusters_count[i]  # Sumowanie punktów dla danej klasy
        
    return np.argmax(result)

# Łączy powyższe 2 funkcje. Zwraca klasy dla segmentów
def validate_segments_(chunks, chunks_y, centroids, fuzzy_labels):
    
    #segment_clusters = get_segments_labels_count_single_points(chunks, centroids, fuzzy_labels)
    #segment_clusters = get_segments_labels_count_summary_labels(chunks, centroids, fuzzy_labels)
    segment_clusters = get_segments_clusters_labels_count_summary_labels(chunks, centroids, fuzzy_labels)

    y = np.concatenate(chunks_y)
    n_classes = len(np.unique(y))
    
    labels = [chunk_y[0] for chunk_y in chunks_y]

    #cluster_to_class = assign_clusters_to_classes_count_single_points(fuzzy_labels, centroids, y)
    cluster_to_class = assign_clusters_to_classes_count_summary_labels(fuzzy_labels, centroids, y)
    
    segment_labels = [find_best_fitting_class(cluster_to_class, clusters_count, n_classes) for clusters_count in segment_clusters]
    #print('segment clusters', [int(x) for x in segment_labels])
    print(cluster_to_class)
    print([int(x) for x in segment_labels])
    #print('y',[int(x) for x in labels])
    #print('segment_labels', [int(x) for x in segment_labels])
    
    return calculate_statistics(labels, segment_labels)
