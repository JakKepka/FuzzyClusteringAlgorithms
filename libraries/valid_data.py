from collections import Counter
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics.cluster import rand_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_auc_score, matthews_corrcoef, log_loss
from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from skfuzzy.cluster import cmeans, cmeans_predict
from libraries.classify_segments import validate_segments, validate_labels, validate_labels_knn, validate_segments_knn, calculate_statistics
from libraries.classify_segments import classify_points_knn_eliminate_minor_class


#################################################################################

                            ##Statistics##

#################################################################################

def most_frequent_in_segments(array, segment_length=100):
    # Sprawdzenie czy tablica ma odpowiedni rozmiar
    if len(array) % segment_length != 0:
        raise ValueError(f"Array length must be a multiple of {segment_length}.")
    
    # Podział tablicy na segmenty
    segments = [array[i:i + segment_length] for i in range(0, len(array), segment_length)]
    
    # Przechowywanie wyników
    results = []

    for segment in segments:
        # Znajdź najczęściej występującą wartość i jej liczbę wystąpień
        counter = Counter(segment)
        most_common_value, count = counter.most_common(1)[0]
        results.append((most_common_value, count))
    
    return results
    
def classify_data_segment(data, cluster_membership, time_segment=100):
    
    results = most_frequent_in_segments(cluster_membership, time_segment)

    results_ = [int(result[0])  for result in results]

    # Zwraca klasę dla każdego odcinku czasowego wielkości time_segment klatek
    return results_
    
def merge_chunks(chunks, chunks_y):
    # Inicjalizacja pustych list na połączone dane
    data_set = []
    y = []

    # Iteracja przez wszystkie segmenty i etykiety
    for chunk, chunk_y in zip(chunks, chunks_y):
        # Rozszerzenie listy data_set o elementy z bieżącego segmentu
        data_set.extend(chunk)
        # Rozszerzenie listy y o elementy z bieżących etykiet
        y.extend(chunk_y)

    # Konwersja data_set i y na numpy.array (opcjonalne)
    data_set = np.array(data_set)
    y = np.array(y)

    return data_set, y



#################################################################################

                            ##Validate##

#################################################################################

def print_statistics(silhouette_avg, davies_bouldin_avg, rand, fpc, statistics):
    print(f'Silhouette Score: {silhouette_avg}')
    print(f'Davies-Bouldin Score: {davies_bouldin_avg}')
    print(f'Rand Score: {rand}')
    if(fpc != 0.0):
        print(f'Tested fpc: {fpc}')
    print('Accuracy:' , statistics['Accuracy'])
    print('Precision: ', statistics['Precision'])
    print('Recall: ', statistics['Recall'])

def valid_data_ifcm(chunks, centroids, chunks_y, clusters_for_each_class=None, m=2, error=0.05, metric='euclidean', print_data=False):
    from libraries.FCM.IFCM import predict_data_ifcm
    # Scalamy segmenty w jeden dataset    
    data_test, y_extended = merge_chunks(chunks, chunks_y)

    # Predykcja algorytmu dissfcm
    cluster_membership, fuzzy_labels, fpc = predict_data_ifcm(data_test, centroids)
    
    # Wyznaczenie wskaźników jakości
    silhouette_avg = silhouette_score(data_test, cluster_membership)
    davies_bouldin_avg = davies_bouldin_score(data_test, cluster_membership)
    rand = rand_score(y_extended, cluster_membership)

    if(clusters_for_each_class is None):
        # Liczba klas
        n_classes = len(clusters_for_each_class)
    
        # Głosowanie większościowe
        validation_y_predicted, cluster_to_class = classify_points_knn_eliminate_minor_class(centroids, n_classes, chunks, predict_data_issfcm, clusters_for_each_class = clusters_for_each_class)
    
        # Klasyfikujemy segmenty
        statistics = calculate_statistics(np.concatenate(chunks_y[:]), validation_y_predicted)  
    else:
         # Statystki dla klasyfikacji segmentów
        statistics, cluster_to_class = validate_segments(chunks, chunks_y, centroids, fuzzy_labels)
    
    # Klasyfikacja punktów oraz ich validacja
    statistics_points = validate_labels(chunks, chunks_y, centroids, fuzzy_labels)

    if(print_data == True):
        print_statistics(silhouette_avg, davies_bouldin_avg, rand, fpc, statistics)

    return  silhouette_avg, davies_bouldin_avg, rand, fpc, statistics, cluster_to_class, fuzzy_labels, statistics_points

def valid_data_issfcm(chunks, centroids, chunks_y, clusters_for_each_class=None, m=2, error=0.05, metric='euclidean', print_data=False):
    from libraries.FCM.ISSFCM import predict_data_issfcm
    # Scalamy segmenty w jeden dataset    
    data_test, y_extended = merge_chunks(chunks, chunks_y)

    # Predykcja algorytmu dissfcm
    cluster_membership, fuzzy_labels, fpc = predict_data_issfcm(data_test, centroids)

    # Wyznaczenie wskaźników jakości
    silhouette_avg = silhouette_score(data_test, cluster_membership)
    davies_bouldin_avg = davies_bouldin_score(data_test, cluster_membership)
    rand = rand_score(y_extended, cluster_membership)

    if(clusters_for_each_class is None):
        # Liczba klas
        n_classes = len(clusters_for_each_class)
    
        # Głosowanie większościowe
        validation_y_predicted, cluster_to_class = classify_points_knn_eliminate_minor_class(centroids, n_classes, chunks, predict_data_issfcm, clusters_for_each_class = clusters_for_each_class)
    
        # Klasyfikujemy segmenty
        statistics = calculate_statistics(np.concatenate(chunks_y[:]), validation_y_predicted)  
    else:
         # Statystki dla klasyfikacji segmentów
        statistics, cluster_to_class = validate_segments(chunks, chunks_y, centroids, fuzzy_labels)
        
    # Klasyfikacja punktów oraz ich validacja
    statistics_points = validate_labels(chunks, chunks_y, centroids, fuzzy_labels)

    if(print_data == True):
        print_statistics(silhouette_avg, davies_bouldin_avg, rand, fpc, statistics)

    return  silhouette_avg, davies_bouldin_avg, rand, fpc, statistics, cluster_to_class, fuzzy_labels, statistics_points

def valid_data_dissfcm(chunks, centroids, chunks_y, clusters_for_each_class=None, m=2, error=0.05, metric='euclidean', print_data=False):
    from libraries.FCM.DISSFCM import predict_data_dissfcm
    # Scalamy segmenty w jeden dataset    
    data_test, y_extended = merge_chunks(chunks, chunks_y)

    # Predykcja algorytmu dissfcm
    cluster_membership, fuzzy_labels, fpc = predict_data_dissfcm(data_test, centroids)
 
    # Wyznaczenie wskaźników jakości
    silhouette_avg = silhouette_score(data_test, cluster_membership)
    davies_bouldin_avg = davies_bouldin_score(data_test, cluster_membership)
    rand = rand_score(y_extended, cluster_membership)

    if(clusters_for_each_class is None):
        # Liczba klas
        n_classes = len(clusters_for_each_class)
    
        # Głosowanie większościowe
        validation_y_predicted, cluster_to_class = classify_points_knn_eliminate_minor_class(centroids, n_classes, chunks, predict_data_dissfcm, clusters_for_each_class = clusters_for_each_class)
    
        # Klasyfikujemy segmenty
        statistics = calculate_statistics(np.concatenate(chunks_y[:]), validation_y_predicted)  

    else:
        # Statystki dla klasyfikacji segmentów
        statistics, cluster_to_class = validate_segments(chunks, chunks_y, centroids, fuzzy_labels)
    
    # Klasyfikacja punktów oraz ich validacja
    statistics_points = validate_labels(chunks, chunks_y, centroids, fuzzy_labels)

    if(print_data == True):
        print_statistics(silhouette_avg, davies_bouldin_avg, rand, fpc, statistics)

    return  silhouette_avg, davies_bouldin_avg, rand, fpc, statistics, cluster_to_class, fuzzy_labels, statistics_points
    
def valid_data_fcm(chunks, centroids, chunks_y, m=2, error=0.05, metric='euclidean', print_data=False):

    data_test, y_extended = merge_chunks(chunks, chunks_y)
    
    fuzzy_labels, u0, d, jm, p, fpc = cmeans_predict(data_test.T, centroids, m=m, error=error, maxiter=1000)

    cluster_membership = np.argmax(fuzzy_labels, axis=0)
    
    # Wyznaczenie wskaźników jakości
    silhouette_avg = silhouette_score(data_test, cluster_membership)
    davies_bouldin_avg = davies_bouldin_score(data_test, cluster_membership)
    rand = rand_score(y_extended, cluster_membership)

    # Klasyfikacja segmentów oraz ich validacja
    statistics, cluster_to_class = validate_segments(chunks, chunks_y, centroids, fuzzy_labels)

    # Klasyfikacja punktów oraz ich validacja
    statistics_points = validate_labels(chunks, chunks_y, centroids, fuzzy_labels)

    if(print_data == True): 
        print_statistics(silhouette_avg, davies_bouldin_avg, rand, fpc, statistics)

    return  silhouette_avg, davies_bouldin_avg, rand, fpc, statistics, cluster_to_class, fuzzy_labels, statistics_points

def valid_data_knn(chunks, chunks_y, knn_model, print_data=False):

    # Scalamy segmenty w jeden dataset    
    data_test, y_extended = merge_chunks(chunks, chunks_y)

    # Przewidujemy klasy dla danych testowych
    cluster_membership = knn_model.predict(data_test)
    
    # Wyznaczenie wskaźników jakości
    silhouette_avg = silhouette_score(data_test, cluster_membership)
    davies_bouldin_avg = davies_bouldin_score(data_test, cluster_membership)
    rand = rand_score(y_extended, cluster_membership)

    # Statystki dla klasyfikacji segmentów
    statistics = validate_segments_knn(chunks, chunks_y, cluster_membership)

    # Klasyfikacja punktów oraz ich validacja
    statistics_points = validate_labels_knn(chunks, chunks_y, cluster_membership)

    if(print_data == True):
        print_statistics(silhouette_avg, davies_bouldin_avg, rand, 0.0, statistics)

    return  silhouette_avg, davies_bouldin_avg, rand, statistics, statistics_points

def valid_data_svm(chunks, chunks_y, svm_classifier, print_data=False):

    # Scalamy segmenty w jeden dataset    
    data_test, y_extended = merge_chunks(chunks, chunks_y)

    # Przewidujemy klasy dla danych testowych
    cluster_membership = svm_classifier.predict(data_test)
    
    # Wyznaczenie wskaźników jakości
    silhouette_avg = silhouette_score(data_test, cluster_membership)
    davies_bouldin_avg = davies_bouldin_score(data_test, cluster_membership)
    rand = rand_score(y_extended, cluster_membership)

    # Statystki dla klasyfikacji segmentów
    statistics = validate_segments_knn(chunks, chunks_y, cluster_membership)

    # Klasyfikacja punktów oraz ich validacja
    statistics_points = validate_labels_knn(chunks, chunks_y, cluster_membership)

    if(print_data == True):
        print_statistics(silhouette_avg, davies_bouldin_avg, rand, 0.0, statistics)

    return  silhouette_avg, davies_bouldin_avg, rand, statistics, statistics_points
