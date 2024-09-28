from sklearn.preprocessing import StandardScaler
from skfuzzy.cluster import cmeans, cmeans_predict
from sklearn.metrics import pairwise_distances_argmin_min

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import arff
import math
import sys
import time

from libraries.process_data import convert_to_dataframe, reshape_data, sort_by_class, shuffle_dataset_with_chunk_sizes
from libraries.process_data import stratify_data, extend_list, map_strings_to_ints, shuffle_dataset, knn_with_library, select_subset
from libraries.clusters import average_by_class, generate_clusters_proportional, label_vector_to_semi_supervised_matrix, create_semi_supervised_matrix, upload_semi_supervised_matrix
from libraries.plot_functions import visualise_labeled_data_all_dimensions, plot_pca, plot_pca_standard
from libraries.plot_functions import create_set_for_stats, compare_models_statistics, overview_plot
from tslearn.datasets import UCR_UEA_datasets
from libraries.chunks import create_chunks, create_dataset_chunks, merge_chunks
from libraries.valid_data import  valid_data_fcm, valid_data_ifcm, valid_data_issfcm, valid_data_dissfcm, valid_data_knn, valid_data_rocket

# Diffrent classification algorithms
from sktime.classification.kernel_based import RocketClassifier

# FCM's
from libraries.FCM.IFCM import incremental_fuzzy_cmeans
from libraries.FCM.ISSFCM import incremental_semi_supervised_fuzzy_cmeans
from libraries.FCM.DISSFCM import dynamic_incremental_semi_supervised_fuzzy_cmeans
from libraries.FCM.DISSFCM import dynamic_train_incremental_semi_supervised_fuzzy_cmeans
from libraries.FCM.DISSFCM import dynamic_local_train_incremental_semi_supervised_fuzzy_cmeans
from libraries.FCM.IFCM import train_incremental_local_fuzzy_cmeans
from libraries.FCM.IFCM import train_incremental_fuzzy_cmeans
from libraries.FCM.IFCM import train_incremental_fuzzy_cmeans_extending_data
from libraries.FCM.ISSFCM import train_incremental_semi_supervised_fuzzy_cmeans
from libraries.FCM.ISSFCM import train_local_incremental_semi_supervised_fuzzy_cmeans


def test_non_incremental_algorithms(n_clusters, n_classes, X_train, y_train, y_train_matrix, X_test, y_test, chunks_test, chunks_test_y, error=0.05, m=2, plot_func=plot_pca, metric='euclidean', visualise_non_incremental_data=False):
    models = {}

    ###########################################################################################################
    # Początek pomiaru czasu
    start_time = time.time()
    
    # Testowanie bibliotecznego modelu fcm
    print('  Algorytm biblioteczny FCM')
    centroids, fuzzy_labels, u0, d, jm, p, fpc = cmeans(X_train.T, c=n_clusters, m=m, error=error, maxiter=1000, init=None)
    
    silhouette_avg, davies_bouldin_avg, rand, fpc, stats, cluster_to_class, fuzzy_labels = valid_data_fcm(chunks_test, centroids, chunks_test_y, m, error)

    if (visualise_non_incremental_data == True):
        plot_func(X_test, centroids, fuzzy_labels)
    
    models['FCM_library'] = create_set_for_stats(silhouette_avg, davies_bouldin_avg, rand, fpc, stats)
    
    # Koniec pomiaru czasu
    end_time = time.time()
    
    # Wyświetlenie czasu wykonania
    execution_time = end_time - start_time
    print(f"Czas wykonania: {execution_time} sekund")
    
    ###########################################################################################################
    from libraries.FCM.IFCM import cmeans0
    
    # Początek pomiaru czasu
    start_time = time.time()
    
    # Testowanie własnej implementacji fcm
    print('  Własna implementacja FCM')
    centroids, fuzzy_labels, dist, p, fpc, diagnosis_iteration = incremental_fuzzy_cmeans(X_train, c=n_clusters, m=m, error=error, maxiter=1000, metric = metric, init_centroid=None)
    
    silhouette_avg, davies_bouldin_avg, rand, fpc, stats, cluster_to_class, fuzzy_labels = valid_data_ifcm(chunks_test, centroids, chunks_test_y, m, error)
    
    if (visualise_non_incremental_data == True):
        plot_func(X_test, centroids, fuzzy_labels)
    
    models['FCM'] = create_set_for_stats(silhouette_avg, davies_bouldin_avg, rand, fpc, stats)
    
    # Koniec pomiaru czasu
    end_time = time.time()
    
    # Wyświetlenie czasu wykonania
    execution_time = end_time - start_time
    print(f"Czas wykonania: {execution_time} sekund")
    
    ###########################################################################################################
    # Testowanie własnej implementacji semi-supervised fcm
    
    # Początek pomiaru czasu
    start_time = time.time()
    
    print('  Algorytm SSFCM')
    centroids, fuzzy_labels, dist, p, fpc, diagnosis_iteration = incremental_semi_supervised_fuzzy_cmeans(X_train, y_train_matrix, c = n_clusters, m = m, error=error, maxiter=1000, metric = metric, init_centroid=None)
    
    silhouette_avg, davies_bouldin_avg, rand, fpc, stats, cluster_to_class, fuzzy_labels = valid_data_issfcm(chunks_test, centroids, chunks_test_y, m, error)
    
    if (visualise_non_incremental_data == True):
        plot_func(X_test, centroids, fuzzy_labels)
    
    models['SSFCM'] = create_set_for_stats(silhouette_avg, davies_bouldin_avg, rand, fpc, stats)
    
    # Koniec pomiaru czasu
    end_time = time.time()
    
    # Wyświetlenie czasu wykonania
    execution_time = end_time - start_time
    print(f"Czas wykonania: {execution_time} sekund")
    
    ##########################################################################################################
    # Testiwabeu knn 

    # Scalamy segmenty w jeden dataset    
    data_test, y_extended = merge_chunks(chunks_test, chunks_test_y)
    
    # Początek pomiaru czasu
    start_time = time.time()
    
    k = 7
    print(f'  KNN   k = {k}')
    knn_model = knn_with_library(X_train, y_train, k)
    
    cluster_membership = knn_model.predict(data_test)
    
    silhouette_avg, davies_bouldin_avg, rand, statistics = valid_data_knn(chunks_test, chunks_test_y, knn_model)
    
    models['KNN'] = create_set_for_stats(silhouette_avg, davies_bouldin_avg, rand, 0.0, statistics)
    
    if (visualise_non_incremental_data == True):
        plot_pca_standard(data_test, cluster_membership)
    
    # Koniec pomiaru czasu
    end_time = time.time()
    
    # Wyświetlenie czasu wykonania
    execution_time = end_time - start_time
    print(f"Czas wykonania: {execution_time} sekund")
    
    ###########################################################################################################
    # Testowanie knn 
    
    # Początek pomiaru czasu
    start_time = time.time()
    
    print(f'  KNN z 40% danymi treningowymi, k = {k}')
    
    X_train_subset, y_train_subset = select_subset(X_train, y_train, p=0.3)
    
    knn_model_p = knn_with_library(X_train_subset, y_train_subset, k)
    
    cluster_membership = knn_model.predict(data_test)
    
    silhouette_avg, davies_bouldin_avg, rand, statistics = valid_data_knn(chunks_test, chunks_test_y, knn_model_p)
    
    models['KNN p%'] = create_set_for_stats(silhouette_avg, davies_bouldin_avg, rand, 0.0, statistics)
    
    if (visualise_non_incremental_data == True):
        plot_pca_standard(data_test, cluster_membership)
    
    # Koniec pomiaru czasu
    end_time = time.time()
    
    # Wyświetlenie czasu wykonania
    execution_time = end_time - start_time
    print(f"Czas wykonania: {execution_time} sekund")
    
    ###########################################################################################################

    ###########################################################################################################
    # ROCKET
    # Początek pomiaru czasu
    start_time = time.time()
    
    print(f'  RocketClassifier')

    n_samples, n_features = X_train.shape
    # Konwersja danych 2D na panel danych (sktime)
    X_train_panel = pd.DataFrame({f'feature_{i}': [pd.Series(X_train[j, :]) for j in range(n_samples)] 
                                  for i in range(n_features)})

    n_samples, n_features = data_test.shape
    X_test_panel = pd.DataFrame({f'feature_{i}': [pd.Series(data_test[j, :]) for j in range(n_samples)] 
                                 for i in range(n_features)})

    # Inicjalizacja RocketClassifier
    rocket_classifier = RocketClassifier()
    
    # Trenowanie modelu na danych treningowych
    rocket_classifier.fit(X_train_panel, y_train)

    cluster_membership = rocket_classifier.predict(X_test_panel)
    
    silhouette_avg, davies_bouldin_avg, rand, statistics = valid_data_rocket(chunks_test, chunks_test_y, X_test_panel, rocket_classifier)
    
    models['ROCKET'] = create_set_for_stats(silhouette_avg, davies_bouldin_avg, rand, 0.0, statistics)
    
    if (visualise_non_incremental_data == True):
        plot_pca_standard(data_test, cluster_membership)
    
    # Koniec pomiaru czasu
    end_time = time.time()
    
    # Wyświetlenie czasu wykonania
    execution_time = end_time - start_time
    print(f"Czas wykonania: {execution_time} sekund")
###########################################################################################################

    
    # porównanie wyników
    compare_models_statistics(models)

    return models


# This function tests the given dataset with the set parameters using various methods: IFCM, FCM, ISSFCM, DISSFCM, and KNN.
def test_dataset(chunk_length_train = 1000, chunk_length_test = 50, std_div = 0, n_clusters = 8, n_classes = 4, dim = 6, injection = 1.0, m = 2, error = 0.05, stratify_percantage = 1.0, dataset_name = 'BasicMotions', visualise_processed_data = False, visualise_non_incremental_data = False, visualise_incremental_data = False, visualise_output_of_incremental_data = False, print_statistics_incremental_data = False):

    # Początek pomiaru czasu
    start_time = time.time()
    
    # Inicjalizacja obiektu odpowiedzialnego za zbiory danych
    ucr_uea = UCR_UEA_datasets()

    # Pobieranie zbioru danych 'EigenWorms'
    X_train_, y_train_, X_test_, y_test_ = ucr_uea.load_dataset(dataset_name)

    if(X_train_ is None):
        raise ValueError("Wystąpił błąd: nie pobrano dataset'u. Sprawdź czy istnieje.")

    X_train, y_train = convert_to_dataframe(X_train_, y_train_)
    X_test, y_test = convert_to_dataframe(X_test_, y_test_)
    
    sample_size = X_train.shape[1]

    # Poprawiamy rozmiar danych do 2d
    X_train, y_train = reshape_data(X_train, y_train, sample_size)
    X_test, y_test = reshape_data(X_test, y_test, sample_size)
    # Stratyfikacja danych
    if stratify_percantage < 1.0:
        X_train, y_train = stratify_data(X_train, y_train, stratify_percantage)
        X_test, y_test = stratify_data(X_train, y_train, stratify_percantage)

    # Zamienianie stringów na int
    y_train = map_strings_to_ints(y_train)
    y_test = map_strings_to_ints(y_test)

    # Dziele punkty na segmenty średniej weilkości chunk_length_train z odchyleniem standardowym std_div. Segmenty są jednej klasy
    # Następnie tasuje w ten sposób utworzone segmenty
    X_train, y_train, chunk_train_sizes = shuffle_dataset_with_chunk_sizes(X_train, y_train, chunk_length_train, std_div)
    X_test, y_test, chunk_test_sizes = shuffle_dataset_with_chunk_sizes(X_test, y_test, chunk_length_test, std_div)

    # Wyświetlamy zlabelowane dane oraz każdy ich wymiar
    if(visualise_processed_data == True):
        # Wizualizacja każdego wymiaru danych z osobna
        visualise_labeled_data_all_dimensions(X_train, y_train, dim)

    # Dane losowo potasowane
    X_train_shuffled, y_train_shuffled = shuffle_dataset(X_train, y_train)
    X_test_shuffled, y_test_shuffled = shuffle_dataset(X_test, y_test)

    # Wyświetlamy zlabelowane dane oraz każdy ich wymiar
    if(visualise_processed_data == True):
        # Wizualizacja każdego wymiaru danych z osobna po potasowaniu.
        visualise_labeled_data_all_dimensions(X_train_shuffled, y_train_shuffled, dim)

    # Inicjalizacja centroidów oraz stworzenie y_matrix_label dla odmian algorytmu semi-supervised.
    y_train_matrix, init_centroids, clusters_for_each_class = create_semi_supervised_matrix(X_train, y_train, n_clusters)
    y_train_matrix_shuffled, init_centroids_shuffled, clusters_for_each_class_shuffled = create_semi_supervised_matrix(X_train_shuffled, y_train_shuffled, n_clusters, injection=injection)

    # Dzielimy dane oraz labele na chunki długości elementów listy chunk_train_sizes (ew. test)
    chunks, chunks_y, chunks_y_matrix = create_dataset_chunks(chunk_train_sizes, X_train, y_train, y_train_matrix)
    chunks_test, chunks_test_y, _ = create_dataset_chunks(chunk_test_sizes, X_test, y_test)
    
    # Dzielimy dane oraz labele na chunki długości elementów listy chunk_train_sizes (ew. test)
    chunks_shuffled, chunks_y_shuffled, chunks_y_matrix_shuffled = create_dataset_chunks(chunk_train_sizes, X_train_shuffled, y_train_shuffled, y_train_matrix_shuffled)
    chunks_test_shuffled, chunks_test_y_shuffled, _ = create_dataset_chunks(chunk_test_sizes, X_test_shuffled, y_test_shuffled)

    # Porównujemy algorytmy FCM nie inkrementacyjne oraz knn
    test_non_incremental_algorithms(n_clusters, n_classes, X_train, y_train, y_train_matrix, X_test, y_test, chunks_test, chunks_test_y, error=error, m=m, visualise_non_incremental_data=visualise_non_incremental_data)

#########################################################################################################################################

    output = {}
    
    print('Local DISSFCM')
    # Trenowanie Local DISSFCM 
    diagnosis_chunk_ldissfcm, diagnosis_iterations_ldissfcm, best_centroids_ldissfcm, best_centroids_statistics_ldissfcm = dynamic_local_train_incremental_semi_supervised_fuzzy_cmeans(n_clusters, n_classes, chunks, chunks_y, chunks_y_matrix, chunks_test, chunks_test_y, clusters_for_each_class.copy(), injection, m=m, visualise_data=visualise_incremental_data,  
    print_statistics=print_statistics_incremental_data, init_centroids=init_centroids.copy())

    if(visualise_output_of_incremental_data == True):
        overview_plot(diagnosis_chunk_ldissfcm, diagnosis_iterations_ldissfcm)

    # Zapisz wyniki do słownika
    output['Local DISSFCM'] = [diagnosis_chunk_ldissfcm, diagnosis_iterations_ldissfcm, best_centroids_ldissfcm, best_centroids_statistics_ldissfcm]
    
    print('DISSFCM')
    # Trenowanie DISSFCM
    diagnosis_chunk_dissfcm, diagnosis_iterations_dissfcm, best_centroids_dissfcm, best_centroids_statistics_dissfcm = dynamic_train_incremental_semi_supervised_fuzzy_cmeans(n_clusters, chunks_shuffled, chunks_y_shuffled, chunks_y_matrix_shuffled, chunks_test, chunks_test_y, clusters_for_each_class_shuffled.copy(), injection, m=m, visualise_data=visualise_incremental_data, 
    print_statistics = print_statistics_incremental_data, init_centroids=init_centroids_shuffled.copy())

    if(visualise_output_of_incremental_data == True):
        overview_plot(diagnosis_chunk_dissfcm, diagnosis_iterations_dissfcm)

    # Zapisz wyniki do słownika
    output['DISSFCM'] = [diagnosis_chunk_dissfcm, diagnosis_iterations_dissfcm, best_centroids_dissfcm, best_centroids_statistics_dissfcm]
  
    print('IFCM')
    # Trenowanie IFCM
    diagnosis_chunk_ifcm, diagnosis_iterations_ifcm, best_centroids_ifcm, best_centroids_statistics_ifcm = train_incremental_fuzzy_cmeans(n_clusters, chunks_shuffled, chunks_test, chunks_test_y, visualise_data=visualise_incremental_data, print_statistics=print_statistics_incremental_data, plot_func=plot_pca, init_centroids=init_centroids_shuffled.copy())

    if(visualise_output_of_incremental_data == True):
        overview_plot(diagnosis_chunk_ifcm, diagnosis_iterations_ifcm)

    # Zapisz wyniki do słownika
    output['IFCM'] = [diagnosis_chunk_ifcm, diagnosis_iterations_ifcm, best_centroids_ifcm, best_centroids_statistics_ifcm]
  
    print('Local IFCM')
    # Trenowanie Local IFCM
    diagnosis_chunk_lifcm, diagnosis_iterations_lifcm, best_centroids_lifcm, best_centroids_statistics_lifcm = train_incremental_local_fuzzy_cmeans(n_clusters, chunks, chunks_y, chunks_test, chunks_test_y, visualise_data=visualise_incremental_data, 
    print_statistics=print_statistics_incremental_data, plot_func=plot_pca, clusters_for_each_class=clusters_for_each_class.copy(), init_centroids=init_centroids.copy())

    if(visualise_output_of_incremental_data == True):
        overview_plot(diagnosis_chunk_lifcm, diagnosis_iterations_lifcm)

    # Zapisz wyniki do słownika
    output['Local IFCM'] = [diagnosis_chunk_lifcm, diagnosis_iterations_lifcm, best_centroids_lifcm, best_centroids_statistics_lifcm]
  
    print('extending IFCM')
    # Trenowanie z rozszerzającymi się danymi IFCM
    diagnosis_chunk_eifcm, diagnosis_iterations_eifcm, best_centroids_eifcm, best_centroids_statistics_eifcm = train_incremental_fuzzy_cmeans_extending_data(n_clusters, chunks_shuffled, chunks_test, chunks_test_y, visualise_data=visualise_incremental_data, print_statistics=print_statistics_incremental_data, init_centroids=init_centroids_shuffled.copy())
    
    if(visualise_output_of_incremental_data == True):
        overview_plot(diagnosis_chunk_eifcm, diagnosis_iterations_eifcm)
    
    # Zapisz wyniki do słownika
    output['extending IFCM'] = [diagnosis_chunk_eifcm, diagnosis_iterations_eifcm, best_centroids_eifcm, best_centroids_statistics_eifcm]
  
    print('ISSFCM')
    # Trenowanie ISSFCM
    diagnosis_chunk_issfcm, diagnosis_iterations_issfcm, best_centroids_issfcm, best_centroids_statistics_issfcm = train_incremental_semi_supervised_fuzzy_cmeans(n_clusters, chunks_shuffled, chunks_y_shuffled, chunks_y_matrix, chunks_test, chunks_test_y, m=m, visualise_data=visualise_incremental_data, print_statistics=print_statistics_incremental_data, init_centroids=init_centroids_shuffled.copy())
    
    if(visualise_output_of_incremental_data == True):
        overview_plot(diagnosis_chunk_issfcm, diagnosis_iterations_issfcm)

    # Zapisz wyniki do słownika
    output['ISSFCM'] = [diagnosis_chunk_issfcm, diagnosis_iterations_issfcm, best_centroids_issfcm, best_centroids_statistics_issfcm]
  
    print('Local ISSFCM')
    # Trenowanie Local ISSFCM
    diagnosis_chunk_lissfcm, diagnosis_iterations_lissfcm,  best_centroids_lissfcm, best_centroids_statistics_lissfcm = train_local_incremental_semi_supervised_fuzzy_cmeans(n_clusters, chunks, chunks_y, chunks_y_matrix, chunks_test, chunks_test_y, clusters_for_each_class.copy(), m=m, visualise_data=visualise_incremental_data, init_centroids=init_centroids.copy())
    
    if(visualise_output_of_incremental_data == True):
        overview_plot(diagnosis_chunk_lissfcm, diagnosis_iterations_lissfcm)

    # Zapisz wyniki do słownika
    output['Local ISSFCM'] = [diagnosis_chunk_lissfcm, diagnosis_iterations_lissfcm, best_centroids_lissfcm, best_centroids_statistics_lissfcm]
  
    # Koniec pomiaru czasu
    end_time = time.time()
    
    # Wyświetlenie czasu wykonania
    execution_time = end_time - start_time
    print(f"Czas wykonania wszystkich algorytmów: {execution_time} sekund")

    # Zwracamy słownik wraz z wynikami oraz centroidami
    return output









    
    


        