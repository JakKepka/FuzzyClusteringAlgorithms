from scipy.spatial.distance import cdist
from libraries.diagnosis_tools import DiagnosisTools, Multilist
import numpy as np
import time 
from tqdm import tqdm
from IPython.display import clear_output

from libraries.plot_functions import plot_pca, plot_pca_cluster
from libraries.valid_data import valid_data_issfcm
from libraries.process_data import merge_chunks
from libraries.semi_supervised_matrix import upload_semi_supervised_matrix
from libraries.chunks import create_chunks
from libraries.clusters import count_points_for_clusters, sum_probability_for_clusters, popularity_of_clusters

#################################################################################

                            ##Normalizacja##

#################################################################################
init_centroids = None

def normalize_columns(columns):
    # broadcast sum over columns
    normalized_columns = columns/np.sum(columns, axis=0, keepdims=1)

    return normalized_columns

def reflect_labels(y):
    result = 1 - np.sum(y, axis=0, keepdims=1)
    
    return result

def normalize_power_columns(x, exponent):
    assert np.all(x >= 0.0)

    x = x.astype(np.float64)

    # values in range [0, 1]
    x = x/np.max(x, axis=0, keepdims=True)

    # values in range [eps, 1]
    x = np.fmax(x, np.finfo(x.dtype).eps)

    if exponent < 0:
        # values in range [1, 1/eps]
        x /= np.min(x, axis=0, keepdims=True)

        # values in range [1, (1/eps)**exponent] where exponent < 0
        # this line might trigger an underflow warning
        # if (1/eps)**exponent becomes zero, but that's ok
        x = x**exponent
    else:
        # values in range [eps**exponent, 1] where exponent >= 0
        x = x**exponent

    result = normalize_columns(x)

    return result


#################################################################################

                          ##Algorytmty Trenujące##

#################################################################################

def choose_random_rows(array, c):
    if c > array.shape[0]:
        raise ValueError("Liczba wierszy do wybrania jest większa niż liczba dostępnych wierszy w tablicy.")
    
    # Wybór c unikalnych indeksów wierszy
    row_indices = np.random.choice(array.shape[0], c, replace=False)
    
    # Wybranie wierszy o wybranych indeksach
    selected_rows = array[row_indices]
    
    return selected_rows
    
def initialize_c_first_centroids(data, c):
    # Inicjalizuje biorąc pierwsze k punktów jako centroidy
    selected_rows = choose_random_rows(data, c)
    return selected_rows

def initialize_average_c_centroids(data, y_train, c):
    # Oblcza dla każdego segmentu średnią liczbę punktów
    return data[0:c,:]
    
def create_labels(data, y, centroids, metric, m):
    # Tablica dystansów
    dist = _distance(data, centroids, metric)

    # Tablica prawdopodobieństw z zwykłego algorytmu FCM
    fuzzy_labels = normalize_power_columns(dist, - 2. / (m - 1))

    # 1 - sum j = 1:C y(j)
    y_ = reflect_labels(y)
    y_ = np.tile(y_, (fuzzy_labels.shape[0], 1))
    
    fuzzy_labels = y + np.multiply(fuzzy_labels, y_)
    
    return fuzzy_labels

def _fp_coeff(u):
    # Mierzy rozmytość wyliczonych klastrów
    n = u.shape[1]
    
    return np.trace(u.dot(u.T)) / float(n)

def _distance(data, centroids, metric='euclidean'):
    # Oblicza dystans dla każdego punktu do każdego centroidu
    dist = cdist(data, centroids, metric=metric).T
    
    return np.fmax(dist, np.finfo(np.float64).eps)

def semi_supervised_cmeans0(data, y, centroids, metric, c, m):
    # Obliczanie tablicy dystansów
    dist = _distance(data, centroids, metric)

    # Obliczanie fuzzy_labels na podstawie centroidów i tablicy dystansów
    fuzzy_labels = create_labels(data, y, centroids, metric, m)

    fuzzy_labels_supervised = abs(fuzzy_labels - y)
    
    fuzzy_labels_supervised_m = fuzzy_labels_supervised ** m
    
    # Aktualizowanie centroidów
    centroids = fuzzy_labels_supervised_m.dot(data) / np.atleast_2d(fuzzy_labels_supervised_m.sum(axis=1)).T

    jm = (fuzzy_labels_supervised_m * dist ** 2).sum()
    
    return centroids, fuzzy_labels, jm, dist


def incremental_semi_supervised_fuzzy_cmeans(data, y, c, m, error, maxiter, metric = 'euclidean', init_centroid=None):
    # data jeste postaci (n_samples, k_features)

    # Struktura do której bedziemy zbierać informacje z każdej iteracji
    statistics = Multilist(['fpc'])
    
    centroids = init_centroid
    
    if(init_centroid is None):
        centroids = initialize_c_first_centroids(data, c)
        #centroids = initialize_average_c_centroids(data, y_train, k)
    
    fuzzy_labels = create_labels(data, y.T,  centroids, metric, m)

    # Initialize loop parameters
    p = 0

    # Main cmeans loop
    while p < maxiter - 1:
        fuzzy_labels_copy = fuzzy_labels.copy()
        centroids_copy = centroids.copy()

        [centroids, fuzzy_labels, Jjm, dist] = semi_supervised_cmeans0(data, y.T, centroids_copy, metric, c, m)

        fpc = _fp_coeff(fuzzy_labels)
        statistics.add_elements([fpc])
        p += 1
        
        # Stopping rule
        if np.linalg.norm(fuzzy_labels - fuzzy_labels_copy) < error and p > 1:
            break
        if np.linalg.norm(centroids_copy - centroids) < error and p > 1:
            break
            
    # Final calculations
    error = np.linalg.norm(fuzzy_labels - fuzzy_labels_copy)
    fpc = _fp_coeff(fuzzy_labels)

    return centroids, fuzzy_labels, dist, p, fpc, statistics

#################################################################################

                        ##Algorytmty Predykcyjne##

#################################################################################

def predict_data_issfcm(data_test, centroids, m=2, error=0.05, metric='euclidean'):

    fuzzy_labels, u0, d, jm, p, fpc = incremental_semi_supervised_fuzzy_cmeans_predict(data_test.T, centroids, m=m, error=error, maxiter=1000, metric=metric, init=None)
    
    cluster_membership = np.argmax(fuzzy_labels, axis=0)

    return  cluster_membership, fuzzy_labels, fpc

def incremental_semi_supervised_fuzzy_cmeans_predict(test_data, cntr_trained, m, error, maxiter, metric='euclidean', init=None, seed=None):
    c = cntr_trained.shape[0]

    # Setup u0
    if init is None:
        if seed is not None:
            np.random.seed(seed=seed)
        n = test_data.shape[1]
        u0 = np.random.rand(c, n)
        u0 = normalize_columns(u0)
        init = u0.copy()
    u0 = init
    u = np.fmax(u0, np.finfo(np.float64).eps)

    # Initialize loop parameters
    jm = np.zeros(0)
    p = 0

    # Main cmeans loop
    while p < maxiter - 1:
        u2 = u.copy()
        [u, Jjm, d] = incremental_semi_supervised_cmeans_predict0(test_data, cntr_trained, u2, c, m, metric)
        jm = np.hstack((jm, Jjm))
        p += 1

        # Stopping rule
        if np.linalg.norm(u - u2) < error:
            break

    # Final calculations
    error = np.linalg.norm(u - u2)
    fpc = _fp_coeff(u)

    return u, u0, d, jm, p, fpc 
    
def incremental_semi_supervised_cmeans_predict0(test_data, cntr, u_old, c, m, metric):

    # Normalizing, then eliminating any potential zero values.
    u_old = normalize_columns(u_old)
    u_old = np.fmax(u_old, np.finfo(np.float64).eps)
    
    um = u_old ** m
    test_data = test_data.T

    # For prediction, we do not recalculate cluster centers. The test_data is
    # forced to conform to the prior clustering.

    d = _distance(test_data, cntr, metric)
    d = np.fmax(d, np.finfo(np.float64).eps)

    jm = (um * d ** 2).sum()

    u = normalize_power_columns(d, - 2. / (m - 1))

    return u, jm, d

#################################################################################

                        ##Local incrmental##

#################################################################################

# Funkcja liczy centroidy dla kolejnych chunków, centroidy są przekazywane jako parametr inicjalizacyjny dla kolejnych iteracji algorytmu.
# Liczba punktów w algorytmie jest stała, do kolejnej iteracji algorytmu poprzednie punkty są zapominane.
# Input:
#       n_clusters - liczba centroidów
#       chunks - dane w postaci listy chunków
#       chunks_y - lista chunków labeli odpowiadających chunks. Labele nie są postaci listy tylko macierzy rozmytych przynależności do danej klasy.
#       validation_chunks - dane validacyjne
#       validation_chunks_y - labele dla danych validacyjnych
def train_local_incremental_semi_supervised_fuzzy_cmeans(n_clusters, chunks, chunks_y, chunks_y_supervised, chunks_classification_train, validation_chunks, validation_chunks_y, clusters_for_each_class, m=2, error=0.05, visualise_data=False, plot_func=plot_pca, metric='euclidean', init_centroids=init_centroids):    
    # Początek pomiaru czasu
    start_time = time.time()
    
    # Inicjalizacjia multi listy, która będzie zbierać potrzbne statystki
    diagnosis_tools = DiagnosisTools()
    diagnosis_iterations = []

    # Mergujemy chunki w dataset
    X_validation, y_validation = merge_chunks(validation_chunks, validation_chunks_y)
    # Mergujemy chunki w dataset
    X_train, y_train = merge_chunks(chunks, chunks_y)
    # Rozmiary chunkw treningowych
    chunk_train_sizes = [len(chunk) for chunk in chunks_y]
    
    
    centroids = init_centroids
    # Kolejne trenowanie modelu
    for count, data in enumerate(chunks):
            
        chunk_y_supervised = chunks_y_supervised[count]
        
        # Segment jest klasy current_class
        current_class = chunks_classification_train[count]
        
        # Wybieramy tylko centroidy do treningu, które łączą się z daną klasą.
        clusters = list(clusters_for_each_class[current_class])

        # Wybieramy centroidy które chcemy uczyć
        centroids_local = centroids[clusters]
        chunk_y_supervised_local = chunk_y_supervised[:,clusters]
        
        # Algorytm (di)ssfcm dla jednej iteracji, dla jednego chunk'a
        centroids_local, fuzzy_labels, dist, p, fpc, diagnosis_iteration = incremental_semi_supervised_fuzzy_cmeans(data, chunk_y_supervised_local, c = n_clusters, m=m, error=error, maxiter=1000, metric='euclidean', init_centroid=centroids_local)

        # Łączenie wyćwiczone centroidy z starymi
        centroids[clusters] = centroids_local
        chunk_y_supervised[:,clusters] = chunk_y_supervised_local

        # Obliczam fuzzy_labels dla przy pomocy wszystkich centroidów
        _, fuzzy_labels, fpc = predict_data_issfcm(data, centroids)
        
        if(visualise_data):
            plot_func(data, centroids, fuzzy_labels)
        
        # Validacja danych
        silhouette_avg, davies_bouldin_avg, rand, fpc_test, statistics, cluster_to_class_assigned, fuzzy_labels = valid_data_issfcm(validation_chunks, centroids, validation_chunks_y, m, error, metric)
        diagnosis_tools.add_elements(silhouette_avg, davies_bouldin_avg, fpc_test, rand, statistics)
        diagnosis_tools.add_centroids(centroids)

        diagnosis_iterations.append(diagnosis_iteration)
        
        # Czyszczenie poprzedniego outputu
        if(visualise_data == False):
            clear_output(wait=True)
        
        # Wyświetlanie paska postępu
        print('Rozważamy obecnie chunk numer: ', count)
        print('Liczba klastrów: ', n_clusters)
        tqdm(range(len(chunks)), desc="Processing", total=len(chunks), initial=count + 1)
    # Wyswielenie wyników
    _, fuzzy_labels, fpc = predict_data_issfcm(X_validation, centroids)
    plot_func(X_validation, centroids, fuzzy_labels)
    popularity_of_clusters(fuzzy_labels, n_clusters)   

    # Koniec pomiaru czasu
    end_time = time.time()
    
    # Wyświetlenie czasu wykonania
    execution_time = end_time - start_time
    print(f"Czas wykonania: {execution_time} sekund")

    return diagnosis_tools, diagnosis_iterations
    
#################################################################################

                        ##Incrmental##

#################################################################################

# Funkcja liczy centroidy dla kolejnych chunków, centroidy są przekazywane jako parametr inicjalizacyjny dla kolejnych iteracji algorytmu.
# Liczba punktów w algorytmie jest stała, do kolejnej iteracji algorytmu poprzednie punkty są zapominane.
# Input:
#       n_clusters - liczba centroidów
#       chunks - dane w postaci listy chunków
#       chunks_y - lista chunków labeli odpowiadających chunks. Labele nie są postaci listy tylko macierzy rozmytych przynależności do danej klasy.
#       validation_chunks - dane validacyjne
#       validation_chunks_y - labele dla danych validacyjnych
def train_incremental_semi_supervised_fuzzy_cmeans(n_clusters, chunks, chunks_y, chunks_y_supervised, validation_chunks, validation_chunks_y, m=2, error=0.05, visualise_data=False, plot_func=plot_pca, metric='euclidean', init_centroids=init_centroids):    
    # Początek pomiaru czasu
    start_time = time.time()
    
    # Inicjalizacjia multi listy, która będzie zbierać potrzbne statystki
    diagnosis_tools = DiagnosisTools()
    diagnosis_iterations = []

    centroids = init_centroids
    # Kolejne trenowanie modelu
    for count, data in enumerate(chunks):
  
        chunk_y_supervised = chunks_y[count]

        centroids, fuzzy_labels, dist, p, fpc, diagnosis_iteration = incremental_semi_supervised_fuzzy_cmeans(data, chunk_y_supervised, c = n_clusters, m = m, error=error, maxiter=1000, metric = 'euclidean', init_centroid=centroids)

        if(visualise_data):
            plot_func(data, centroids, fuzzy_labels)

        # Validacja danych
        silhouette_avg, davies_bouldin_avg, rand, fpc_test, statistics, cluster_to_class_assigned, fuzzy_labels = valid_data_issfcm(validation_chunks, centroids, validation_chunks_y, m, error, metric)
        diagnosis_tools.add_elements(silhouette_avg, davies_bouldin_avg, fpc_test, rand, statistics)
        diagnosis_tools.add_centroids(centroids)

        diagnosis_iterations.append(diagnosis_iteration)
        
        # Czyszczenie poprzedniego outputu
        if(visualise_data == False):
            clear_output(wait=True)
        
        # Wyświetlanie paska postępu
        print('Rozważamy obecnie chunk numer: ', count)
        print('Liczba klastrów: ', n_clusters)
        tqdm(range(len(chunks)), desc="Processing", total=len(chunks), initial=count + 1)
   
    # Koniec pomiaru czasu
    end_time = time.time()
    
    # Wyświetlenie czasu wykonania
    execution_time = end_time - start_time
    print(f"Czas wykonania: {execution_time} sekund")

    return diagnosis_tools, diagnosis_iterations

