from scipy.spatial.distance import cdist
from libraries.diagnosis_tools import DiagnosisTools, Multilist
import numpy as np
import time 
from tqdm import tqdm
from IPython.display import clear_output

# Plot functions
from libraries.plot_functions import plot_pca, plot_pca_cluster
from libraries.valid_data import valid_data_dissfcm
from libraries.chunks import merge_chunks
from libraries.semi_supervised_matrix import upload_semi_supervised_matrix
from libraries.chunks import create_chunks
from libraries.clusters import count_points_for_clusters, sum_probability_for_clusters, popularity_of_clusters
from libraries.clusters import compare_clusters
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

                            ##Trening##

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


def dynamic_incremental_semi_supervised_fuzzy_cmeans(data, y, c, m, error, maxiter, metric = 'euclidean', init_centroid=None):
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

                            ##Predykcja##

#################################################################################

def predict_data_dissfcm(data_test, centroids, m=2, error=0.05, metric='euclidean'):

    fuzzy_labels, u0, d, jm, p, fpc = dynamic_incremental_semi_supervised_fuzzy_cmeans_predict(data_test.T, centroids, m=m, error=error, maxiter=1000, metric=metric, init=None)
    
    cluster_membership = np.argmax(fuzzy_labels, axis=0)

    return  cluster_membership, fuzzy_labels, fpc


def dynamic_incremental_semi_supervised_fuzzy_cmeans_predict(test_data, cntr_trained, m, error, maxiter, metric='euclidean', init=None, seed=None):
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
        [u, Jjm, d] = semi_supervised_cmeans_predict0(test_data, cntr_trained, u2, c, m, metric)
        jm = np.hstack((jm, Jjm))
        p += 1

        # Stopping rule
        if np.linalg.norm(u - u2) < error:
            break

    # Final calculations
    error = np.linalg.norm(u - u2)
    fpc = _fp_coeff(u)

    return u, u0, d, jm, p, fpc 
    
def semi_supervised_cmeans_predict0(test_data, cntr, u_old, c, m, metric):

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

                            ##Split##

#################################################################################


# Liczy funkcje kosztów, dla danej iteracji pętli
def split_centroids_cost_function(v, data, z, m=2):
    # Inicjalizacja funkji kosztów
    J = 0

    for k in range(2):
        J += (v[:,k] ** m).dot(np.linalg.norm(data - z[k,:], axis=1)**m)

    return J

# Inicjalizacja tablicy przyporządkowań
def init_v(data, fuzzy_labels_spliting):
    # Inicjalizacja tablicy
    v = np.zeros((data.shape[0], 2))

    for i in range(data.shape[0]):
        for k in range(2):
            # Chcemy aby nowa tablica sumowała się po wierszach do wiersza tablicy przyporządkowań która modyfikujemy.
            v[i,k] = fuzzy_labels_spliting[i] / 2
    return v

# Liczymy nowo otrzymane centroidy)
def calculate_z(data, v_m):
    # Mnożenie pierwszej kolumny matrix1 przez matrix2 i sumowanie
    z1 = np.sum(v_m[:, 0][:, np.newaxis] * data, axis=0) / np.atleast_2d(v_m[:, 0].sum(axis=0)).T

    # Mnożenie drugiej kolumny matrix1 przez matrix2 i sumowanie
    z2= np.sum(v_m[:, 1][:, np.newaxis] * data, axis=0) / np.atleast_2d(v_m[:, 1].sum(axis=0)).T
    
    z = np.vstack((z1,z2))

    return z

def exchange_rows(array, row_index, new_rows):
    # Tworzenie nowej tablicy z zastąpionym wierszem
    new_array = np.vstack((array[:row_index], new_rows, array[row_index+1:]))

    return new_array

def exchange_columns(array, col_index, new_cols):
    # Tworzenie nowej tablicy z zastąpioną kolumną
    new_array = np.hstack((array[:, :col_index], new_cols, array[:, col_index+1:]))

    return new_array

def init_z_centroids(centroids, spliting_cluster, data):
    # Nowe centroidy
    z = np.random.rand(2, data.shape[1]) * 10

    # Parametry rozkładu normalnego
    mean = 0       # Średnia
    sigma = 20    # Odchylenie standardowe

    # Generowanie szumu z rozkładu normalnego
    noise1 = np.random.normal(mean, sigma, data.shape[1])
    noise2 = np.random.normal(mean, sigma, data.shape[1])

    z[0,:] = centroids[spliting_cluster,:] + noise1
    z[1,:] = centroids[spliting_cluster,:] + noise2
    
    return z

def split_centroids(data, fuzzy_labels, centroids, spliting_cluster, m=2, metric='euclidean', maxiter=100, error=0.05):

    # Transpozycja fuzzy_labels aby posiadało rozmiar (n, n_clusters)
    fuzzy_labels = fuzzy_labels.T

    # Wyciągamy tylko wiersz który podzielimy.
    fuzzy_labels_spliting =  fuzzy_labels[:, spliting_cluster]

    # Nowe centroidy
    z = init_z_centroids(centroids, spliting_cluster, data)

    # Nowa tablica z prawdopodobieństwem przyporządkowanym do elementów.
    v = init_v(data, fuzzy_labels_spliting)

    # Funkcja kosztu
    J = split_centroids_cost_function(v, data, z, m)
    
    J_prev = 0
    
    i = 0
    while (abs(J - J_prev) > error and i < maxiter):
        
        dist = _distance(data, z, metric)

        # Tablica przyporządkowań
        v = (normalize_power_columns(dist, -m) * (fuzzy_labels_spliting)).T

        v_m = v ** m

        # Wyznaczamy centroidy
        z = calculate_z(data, v_m)

        i += 1
        J_prev = J
        J = split_centroids_cost_function(v, data, z, m)

    # Zamieniamy stary centroid dwoma nowymi
    new_centroids = exchange_rows(centroids, spliting_cluster, z)

    # Zamieniamy tablicę przyporządkowań nową, zaktualizowną.
    new_fuzzy_labels = exchange_columns(fuzzy_labels, spliting_cluster, v)

    return new_centroids, new_fuzzy_labels.T


#################################################################################

                            ##Reconstruction split##

#################################################################################

# Funkcja liczy błąd rekonstrukcji dla każdego clustra, następnie zwraca nawiększą wartość
def reconstruction_error(data, fuzzy_labels, centroids, m=2):

    fuzzy_labels = fuzzy_labels.T

    # Przynależność do clustró dla punktów z data
    cluster_membership = np.argmax(fuzzy_labels.T, axis=0)
    
    n_clusters = fuzzy_labels.shape[1]
    
    # Obliczanie współczynika q, suma kwadratów danychs.
    q = (np.sum(data ** m)) ** (1/m)

    fuzzy_labels_m = fuzzy_labels ** m
    
    # Tablica do przechowywania wyników x_j
    normalized_data = np.zeros(data.shape)
    
    # Liczba wierszy w macierzy u
    N = fuzzy_labels.shape[0]
    
    # Obliczanie x_j dla każdego j
    for j in range(N):
        numerator = np.sum(fuzzy_labels_m[j, :][:, np.newaxis] * centroids, axis=0)  # Licznik sumy
        denominator = np.sum(fuzzy_labels_m[j, :])  # Mianownik sumy
        normalized_data[j,:] = numerator / denominator  
  
    V = np.zeros(n_clusters)

    for i in range(n_clusters):
        label = cluster_membership == i
        V[i] =  np.sum(np.linalg.norm(data[label] - normalized_data[label], axis=1)**m) / q

    # Zwracamy nawiększą wartość i indeks clustra
    return np.max(V), np.argmax(V)



#################################################################################

                            ##Local Incremental##

#################################################################################


# Funkcja liczy centroidy dla kolejnych chunków, centroidy są przekazywane jako parametr inicjalizacyjny dla kolejnych iteracji algorytmu.
# Liczba punktów w algorytmie jest stała, do kolejnej iteracji algorytmu poprzednie punkty są zapominane.
# Input:
#       n_clusters - liczba centroidów
#       chunks - dane w postaci listy chunków
#       chunks_y - lista chunków labeli odpowiadających chunks. Labele nie są postaci listy tylko macierzy rozmytych przynależności do danej klasy.
#       validation_chunks - dane validacyjne
#       validation_chunks_y - labele dla danych validacyjnych
def dynamic_local_train_incremental_semi_supervised_fuzzy_cmeans(n_clusters, n_classes, chunks, chunks_y, chunks_y_supervised, validation_chunks, validation_chunks_y, clusters_for_each_class, injection, m=2, error=0.05, visualise_data=False, print_statistics=False, plot_func=plot_pca_cluster, metric='euclidean', init_centroids=init_centroids):    
    # Początek pomiaru czasu
    start_time = time.time()
    centroids = init_centroids
    
    # Inicjalizacjia multi listy, która będzie zbierać potrzbne statystki
    # Zbierane statystki dla każdej iteracji.
    diagnosis_tools = DiagnosisTools()
    # Zbierane statystki dla każdej iteracji, wraz z iteracjami wewntrznymi.
    diagnosis_iterations = []

    # Mergujemy chunki w dataset
    X_validation, y_validation = merge_chunks(validation_chunks, validation_chunks_y)
    # Mergujemy chunki w dataset
    X_train, y_train = merge_chunks(chunks, chunks_y)
    # Rozmiary chunkw treningowych
    chunk_train_sizes = [len(chunk) for chunk in chunks_y]

    # Najlepsze centroidy inicjalizacja
    silhouette_avg, davies_bouldin_avg, rand, fpc_test, statistics, cluster_to_class_assigned, fuzzy_labels = valid_data_dissfcm(validation_chunks, centroids, validation_chunks_y, m, error, metric, print_statistics)
    best_centroids = init_centroids
    best_centroids_statistics = statistics
    
    # Tablica blędów poprzednich dla każdej klasy
    V_max_prev = [0] * n_classes
    cluster_to_class_assigned = 0

    # Kolejne trenowanie modelu
    with tqdm(total=len(chunks), desc="Processing") as pbar:
        # Kolejne trenowanie modelu
        for count, data in enumerate(chunks):
    
            chunk_y_supervised = chunks_y_supervised[count]
    
            # Segment jest klasy current_class
            current_class = chunks_y[count][0]
            
            # Wybieramy tylko centroidy do treningu, które łączą się z daną klasą.
            clusters = list(clusters_for_each_class[current_class])
    
            # Wybieramy centroidy które chcemy uczyć
            centroids_local = centroids[clusters]
            chunk_y_supervised_local = chunk_y_supervised[:,clusters]
            
            # Algorytm (di)ssfcm dla jednej iteracji, dla jednego chunk'a
            centroids_local, fuzzy_labels, dist, p, fpc, diagnosis_iteration = dynamic_incremental_semi_supervised_fuzzy_cmeans(data, chunk_y_supervised_local, c=n_clusters, m=m, error=error, maxiter=1000, metric='euclidean', init_centroid=centroids_local)
            
            # Łączenie wyćwiczone centroidy z starymi
            centroids[clusters] = centroids_local
            chunk_y_supervised[:,clusters] = chunk_y_supervised_local
            
            # Predykcja algorytmu dissfcm
            cluster_membership, fuzzy_labels, fpc = predict_data_dissfcm(X_train, centroids)
        
            # błąd rekonstrukcji
            V_max, V_max_cluster_id = reconstruction_error(X_train, fuzzy_labels, centroids, m)
    
            # Numer iteracji pętli
            split_while_iteration = 0
            
            # Pętla Split
            while (V_max > V_max_prev[current_class] and count > 0) and split_while_iteration < 10:
                # Funkcja Split, dzieli centroidy/generuje nowe.
                centroids, fuzzy_labels = split_centroids(X_train, fuzzy_labels, centroids, V_max_cluster_id, m=m, metric='euclidean', maxiter=100, error=error)
                n_clusters += 1
                
                # Aktualizowane chunks_y_train
                y_train_matrix, clusters_for_each_class = upload_semi_supervised_matrix(y_train, V_max_cluster_id, clusters_for_each_class, n_clusters, injection)
                chunks_y_supervised = create_chunks(chunk_train_sizes, y_train_matrix)
    
                # Ponowne obliczanie blędu rekonstrukcji
                V_max, V_max_cluster_id = reconstruction_error(X_train, fuzzy_labels, centroids, m)
    
                # Aktualizacja numery rozaptrywanej obecnie klasy
                for i in range(n_classes):
                    if(V_max_cluster_id in clusters_for_each_class[i]):
                        current_class = i
                        
                # Aktualizujemy błąd w trakcie działania pętli (do rozważenia)
                #V_max_prev[current_class] = V_max
    
                # Zwiększamy numer iteracji pętli.
                split_while_iteration += 1
                   
            # Zapamiętujemy V_max z poprzedniego chunk'a
            V_max_prev[current_class] = V_max
                
            # Validacja danych
            silhouette_avg, davies_bouldin_avg, rand, fpc_test, statistics, cluster_to_class_assigned, fuzzy_labels = valid_data_dissfcm(validation_chunks, centroids, validation_chunks_y, m, error, metric, print_statistics)
            diagnosis_tools.add_elements(silhouette_avg, davies_bouldin_avg, fpc_test, rand, statistics)
            diagnosis_tools.add_centroids(centroids)
            diagnosis_iterations.append(diagnosis_iteration)
            
            if(visualise_data == True):
                plot_func(X_validation, centroids, fuzzy_labels, cluster_to_class_assigned)
                plot_func(X_validation, centroids, fuzzy_labels, cluster_to_class_assigned, y_validation)
            
            # Szukamy najlepszych centroidów
            if(compare_clusters(best_centroids_statistics, statistics) == True):
                best_centroids_statistics = statistics 
                best_centroids = centroids
                
            # Wyświetlanie paska postępu
            pbar.update(1)

    if(visualise_data == True):
        # Wyswielenie wyników
        silhouette_avg, davies_bouldin_avg, rand, fpc_test, statistics, cluster_to_class_assigned, fuzzy_labels = valid_data_dissfcm(validation_chunks, centroids, validation_chunks_y, m, error, metric, print_statistics)
        plot_func(X_validation, centroids, fuzzy_labels, cluster_to_class_assigned)
        popularity_of_clusters(fuzzy_labels, n_clusters)
        
    # Koniec pomiaru czasu
    end_time = time.time()
    
    # Wyświetlenie czasu wykonania
    execution_time = end_time - start_time
    print(f"Czas wykonania: {execution_time} sekund")
    
    return diagnosis_tools, diagnosis_iterations, best_centroids, best_centroids_statistics


    
#################################################################################

                            ##Incremental##

#################################################################################


# Funkcja liczy centroidy dla kolejnych chunków, centroidy są przekazywane jako parametr inicjalizacyjny dla kolejnych iteracji algorytmu.
# Liczba punktów w algorytmie jest stała, do kolejnej iteracji algorytmu poprzednie punkty są zapominane.
# Input:
#       n_clusters - liczba centroidów
#       chunks - dane w postaci listy chunków
#       chunks_y - lista chunków labeli odpowiadających chunks. Labele nie są postaci listy tylko macierzy rozmytych przynależności do danej klasy.
#       validation_chunks - dane validacyjne
#       validation_chunks_y - labele dla danych validacyjnych
def dynamic_train_incremental_semi_supervised_fuzzy_cmeans(n_clusters, chunks, chunks_y, chunks_y_matrix, validation_chunks, validation_chunks_y, clusters_for_each_class, injection, m=2, error=0.05, visualise_data=False, print_statistics=False, plot_func=plot_pca, metric='euclidean', init_centroids=init_centroids):    
    # Początek pomiaru czasu
    start_time = time.time()
    # Inicjalizacja centroidów
    centroids = init_centroids
    
    # Inicjalizacjia multi listy, która będzie zbierać potrzbne statystki
    # Zbierane statystki dla każdej iteracji.
    diagnosis_tools = DiagnosisTools()
    # Zbierane statystki dla każdej iteracji, wraz z iteracjami wewntrznymi.
    diagnosis_iterations = []

    # Mergujemy chunki w dataset
    X_validation, y_validation = merge_chunks(validation_chunks, validation_chunks_y)
    # Mergujemy chunki w dataset
    X_train, y_train = merge_chunks(chunks, chunks_y)
    # Rozmiary chunkw treningowych
    chunk_train_sizes = [len(chunk) for chunk in chunks_y]

    # Najlepsze centroidy inicjalizacja
    silhouette_avg, davies_bouldin_avg, rand, fpc_test, statistics, cluster_to_class_assigned, fuzzy_labels = valid_data_dissfcm(validation_chunks, centroids, validation_chunks_y, m, error, metric, print_statistics)
    best_centroids = init_centroids
    best_centroids_statistics = statistics
    
    V_max_prev = np.inf
    
    # Kolejne trenowanie modelu
    with tqdm(total=len(chunks), desc="Processing") as pbar:
        for count, data in  enumerate(chunks):
                
            chunk_y_supervised = chunks_y_matrix[count]
    
            # Algorytm ssfcm dla jednej iteracji, dla jednego chunk'a
            centroids, fuzzy_labels, dist, p, fpc, diagnosis_iteration = dynamic_incremental_semi_supervised_fuzzy_cmeans(data, chunk_y_supervised, c = n_clusters, m = m, error=error, maxiter=1000, metric = 'euclidean', init_centroid=centroids)
            
            # błąd rekonstrukcji
            V_max, V_max_cluster_id = reconstruction_error(data, fuzzy_labels, centroids, m)
            
            if(visualise_data == True):
                plot_func(data, centroids, fuzzy_labels)
                
            while V_max > V_max_prev and count > 0:
    
                # Funkcja Split, dzieli centroidy/generuje nowe.
                centroids, fuzzy_labels = split_centroids(data, fuzzy_labels, centroids, V_max_cluster_id, m=m, metric='euclidean', maxiter=100, error=0.05)
                n_clusters += 1
                
                # Aktualizowane chunks_y_train
                y_train_matrix, clusters_for_each_class = upload_semi_supervised_matrix(y_train, V_max_cluster_id, clusters_for_each_class, n_clusters, injection)
                chunks_y_matrix = create_chunks(chunk_train_sizes, y_train_matrix)
    
                # Ponowne obliczanie blędu rekonstrukcji
                V_max_prev = V_max
                V_max, V_max_cluster_id = reconstruction_error(data, fuzzy_labels, centroids, m)
    
            V_max_prev = V_max
            
            # Validacja danych
            silhouette_avg, davies_bouldin_avg, rand, fpc_test, statistics, cluster_to_class_assigned, fuzzy_labels = valid_data_dissfcm(validation_chunks, centroids, validation_chunks_y, m, error, metric, print_statistics)
            diagnosis_tools.add_elements(silhouette_avg, davies_bouldin_avg, fpc_test, rand, statistics)
            diagnosis_tools.add_centroids(centroids)
            diagnosis_iterations.append(diagnosis_iteration)
    
            # Szukamy najlepszych centroidów
            if(compare_clusters(best_centroids_statistics, statistics) == True):
                best_centroids_statistics = statistics 
                best_centroids = centroids
    
            # Update paska
            pbar.update(1)
        
    if(visualise_data == True):
        # Wyswielenie wyników
        _, fuzzy_labels, fpc = predict_data_dissfcm(X_validation, centroids)
        plot_func(X_validation, centroids, fuzzy_labels)
        popularity_of_clusters(fuzzy_labels, n_clusters)

    # Koniec pomiaru czasu
    end_time = time.time()
    
    # Wyświetlenie czasu wykonania
    execution_time = end_time - start_time
    print(f"Czas wykonania: {execution_time} sekund")
    
    return diagnosis_tools, diagnosis_iterations, best_centroids, best_centroids_statistics