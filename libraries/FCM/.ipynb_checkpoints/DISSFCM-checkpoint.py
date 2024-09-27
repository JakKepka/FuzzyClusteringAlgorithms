from scipy.spatial.distance import cdist
from libraries.diagnosis_tools import DiagnosisTools, Multilist
import numpy as np
import time 
from tqdm import tqdm
from IPython.display import clear_output

# Plot functions
from libraries.plot_functions import plot_pca, plot_pca_cluster, custom_plot
from libraries.valid_data import valid_data_dissfcm
from libraries.chunks import merge_chunks
from libraries.semi_supervised_matrix import upload_semi_supervised_matrix
from libraries.chunks import create_chunks
from libraries.clusters import count_points_for_clusters, sum_probability_for_clusters, popularity_of_clusters
from libraries.clusters import compare_clusters
#################################################################################

                        ##Glosowanie turowe##


#################################################################################
from collections import Counter
from scipy import stats
from libraries.classify_segments import validate_segments, validate_segments_knn, calculate_statistics

def assign_clusters_to_classes(fuzzy_labels, centroids, y, n_classes):
    # Zliczam pierwsze punkty do jakich klas należą, następnie dopiero patrzę na segmenty.
    cluster_membership = np.argmax(fuzzy_labels, axis=0)

    count_points = np.zeros((centroids.shape[0], n_classes))
    
    for i, label in enumerate(cluster_membership):
        count_points[label, y[i]] += 1

    # Zwracamy tablicę z przyporządkowanymi klasami dla każdego clustra.
    return np.argmax(count_points, axis=1)


def assign_class_to_points(fuzzy_labels, cluster_to_class):
    cluster_membership = np.argmax(fuzzy_labels, axis=0)

    result = np.zeros(len(cluster_membership), dtype=fuzzy_labels.dtype)

    result[:] = cluster_to_class[cluster_membership]
    return result


def classify_points(trained_x, trained_y, validation_x, validation_y, centroids, metric, m, n_classes, classify_whole_segment = False, validation_x_chunked = None):
    # przynależności wszystkich punktów ze zbioru treningowego do centroidów
    fuzzy_labels_trained = create_labels(trained_x, centroids, metric, m)
    
    # przynależność klastrów do klas
    cluster_to_class = assign_clusters_to_classes(fuzzy_labels_trained, centroids, trained_y, n_classes)
    
    # przynależność wszystkich punktów ze zbioru walidacyjnego do centroidów
    fuzzy_labels_val = create_labels(validation_x, validation_y.T ,centroids, metric, m)

    validation_classified = None
    # wyznaczanie klas na podstawie przynależności do centroidów dla zbioru walidacyjnego
    if classify_whole_segment:
        validation_classified = []

        for chunk in validation_x_chunked:
            fuzzy_labels_chunk = create_labels(chunk, centroids, metric, m)
            chunk_classified = assign_class_to_points(fuzzy_labels_chunk, cluster_to_class)
            mode_value, count = stats.mode(chunk_classified)
            
            validation_classified.append(np.full(chunk_classified.shape, mode_value))
        validation_classified = np.concatenate(validation_classified)  
    else:
        validation_classified = assign_class_to_points(fuzzy_labels_val, cluster_to_class)
    
    return validation_classified
    

def majority_vote_with_elimination(class_vectors, n_classes):
    """
    Przeprowadza głosowanie większościowe z eliminacją najmniej popularnych klas.
    
    Args:
    class_vectors (list of list): Lista wektorów indeksów klas uporządkowanych według przynależności
                                  dla każdego punktu walidacyjnego.
    
    Returns:
    int: Ostateczna wybrana klasa po głosowaniu.
    """
    counter = 0
    #print(class_vectors)
    mark_deletion = np.zeros(n_classes)
    while True:
        # Zliczanie pierwszych klas (najbardziej przynależnych) dla wszystkich punktów
        first_choices = [classes[0] for classes in class_vectors if classes.size > 0]
        class_counter = Counter(first_choices)
        #print("class_vectors", class_vectors)
        #print("first_choices", first_choices)
        #return first_choices

        #print("tura: ", counter)    
        #print("first_choices: ", first_choices)
        #print("class_counter: ", class_counter)
        #return first_choices
        # Sprawdzenie, czy mamy jedną dominującą klasę
        if len(class_counter) == 1:
            return first_choices  # Zwróć dominującą klasę
        
        # Znajdź najmniej popularną klasę (lub klasy, jeśli są remisowe)
        min_count = min(class_counter.values())
        least_common_classes = [cls for cls, count in class_counter.items() if count == min_count]

        # Dla każdej klasy do usunięcia
        for cls_to_remove in least_common_classes:
            mark_deletion[cls_to_remove] = 1
            #print("Klasa do usuniecia: ", cls_to_remove)
            # Przejdź przez każdy punkt walidacyjny
            for i, classes in enumerate(class_vectors):
                # Jeśli pierwsza klasa jest tą do usunięcia, usuń ją
                if classes.size > 0 and mark_deletion[classes[0]] == 1:
                    #print("Usuwam: ", classes[0])
                    class_vectors[i] = np.delete(classes, 0)

        # Sprawdź, czy wszystkie wektory klas zostały wyeliminowane
        if all(classes.size == 1 for classes in class_vectors):
            return first_choices  # Zwróć None, jeśli wszystkie klasy zostały wyeliminowane

        if counter >= 3:
            # Zwróć pierwszą klasę, która pozostała na końcu eliminacji
            return first_choices
        
        counter += 1
def classify_with_knn_eliminate_minor(train_matrix, val_matrix, k, prototype_to_class ,n_classes, centroids):
    """
    Klasyfikuje dane walidacyjne na podstawie k najbliższych sąsiadów z użyciem macierzy przynależności.
    
    Args:
    val_matrix (numpy.ndarray): Macierz przynależności danych walidacyjnych, rozmiar [n_val x K].
    train_matrix (numpy.ndarray): Macierz przynależności danych treningowych, rozmiar [n_train x K].
    k (int): Liczba najbliższych sąsiadów do znalezienia.
    prototype_to_class (list): Lista mapująca każdy prototyp na odpowiednią klasę.
    
    Returns:
    list: Lista sklasyfikowanych klas dla każdej serii czasowej z walidacji.
    """

    n_val = val_matrix.shape[1]
    n_train = train_matrix.shape[1]
    
    classified_labels = []
    
    for i in range(n_val):
        val_series = val_matrix[:, i]
        #print("val_series ", val_series)
        # Oblicz odległość euklidesową między i-tym rzędem w val_matrix a każdym rzędem w train_matrix
        v_expanded = val_series[:, np.newaxis]  # Kształt: (8, 1)

        # Oblicz różnicę pomiędzy punktami a wektorem
        diff = train_matrix - v_expanded
        
        # Oblicz dystans Euklidesowy
        distances = np.sqrt(np.sum(diff**2, axis=0))
        
        # Znajdź indeksy k najmniejszych wartości (najbliższych sąsiadów)
        k_nearest_indices = np.argsort(distances)[:k]
        
        class_to_max_prototype = np.zeros(n_classes)
        
        distances = np.linalg.norm(centroids - v_expanded, axis=1)
        # for idx in k_nearest_indices:
        #     # Sortuj prototypy według wartości przynależności malejąco dla danego sąsiada
        #     sorted_prototypes = np.argsort(train_matrix[:, idx])[::-1]
        sorted_prototypes = np.argsort(val_series)[::-1]
    
        for prototype_idx in sorted_prototypes:
                # Mapuj prototyp na odpowiednią klasę
                mapped_class = prototype_to_class[prototype_idx]
                
                # Jeśli klasa nie była jeszcze dodana lub obecny prototyp ma większą przynależność, zaktualizuj
                if class_to_max_prototype[mapped_class] == 0 or class_to_max_prototype[mapped_class] < val_series[prototype_idx]:
                    class_to_max_prototype[mapped_class] = val_series[prototype_idx]
                else:
                    break  # Ponieważ sortowanie jest malejące, dalsze prototypy będą miały mniejszą przynależność
        
        sorted_class_indices = np.argsort(class_to_max_prototype)[::-1]  
        # Uzyskaj indeksy centroidów posortowane według odległości
        #print("val_series ", sorted_class_indices)

        # sorted_indices = np.argsort(val_series)
        # # for idx in k_nearest_indices:
        # #     # Sortuj prototypy według wartości przynależności malejąco dla danego sąsiada
        # #     sorted_prototypes = np.argsort(train_matrix[:, idx])[::-1]
            
        # for prototype_idx in sorted_indices:
        #         # Mapuj prototyp na odpowiednią klasę
        #         mapped_class = prototype_to_class[prototype_idx]
                
        #         # Jeśli klasa nie była jeszcze dodana lub obecny prototyp ma większą przynależność, zaktualizuj
        #         if class_to_max_prototype[mapped_class] == 0 or class_to_max_prototype[mapped_class] < val_series[prototype_idx]:
        #             class_to_max_prototype[mapped_class] = val_series[prototype_idx]
        #         else:
        #             break  # Ponieważ sortowanie jest malejące, dalsze prototypy będą miały mniejszą przynależność
        
        #sorted_class_indices = np.argsort(class_to_max_prototype)[::-1]       
        # Zlicz klasy k najbliższych sąsiadów
        #class_counter = Counter(k_nearest_classes)
        
        # # Zwróć klasy uporządkowane od najczęstszej do najmniej częstej
        # sorted_classes = [cls for cls, count in class_counter.most_common()]
        #print("sorted_class_indices")
        #print(sorted_class_indices)
        classified_labels.append(sorted_class_indices)
    
    # Przeprowadź głosowanie większościowe z eliminacją
    final_class = majority_vote_with_elimination(classified_labels, n_classes)
    
    return final_class


def classify_points_knn_eliminate_minor_class(trained_x, centroids, n_classes, validation_x_chunked = None, clusters_for_each_class = None, f_t = None):
    
    _, fuzzy_labels_trained, _ = predict_data_dissfcm(trained_x, centroids)
    if f_t is not None:
        fuzzy_labels_trained = f_t

    max_cluster = len(centroids)
    
    cluster_to_class = np.full(max_cluster, -1)  # Inicjalizujemy wartości np. -1 dla niezdefiniowanych
    
    for class_idx, cluster_range in clusters_for_each_class.items():
        for cluster in cluster_range:
            cluster_to_class[cluster] = class_idx
    k = 7


    validation_classified = None
    validation_classified_chunks_before_voting = []
    validation_classified_chunks_majority = []

    validation_classified_chunks_majority = []

    itr = 0
    for chunk in validation_x_chunked:
        _, fuzzy_labels_chunk, _ = predict_data_dissfcm(chunk, centroids)
        chunk_classified = classify_with_knn_eliminate_minor(fuzzy_labels_trained, fuzzy_labels_chunk, k, cluster_to_class, n_classes, centroids)  
        if chunk_classified is not None:
            mode_value, count = stats.mode(chunk_classified)
            majority = np.full(len(chunk), mode_value)
                
            validation_classified_chunks_majority.append(majority)
            
    validation_classified = np.concatenate(validation_classified_chunks_majority[:])  

    return validation_classified, cluster_to_class
    
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

    return z, new_centroids, new_fuzzy_labels.T


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
    return V, np.max(V), np.argmax(V)



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
    V_max_prev = [np.inf] * n_classes
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
            _, fuzzy_labels, _ = predict_data_dissfcm(data, centroids)
        
            # błąd rekonstrukcji
            V, V_max, V_max_cluster_id = reconstruction_error(data, fuzzy_labels, centroids, m)
            V = V[clusters]
            V_max = np.max(V)
            V_max_cluster_id = np.argmax(V)    
            # Numer iteracji pętli
            split_while_iteration = 0
            fuzzy_labels_local = fuzzy_labels[clusters, :]
   
            # Pętla Split
            print("  V[clusters]: ", V)
            print(" clusters: ", clusters)
            print(" V_max ", V_max)
            print(" V_max_prev[current_class]: ", V_max_prev[current_class])
            while (V_max > V_max_prev[current_class] and count > 0 and abs(V_max - V_max_prev[current_class]) > 1 ) and split_while_iteration < 3:
                # Funkcja Split, dzieli centroidy/generuje nowe.
                z, centroids_updated, fuzzy_labels = split_centroids(data, fuzzy_labels_local, centroids[clusters], V_max_cluster_id, m=m, metric='euclidean', maxiter=100, error=error)
                n_clusters += 1

                centroids = np.vstack((centroids[: clusters[V_max_cluster_id]], z, centroids[ clusters[V_max_cluster_id]+1:]))
          
                y_train_matrix, clusters_for_each_class = upload_semi_supervised_matrix(y_train, clusters[V_max_cluster_id], clusters_for_each_class, n_clusters, injection)
                     

                chunks_y_supervised = create_chunks(chunk_train_sizes, y_train_matrix)
                clusters = list(clusters_for_each_class[current_class])
                
                _, fuzzy_labels, _ = predict_data_dissfcm(data, centroids)
                fuzzy_labels_local = fuzzy_labels[clusters, :]

                # Ponowne obliczanie blędu rekonstrukcji
                V, V_max, V_max_cluster_id = reconstruction_error(data, fuzzy_labels, centroids, m)
                V = V[clusters]
                V_max = np.max(V)
                V_max_cluster_id = np.argmax(V)   
                # Zwiększamy numer iteracji pętli.
                split_while_iteration += 1
                   
            # Zapamiętujemy V_max z poprzedniego chunk'a
            V_max_prev[current_class] = V_max
                
            # Validacja danych
            silhouette_avg, davies_bouldin_avg, rand, fpc_test, statistics, cluster_to_class_assigned, fuzzy_labels = valid_data_dissfcm(validation_chunks, centroids, validation_chunks_y, m, error, metric, print_statistics)

            validation_y_predicted, cluster_to_class = classify_points_knn_eliminate_minor_class(np.concatenate(chunks[:]), centroids, n_classes, validation_chunks, clusters_for_each_class = clusters_for_each_class)

            statistics = calculate_statistics(np.concatenate(validation_chunks_y[:]), validation_y_predicted)  
            diagnosis_tools.add_elements(silhouette_avg, davies_bouldin_avg, fpc_test, rand, statistics)
            diagnosis_tools.add_centroids(centroids)
            diagnosis_iterations.append(diagnosis_iteration)
            
            if(visualise_data == True):
                plot_func(X_validation, centroids, fuzzy_labels, cluster_to_class_assigned)
                plot_func(X_validation, centroids, fuzzy_labels, cluster_to_class_assigned, y_validation)

                custom_plot(data, centroids, np.full(len(data),current_class), cluster_to_class, fuzzy_labels, np.concatenate(validation_chunks[:]))
                custom_plot(np.concatenate(validation_chunks[:]), centroids, validation_y_predicted, cluster_to_class, fuzzy_labels, np.concatenate(validation_chunks[:]))

            
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
    
    return diagnosis_tools, diagnosis_iterations, best_centroids


    
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
            V, V_max, V_max_cluster_id = reconstruction_error(data, fuzzy_labels, centroids, m)
            
            if(visualise_data == True):
                plot_func(data, centroids, fuzzy_labels)
                
            while V_max > V_max_prev and count > 0:
    
                # Funkcja Split, dzieli centroidy/generuje nowe.
                z, centroids, fuzzy_labels = split_centroids(data, fuzzy_labels, centroids, V_max_cluster_id, m=m, metric='euclidean', maxiter=100, error=0.05)
                n_clusters += 1
                
                # Aktualizowane chunks_y_train
                y_train_matrix, clusters_for_each_class = upload_semi_supervised_matrix(y_train, V_max_cluster_id, clusters_for_each_class, n_clusters, injection)
                chunks_y_matrix = create_chunks(chunk_train_sizes, y_train_matrix)
    
                # Ponowne obliczanie blędu rekonstrukcji
                V_max_prev = V_max
                V, V_max, V_max_cluster_id = reconstruction_error(data, fuzzy_labels, centroids, m)
    
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
    
    return diagnosis_tools, diagnosis_iterations, best_centroids