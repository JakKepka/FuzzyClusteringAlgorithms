from scipy.spatial.distance import cdist
from libraries.diagnosis_tools import DiagnosisTools, Multilist
import numpy as np


#################################################################################

                            ##Normalizacja##

#################################################################################

def normalize_columns(columns):
    # broadcast sum over columns
    normalized_columns = columns/np.sum(columns, axis=0, keepdims=1)

    return normalized_columns

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

def create_labels(data, centroids, metric, m):
    # Tablica dystansów
    dist = _distance(data, centroids, metric)
    
    # Tablica prawdopodobieństw
    fuzzy_labels = normalize_power_columns(dist, - 2. / (m - 1))
    
    return fuzzy_labels

def _fp_coeff(u):
    # Mierzy rozmytość wyliczonych klastrów
    n = u.shape[1]
    
    return np.trace(u.dot(u.T)) / float(n)

def _distance(data, centroids, metric='euclidean'):
    # Oblicza dystans dla każdego punktu do każdego centroidu
    dist = cdist(data, centroids, metric=metric).T
    
    return np.fmax(dist, np.finfo(np.float64).eps)

def cmeans0(data, centroids, metric, c, m):
    # Obliczanie tablicy dystansów
    dist = _distance(data, centroids, metric)

    # Obliczanie fuzzy_labels na podstawie centroidów i tablicy dystansów
    fuzzy_labels = create_labels(data, centroids, metric, m)
    
    fuzzy_labels_m = fuzzy_labels ** m

    # Aktualizowanie centroidów
    centroids = fuzzy_labels_m.dot(data) / np.atleast_2d(fuzzy_labels_m.sum(axis=1)).T

    jm = (fuzzy_labels_m * dist ** 2).sum()
    
    return centroids, fuzzy_labels, jm, dist


def incremental_fuzzy_cmeans(data, c, m, error, maxiter, metric = 'euclidean', init_centroid=None, m_low_boost = True, m_low = 1.1, m_low_iter = 3):
    # data jeste postaci (n_samples, k_features)
    
    # Struktura do której bedziemy zbierać informacje z każdej iteracji
    statistics = Multilist(['fpc', 'jm'])
    
    centroids = init_centroid
    
    if(init_centroid is None):
        centroids = initialize_c_first_centroids(data, c)

    fuzzy_labels = create_labels(data, centroids, metric, m)

    # Initialize loop parameters
    p = 0

    # Main cmeans loop
    while p < maxiter - 1:
        fuzzy_labels_copy = fuzzy_labels.copy()
        centroids_copy = centroids.copy()

        if p < m_low_iter and m_low_boost:
            [centroids, fuzzy_labels, Jjm, dist] = cmeans0(data, centroids_copy, metric, c, m_low)
        else:
            [centroids, fuzzy_labels, Jjm, dist] = cmeans0(data, centroids_copy, metric, c, m)

        fpc = _fp_coeff(fuzzy_labels)
        statistics.add_elements([fpc, Jjm/data.shape[0]])
        p += 1
        
        # Stopping rule
        #if np.linalg.norm(fuzzy_labels - fuzzy_labels_copy) < error and p > 1:
        #    break
        if np.linalg.norm(centroids_copy - centroids) < error and p > 1:
            break
            
    # Final calculations
    error = np.linalg.norm(fuzzy_labels - fuzzy_labels_copy)
    fpc = _fp_coeff(fuzzy_labels)

    return centroids, fuzzy_labels, dist, p, fpc, statistics

#################################################################################

                        ##Algorytmty Predykcyjne##

#################################################################################

def incremental_fuzzy_cmeans_predict(test_data, cntr_trained, m, error, maxiter, metric='euclidean', init=None, seed=None):
    c = cntr_trained.shape[0]


    test_data = test_data.T
    # Setup u0
    fuzzy_labels = init
    if init is None:
        fuzzy_labels = create_labels(test_data, cntr_trained, metric, m)
    fuzzy_labels_start = fuzzy_labels
    # Initialize loop parameters
    jm = np.zeros(0)
    p = 0

    # Main cmeans loop
    while p < maxiter - 1:
        fuzzy_labels_copy = fuzzy_labels.copy()
        [fuzzy_labels, Jjm, d] = _cmeans_predict0(test_data, cntr_trained, fuzzy_labels_copy, c, m,metric)
        jm = np.hstack((jm, Jjm))
        p += 1

        # Stopping rule
        if np.linalg.norm(fuzzy_labels - fuzzy_labels_copy) < error:
            break

    # Final calculations
    error = np.linalg.norm(fuzzy_labels - fuzzy_labels_copy)
    fpc = _fp_coeff(fuzzy_labels)

    return fuzzy_labels, fuzzy_labels_start, d, jm, p, fpc 
    
def _cmeans_predict0(test_data, cntr, fuzzy_labels_copy, c, m, metric):

    # Normalizing, then eliminating any potential zero values.
    fuzzy_labels_copy = normalize_columns(fuzzy_labels_copy)
    fuzzy_labels_copy = np.fmax(fuzzy_labels_copy, np.finfo(np.float64).eps)

    fuzzy_labels_m = fuzzy_labels_copy ** m

    # For prediction, we do not recalculate cluster centers. The test_data is
    # forced to conform to the prior clustering.

    d = _distance(test_data, cntr, metric)
    d = np.fmax(d, np.finfo(np.float64).eps)

    jm = (fuzzy_labels_m * d ** 2).sum()

    fuzzy_labels = normalize_power_columns(d, - 2. / (m - 1))

    return fuzzy_labels, jm, d