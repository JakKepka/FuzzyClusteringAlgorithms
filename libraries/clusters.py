import numpy as np

# Sprawdza które statysyki są korzystniejesze dla nas
def compare_clusters(old_statistics, new_statistics):

    if(old_statistics['Precision'] < new_statistics['Precision']):
        return True
    else:
        return False
        

# Zliczamy dla każdego clustra, liczbę wystąpień punktów należących do niego
def count_points_for_clusters(cluster_membership, n_clusters):
    # Zlicz występowanie każdej wartości (każdego klastra)
    counts = np.bincount(cluster_membership, minlength=n_clusters)

    return counts

def sum_probability_for_clusters(fuzzy_labels):
    summed_labels = np.sum(fuzzy_labels, axis=1)

    return summed_labels

# Zlicza popularność clustrów, ile punktów należy do clustra oraz sumę ich prawdopodobieństwa
def popularity_of_clusters(fuzzy_labels, n_clusters):

    cluster_membership = np.argmax(fuzzy_labels, axis=0)
    
    counts = count_points_for_clusters(cluster_membership, n_clusters)

    summed_labels = sum_probability_for_clusters(fuzzy_labels)
    
    # Iteracja przez każdy klaster
    for cluster in range(len(counts)):
        print(f"Cluster {cluster}: counts = {counts[cluster]}, summed_labels = {summed_labels[cluster]}, fcm per point {summed_labels[cluster]/counts[cluster]} ")

# Dla każdej klasy znajdź punkt średni.
def average_by_class(X, y):
    # Unikalne klasy w y
    classes = np.unique(y)
    
    # Słownik do przechowywania średnich dla każdej klasy
    class_averages = {}
    
    for cls in classes:
        # Wybieramy indeksy odpowiadające danej klasie
        indices = np.where(y == cls)
        
        # Wybieramy punkty z X odpowiadające danej klasie
        class_points = X[indices]
        
        # Obliczamy średnią dla danej klasy
        class_avg = np.mean(class_points, axis=0)
        
        # Dodajemy średnią do słownika
        class_averages[cls] = class_avg
    
    return class_averages

# Tworzy clustry na podstawie średnich punktów dla każdej klasy. Dla każdej klasy generuje kilka punktów z lekkim odchyleniem od średniego punktu dla danej klasy.
# Punkty wygenerowane sumują się do n_clusters.
def generate_clusters_proportional(average_points, n_clusters, deviation=0.1):
    # Liczba klas
    num_classes = len(average_points)
    
    # Początkowy przydział punktów do klas (podział równomierny)
    points_per_class = [n_clusters // num_classes] * num_classes
    
    # Jeśli n_clusters nie jest podzielne przez num_classes, rozdysponuj pozostałe punkty
    remainder = n_clusters % num_classes
    for i in range(remainder):
        points_per_class[i] += 1
    
    # Generowanie punktów dla każdej klasy
    generated_points = []

    # Dla każdego centroida przyporządkowana informacja o klasie
    class_of_centroid = []

    # Dla każdej klasy zwraca listę centroidów (jeden bądź wiele punktów)
    clusters_for_each_class = {}
    indicies_start = 0
    
    for i, (cls, avg_point) in enumerate(average_points.items()):
        # Liczba punktów do wygenerowania dla danej klasy
        points_count = points_per_class[i]
        
        # Tworzymy losowe odchylenie dla każdego wymiaru
        deviations = np.random.randn(points_count, avg_point.size) * deviation
        
        # Generujemy punkty z lekkim odchyleniem od średniej
        points = avg_point + deviations

        clusters_for_each_class[cls] = range(indicies_start,indicies_start + points_per_class[i])
        
        indicies_start += points_per_class[i]
        # Przechowujemy wygenerowane punkty w słowniku
        for point in points:
            generated_points.append(point)
            class_of_centroid.append(cls)
    
    return np.array(generated_points), np.array(class_of_centroid), clusters_for_each_class


# Metoda tworząca z wektora labelów tablicę labelów potrzebną do implementacji algorytmu SSFCM.
def label_vector_to_semi_supervised_matrix(y, n_clusters, clusters_for_each_class,  procent_of_data=0.5):
    array = np.zeros((len(y), n_clusters))

    for i, label in enumerate(y):
        
        if(i >= len(y)*procent_of_data):
            break

        injection_power = 1 / len(clusters_for_each_class[label]) - 0.05
        for element in clusters_for_each_class[label]:
            array[i, element] = injection_power
            
    return array

# Tworzy macierz do uczenia nadzorowanego. Przydziela kilka klastrów do danej klasy.
# Injection to procent danych jakie labelujemy
# Zwraca punkty stworzone przez funkcje generate_clusters_proportional.
def create_semi_supervised_matrix(X, y, n_clusters, injection=0.5):

    average_classes = average_by_class(X, y)

    init_centroids, class_of_centroid, clusters_for_each_class = generate_clusters_proportional(average_classes, n_clusters, deviation=1)

    y_matrix = label_vector_to_semi_supervised_matrix(y, n_clusters, clusters_for_each_class, injection)

    return y_matrix, init_centroids, clusters_for_each_class

