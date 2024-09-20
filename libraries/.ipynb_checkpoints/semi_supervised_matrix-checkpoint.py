import numpy as np

# Metoda tworząca z wektora labelów tablicę labelów potrzebną do implementacji algorytmu SSFCM.
def label_vector_to_semi_supervised_matrix(y, n_clusters, clusters_for_each_class,  procent_of_data=0.5):
    array = np.zeros((len(y), n_clusters))

    for i, label in enumerate(y):
        print('label',label)
        if(i >= len(y)*procent_of_data):
            break

        injection_power = 1 / len(clusters_for_each_class[label]) - 0.05
        for element in clusters_for_each_class[label]:
            array[i, element] = injection_power
            
    return array

# Tworzy macierz do uczenia nadzorowanego. Przydziela kilka klastrów do danej klasy.
# Injection to procent danych jakie labelujemy
# Zwraca punkty stworzone przez funkcje generate_clusters_proportional.
def create_semi_supervised_matrix(X, y, n_clusters, injection=1.0):

    average_classes = average_by_class(X, y)

    init_centroids, class_of_centroid, clusters_for_each_class = generate_clusters_proportional(average_classes, n_clusters, deviation=0.1)

    y_matrix = label_vector_to_semi_supervised_matrix(y_train, n_clusters, clusters_for_each_class, injection)

    return y_matrix, init_centroids, clusters_for_each_class

def upload_semi_supervised_matrix(y, new_cluster_id, clusters_for_each_class, n_clusters, injection):

    idx = len(clusters_for_each_class)
    
    for i in range(len(clusters_for_each_class)):
        k = clusters_for_each_class[i].stop 
        n = clusters_for_each_class[i].start
        if new_cluster_id in clusters_for_each_class[i]:
            clusters_for_each_class[i] = range(n, k+1)
            idx = i
        if i > idx:
            clusters_for_each_class[i] = range(n+1,k+1)
    
    y_matrix = label_vector_to_semi_supervised_matrix(y, n_clusters, clusters_for_each_class, injection)
    
    return y_matrix, clusters_for_each_class

