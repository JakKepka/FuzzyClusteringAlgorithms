import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics.cluster import rand_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_auc_score, matthews_corrcoef, log_loss
from collections import Counter
from scipy import stats

from libraries.process_data import merge_chunks

#################################################################################

                        ##Glosowanie turowe##

#################################################################################


# Przypisuje clustry do klas
def assign_clusters_to_classes(fuzzy_labels, centroids, y, n_classes):
    # Zliczam pierwsze punkty do jakich klas należą, następnie dopiero patrzę na segmenty.
    cluster_membership = np.argmax(fuzzy_labels, axis=0)

    count_points = np.zeros((centroids.shape[0], n_classes))
    
    for i, label in enumerate(cluster_membership):
        count_points[label, y[i]] += 1

    # Zwracamy tablicę z przyporządkowanymi klasami dla każdego clustra.
    return np.argmax(count_points, axis=1)

# Przypisuje clasy do punktów
def assign_class_to_points(fuzzy_labels, cluster_to_class):
    cluster_membership = np.argmax(fuzzy_labels, axis=0)

    result = np.zeros(len(cluster_membership), dtype=fuzzy_labels.dtype)

    result[:] = cluster_to_class[cluster_membership]
    return result

# Klasyfikacja punktów
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
    
# Głosowanie większościowe, elimując najmniejpopularne klasy, zwraza ostateczną klase po głosowaniu
def majority_vote_with_elimination(class_vectors, n_classes):

    counter = 0
    mark_deletion = np.zeros(n_classes)
    while True:
        # Zliczanie pierwszych klas (najbardziej przynależnych) dla wszystkich punktów
        first_choices = [classes[0] for classes in class_vectors if classes.size > 0]
        class_counter = Counter(first_choices)

        # Sprawdzenie, czy mamy jedną dominującą klasę
        if len(class_counter) == 1:
            return first_choices  # Zwróć dominującą klasę
        
        # Znajdź najmniej popularną klasę (lub klasy, jeśli są remisowe)
        min_count = min(class_counter.values())
        least_common_classes = [cls for cls, count in class_counter.items() if count == min_count]

        # Dla każdej klasy do usunięcia
        for cls_to_remove in least_common_classes:
            mark_deletion[cls_to_remove] = 1
            for i, classes in enumerate(class_vectors):
                # Jeśli pierwsza klasa jest tą do usunięcia, usuń ją
                if classes.size > 0 and mark_deletion[classes[0]] == 1:
                    class_vectors[i] = np.delete(classes, 0)

        # Sprawdź, czy wszystkie wektory klas zostały wyeliminowane
        if all(classes.size == 1 for classes in class_vectors):
            return first_choices  # Zwróć None, jeśli wszystkie klasy zostały wyeliminowane

        if counter >= 3:
            # Zwróć pierwszą klasę, która pozostała na końcu eliminacji
            return first_choices
        
        counter += 1

# Zwraca listę sklasyfikowanych segmentów
def classify_segment(val_matrix, prototype_to_class, n_classes, centroids):

    n_val = val_matrix.shape[1]
    
    classified_labels = []
    
    for i in range(n_val):
        val_series = val_matrix[:, i]
        v_expanded = val_series[:, np.newaxis]  # Kształt: (8, 1)

        sorted_prototypes = np.argsort(val_series)[::-1]
        class_to_max_prototype = np.zeros(n_classes)

        for prototype_idx in sorted_prototypes:
                # Mapuj prototyp na odpowiednią klasę
                mapped_class = prototype_to_class[prototype_idx]
                
                # Jeśli klasa nie była jeszcze dodana lub obecny prototyp ma większą przynależność, zaktualizuj
                if class_to_max_prototype[mapped_class] == 0 or class_to_max_prototype[mapped_class] < val_series[prototype_idx]:
                    class_to_max_prototype[mapped_class] = val_series[prototype_idx]
                else:
                    break  # Ponieważ sortowanie jest malejące, dalsze prototypy będą miały mniejszą przynależność
        
        sorted_class_indices = np.argsort(class_to_max_prototype)[::-1]  
        classified_labels.append(sorted_class_indices)
    
    # Przeprowadź głosowanie większościowe z eliminacją
    final_class = majority_vote_with_elimination(classified_labels, n_classes)
    
    return final_class


def classify_points_knn_eliminate_minor_class(centroids, n_classes, validation_x_chunked, predict_data, clusters_for_each_class = None):
    max_cluster = len(centroids)
    
    cluster_to_class = np.full(max_cluster, -1)  # Inicjalizujemy wartości np. -1 dla niezdefiniowanych
    
    for class_idx, cluster_range in clusters_for_each_class.items():
        for cluster in cluster_range:
            cluster_to_class[cluster] = class_idx

    validation_classified = None
    validation_classified_chunks_majority = []
    itr = 0
    
    for chunk in validation_x_chunked:
        _, fuzzy_labels_chunk, _ = predict_data(chunk, centroids)
        chunk_classified = classify_segment(fuzzy_labels_chunk, cluster_to_class, n_classes, centroids)  
        
        if chunk_classified is not None:
            mode_value, count = stats.mode(chunk_classified)
            majority = np.full(len(chunk), mode_value)
            validation_classified_chunks_majority.append(majority)
            
    validation_classified = np.concatenate(validation_classified_chunks_majority[:])  

    return validation_classified, cluster_to_class


#################################################################################

                            ##Calculate statistics##

#################################################################################


# Oblicza statystki
def calculate_statistics(y_true, y_pred, y_proba=None):
    stats = {}

    # Accuracy
    stats['Accuracy'] = accuracy_score(y_true, y_pred)

    # Precision
    stats['Precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)

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

def clusters_list_to_set(clusters_to_class):
    clusters_for_each_class = {}
    
    # Iteracja przez listę clusters_to_class
    for cluster_index, class_id in enumerate(clusters_to_class):
        # Jeśli klasa nie istnieje w słowniku, dodaj ją
        if class_id not in clusters_for_each_class:
            clusters_for_each_class[class_id] = []
        
        # Dodaj indeks klastra do odpowiedniej klasy
        clusters_for_each_class[class_id].append(cluster_index)
    
    return clusters_for_each_class

#################################################################################

                            ##Exam segments##

#################################################################################


# Dla każdego segmentu podaje najczęściej występujący cluster, w sesnsie liczości punktów
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

# Dla każdego segmentu podaje najczęściej występujący cluster w sensie sumy prawdopodobieństw
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

# Znajduje najcześciej występujący element z listy
def find_most_common(lst):
    # Tworzymy słownik do zliczania wystąpień liczb
    counts = {}
    
    # Zliczanie wystąpień każdej liczby
    for num in lst:
        if num in counts:
            counts[num] += 1
        else:
            counts[num] = 1
    
    # Znalezienie liczby z maksymalną liczbą wystąpień
    most_common = max(counts, key=counts.get)
    
    return most_common

# Zwraca najczęściej wystepujący element dla każdego chunk'u
def get_label_of_segment_knn(chunks, cluster_membership):

    most_common_elements = []
    start = 0

    # Przetwarzanie każdego segmentu
    for chunk in chunks:
        # Rozmiar chunku
        chunk_size = len(chunk)
        
        # Wydobywamy odpowiedni segment z cluster_membership
        segment = cluster_membership[start:start + chunk_size]

        # Znajdujemy najczęściej występujący element w segmencie
        most_common_element = find_most_common(segment)

        # Dodajemy wynik do listy
        most_common_elements.append(most_common_element)
        
        # Przesuwamy początek dla następnego segmentu
        start += chunk_size
    
    return most_common_elements
    
#################################################################################

                            ##Exam segments##

#################################################################################

# Oblicza statystyki dla zklasyfikowanych segmentów. W pierwszym przypadku (warunku if) oblicza wyniki przy założeniu że clustry przypisane są tak jak w cluster_for_each_class (metoda indukcyjna), drugi przypadek sam wyznacza przyporządkowanie clustrów na podstawie danych. Trzeciy warunek nie korzysta z metody głosowania turowego.
def segment_statistics(clusters_for_each_class, auto_classify_segments, centroids, n_classes, chunks, chunks_y, predict_data_algorithm, fuzzy_labels):
    
    if(clusters_for_each_class is not None and auto_classify_segments == False):
    # Korzystamy z przypisanych clustrów w wersji indukcyjnej.
        # Głosowanie większościowe
        validation_y_predicted, cluster_to_class = classify_points_knn_eliminate_minor_class(centroids, n_classes, chunks, predict_data_algorithm, clusters_for_each_class = clusters_for_each_class)
    
        # Klasyfikujemy segmenty
        statistics = calculate_statistics(np.concatenate(chunks_y[:]), validation_y_predicted)  
    elif(auto_classify_segments == True):
    # Korzystamy z przypisanych clustrów w wersji empirycznej.
        # Statystki dla klasyfikacji segmentów
        statistics, cluster_to_class = validate_segments(chunks, chunks_y, centroids, fuzzy_labels)

        clusters_for_each_class = clusters_list_to_set(cluster_to_class)
        
        # Głosowanie większościowe
        validation_y_predicted, cluster_to_class = classify_points_knn_eliminate_minor_class(centroids, n_classes, chunks, predict_data_algorithm, clusters_for_each_class = clusters_for_each_class)
    
        # Klasyfikujemy segmenty
        statistics = calculate_statistics(np.concatenate(chunks_y[:]), validation_y_predicted)  
    else:
    # Implementacja clustrów, bez głosowania większościwoego.
        # Statystki dla klasyfikacji segmentów
        statistics, cluster_to_class = validate_segments(chunks, chunks_y, centroids, fuzzy_labels)

    return statistics, cluster_to_class
    
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


# Liczy statystki, przy ocenianiu/klasyfikiowaniu każdego punktu.
def validate_labels(chunks, chunks_y, centroids, fuzzy_labels):

    # Mergujemy chunki w dataset
    data, y_true = merge_chunks(chunks, chunks_y)
    
    # Przydzielenie punktów do danych klustrów. Przydzielamy cluster najczesciej wystepujacy u sasiadow
    y_pred = np.argmax(fuzzy_labels, axis=0)

    # Przyporządkujemy clustry do klas na podstawie danych treningowych.
    cluster_to_class = assign_clusters_to_classes_count_single_points(y_pred, centroids, y_true)

    y_pred = [cluster_to_class[x] for x in y_pred]
    
    return calculate_statistics(y_true, y_pred)

# Liczy statystki, przy ocenianiu/klasyfikiowaniu każdego punktu.
def validate_labels_knn(chunks, chunks_y, y_pred):

    # Mergujemy chunki w dataset
    data, y_true = merge_chunks(chunks, chunks_y)
    
    return calculate_statistics(y_true, y_pred)

# Łączy powyższe 2 funkcje. Zwraca klasy dla segmentów
def validate_segments_knn(chunks, chunks_y, cluster_membership):

    # Mergujemy chunki w dataset
    data, y = merge_chunks(chunks, chunks_y)
    
    # Klasy segmentów
    labels = [chunk_y[0] for chunk_y in chunks_y]

    # Labele dla segmentów
    segment_labels = get_label_of_segment_knn(chunks, cluster_membership)
    
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
    
    return calculate_statistics(labels, segment_labels)
