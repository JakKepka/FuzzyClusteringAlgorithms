import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
import random
from tqdm import tqdm
import numpy as np
import math
import itertools

#################################################################################

                            ##Dimentional Reduction Plots##

#################################################################################

def plot_metrics_by_algorithm(output_list, output_list_name):
    
    # Zbieramy metryki, które będziemy analizować
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    num_metrics = len(metric_names)
    num_datasets = len(output_list)

    # Przygotowujemy dane do wykresów
    all_keys = list(output_list[0].keys())  # Zakładamy, że każdy dataset ma te same klucze
    num_keys = len(all_keys)

    # Tworzymy wykres dla każdego klucza (np. "cluster1", "cluster2")
    for key in all_keys:
        fig, ax = plt.subplots(figsize=(len(output_list_name)*6, 6))

        # Zbieramy dane dla każdego datasetu i danego klucza
        labels = []
        for i, dataset_name in enumerate(output_list_name):
            data_dict = output_list[i]
            stats = data_dict[key][3]  # Pobieramy statystyki dla danego klucza
            
            # Zbieramy metryki dla tego klucza i datasetu
            values = [stats.get(metric, 0) for metric in metric_names]
            labels.append(f"{dataset_name}")
            
            # Rysowanie słupków dla tego datasetu
            x = np.arange(num_metrics) + i * 0.2  # Przesunięcie słupków
            bars = ax.bar(x, values, width=0.2, label=dataset_name)
            
            # Dodanie wartości liczbowych nad słupkami
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}', ha='center', va='bottom')

        # Konfiguracja osi i tytułów
        ax.set_xticks(np.arange(num_metrics) + (num_datasets - 1) * 0.1)  # Środek grupy słupków
        ax.set_xticklabels(metric_names)
        ax.set_ylabel('Wartości Metryk')
        ax.set_title(f'Metryki dla klucza: {key}')
        ax.legend(title="Datasety", loc="upper left", bbox_to_anchor=(1.05, 1))
        
        plt.tight_layout()
        plt.show()


def plot_metrics_by_metrics(output_list, output_list_name):
    # Zbieramy statystyki dla każdego słownika
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    num_metrics = len(metric_names)
    num_datasets = len(output_list)
    
    # Przygotowanie struktury danych do zbierania statystyk
    metrics_per_key = {name: [] for name in metric_names}
    labels = []

    # Iterujemy po każdym elemencie słownika
    all_keys = list(output_list[0].keys())  # Zakładamy, że każdy dataset ma te same klucze
    for key in all_keys:
        labels.append(f"{key}")  # Etykiety dla każdego elementu (klucza)
        
        # Zbieramy statystyki dla każdego datasetu i danego klucza
        for metric in metric_names:
            stats_for_metric = []
            for i, data_dict in enumerate(output_list):
                # Pobierz statystyki (czwarty element listy)
                stats = data_dict[key][3]
                stats_for_metric.append(stats.get(metric, 0))  # Domyślnie 0, jeśli brak metryki
            metrics_per_key[metric].append(stats_for_metric)
    
    # Tworzenie wykresów dla każdej z metryk
    x = np.arange(len(all_keys))  # Oś X (indeksy dla kluczy)
    width = 0.15  # Szerokość słupków

    fig, axs = plt.subplots(1, num_metrics, figsize=(num_metrics * len(output_list_name) * 6, 6), sharey=True)
    
    # Rysowanie słupków dla każdej metryki
    for i, metric in enumerate(metric_names):
        ax = axs[i]
        for j, dataset_name in enumerate(output_list_name):
            offsets = (j - num_datasets / 2) * width  # Przesunięcie dla każdego datasetu
            data_to_plot = [metrics_per_key[metric][k][j] for k in range(len(all_keys))]
            bars = ax.bar(x + offsets, data_to_plot, width, label=dataset_name)
            
            # Dodanie wartości liczbowych nad słupkami
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}', ha='center', va='bottom')

        # Konfiguracja osi i tytułów
        ax.set_xlabel('Elementy (klucze)')
        ax.set_title(metric)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")

    # Dodanie legendy i konfiguracja ogólna
    axs[0].set_ylabel('Wartości Metryk')
    axs[0].legend(output_list_name, title="Datasety", loc="upper left", bbox_to_anchor=(1.05, 1))
    
    plt.tight_layout()
    plt.show()
    
def plot_metrics_by_dataset(output_list, output_list_name):
    # Zbieramy statystyki dla każdego słownika
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

    # Tworzenie wykresów obok siebie (jeden subplot na dataset)
    n_datasets = len(output_list)
    fig, axes = plt.subplots(1, n_datasets, figsize=(6 * n_datasets, 6))  # 6 jednostek szerokości na każdy subplot
    
    if n_datasets == 1:
        axes = [axes]  # Dla przypadku, gdy mamy tylko jeden dataset, musimy zamienić to na listę.

    # Iterujemy po każdym słowniku
    for i, data_dict in enumerate(output_list):
        ax = axes[i]  # Pobieramy odpowiedni subplot
        name = output_list_name[i]
        metrics = {name: [] for name in metric_names}
        labels = []

        for key, value in data_dict.items():
            # Pobierz statystyki (czwarty element listy)
            stats = value[3]
            labels.append(key)  # Nazwy do osi X dla danego datasetu
            
            # Zapisz metryki
            for metric in metric_names:
                metrics[metric].append(stats.get(metric, 0))  # Domyślna wartość to 0, jeśli brak metryki

        # Tworzenie wykresów dla każdej z metryk
        x = np.arange(len(labels))  # Oś X (indeksy dla kluczy)
        width = 0.2  # Szerokość słupków
        
        # Przesunięcia dla każdej z metryk, aby słupki się nie nakładały
        offsets = np.linspace(-width, width, len(metric_names))

        # Rysowanie słupków dla każdej metryki
        for j, metric in enumerate(metric_names):
            bars = ax.bar(x + offsets[j], metrics[metric], width, label=metric)
            
            # Dodawanie wartości nad słupkami
            for bar in bars:
                yval = bar.get_height()  # Pobierz wysokość (wartość) słupka
                ax.text(
                    bar.get_x() + bar.get_width() / 2,  # Pozycja X (środek słupka)
                    yval,  # Pozycja Y (na górze słupka)
                    f'{yval:.2f}',  # Tekst do wyświetlenia
                    ha='center', va='bottom'  # Wyrównanie tekstu
                )

        # Konfiguracja wykresu dla danego datasetu
        ax.set_xlabel('Klucze')
        ax.set_ylabel('Wartości Metryk')
        ax.set_title(f'Metryki dla {name}')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.legend()

    plt.tight_layout()
    plt.show()

def plot_centroids_from_dict(data_dict):

    colors = itertools.cycle(plt.cm.tab10.colors)  # Użyj różnych kolorów dla każdego zestawu centroidów

    plt.figure()
    plt.title("Centroidy dla różnych klastrów (z PCA dla wymiarów > 2)")
    
    for name, values in data_dict.items():
        centroids = np.array(values[2])  # Pobierz centroidy (trzeci element listy)
        n_clusters, dim = centroids.shape

        # Redukcja PCA dla wymiarów > 2
        if dim > 2:
            pca = PCA(n_components=2)
            centroids_2d = pca.fit_transform(centroids)
        else:
            centroids_2d = centroids  # Jeśli wymiary są 2D, nie trzeba redukować

        # Rysuj centroidy na tym samym wykresie, z unikalnym kolorem
        color = next(colors)
        plt.scatter(centroids_2d[:, 0], centroids_2d[:, 1], c=[color], label=name, marker='o', alpha=0.5)
    
    plt.xlabel('Wymiar 1 (lub PCA komponent 1)')
    plt.ylabel('Wymiar 2 (lub PCA komponent 2)')
    plt.legend()
    plt.grid(True)
    plt.show()

def simple_plot(X, cntr, cluster_labels, name):
    n_clusters = cntr.shape[0]
    if X.shape[1] > 2:
        # Jeżeli dane mają więcej niż 2 wymiary, to można redukować ich wymiarowość. 
        # Zwracamy zatem True, aby użyć algorytmu redukującego wymiarowość
        return True
    
    elif X.shape[1] == 2:
        # Tworzymy wykres dla danych 2 wymiarowych
        # Wizualizacja klastrów
        plt.figure(figsize=(10, 8))
        for i in range(n_clusters):
            plt.scatter(X[cluster_labels == i, 0], X[cluster_labels == i, 1], label=f'Cluster {i+1}')
    
        # Dodanie centrów klastrów do wykresu
        plt.title('Fuzzy C-Means Clustering 2D ' + name)
        plt.scatter(cntr[:, 0], cntr[:, 1], marker='x', s=200, c='black', label='Cluster Centers')
    
    elif X.shape[1] == 1:
        #Losujemy liste kolorow
        # Użycie palety ciągłej 'viridis'
        # Tworzenie palety z 100 kolorami
        palette = sns.color_palette("husl", 100)
        
        # Konwertowanie do listy RGB
        colors_list = [color for color in palette]
              
        # Tworzymy wykres dla danych jednowymiarowych
        plt.figure(figsize=(10, 8))
        plt.title('Fuzzy C-Means Clustering 1D ' + name)
        for i in range(n_clusters):       
            plt.plot(np.array(np.where(cluster_labels == i)).reshape(-1), X[cluster_labels == i, 0], label=f'Cluster {i+1}', color=colors_list[i], marker='o')
    else:
        print('Zły wymiar danych, plot function')
        return False
        
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.show()
    return False

def plot_pca(X, cntr, fuzzy_labels):
    cluster_labels = np.argmax(fuzzy_labels, axis=0)
    # Sprawdzamy czy można redukować wymiar, czy wystarczy narysować wykres bez zmian
    data_type = simple_plot(X, cntr, cluster_labels, 'pca')

    n_clusters = cntr.shape[0]

    # Redukcja wymiarowości za pomocą PCA do 2 wymiarów
    if data_type:
        pca = PCA(n_components=2)
        data_pca = pca.fit_transform(X)
    
        # Redukcja wymiarowości centrów klastrów
        cntr_pca = pca.transform(cntr)
        
        # Wizualizacja klastrów
        plt.figure(figsize=(10, 8))
        for i in range(n_clusters):
            plt.scatter(data_pca[cluster_labels == i, 0], data_pca[cluster_labels == i, 1], label=f'Cluster {i+1}')
    
        # Dodanie centrów klastrów do wykresu
        plt.title('Fuzzy C-Means Clustering (PCA Reduced Data)')
        plt.scatter(cntr_pca[:, 0], cntr_pca[:, 1], marker='x', s=200, c='black', label='PCA Cluster Centers')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend()
        plt.show()

def plot_pca_cluster(X, cntr, fuzzy_labels, cluster_to_class, y_labels=None):
    # Wyznaczamy etykiety klastrów na podstawie największej wartości z macierzy przynależności
    cluster_labels = np.argmax(fuzzy_labels, axis=0)
    
    if y_labels is not None:
        cluster_labels = y_labels
    # Sprawdzamy czy można redukować wymiar (funkcja simple_plot musi istnieć)
    data_type = simple_plot(X, cntr, cluster_labels, 'pca')

    n_clusters = cntr.shape[0]

    # Redukcja wymiarowości za pomocą PCA do 2 wymiarów
    if data_type:
        pca = PCA(n_components=2)
        data_pca = pca.fit_transform(X)
    
        # Redukcja wymiarowości centrów klastrów
        cntr_pca = pca.transform(cntr)
        
        # Lista kolorów dla klas
        colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan', 'magenta']
        
        # Wizualizacja klastrów
        plt.figure(figsize=(10, 8))
        
        # Rysowanie punktów danych (bez zmiany kolorowania)
        for i in range(n_clusters):
            plt.scatter(data_pca[cluster_labels == i, 0], data_pca[cluster_labels == i, 1], label=f'Cluster {i+1}')
    
        # Rysowanie centroidów z kolorowaniem na podstawie cluster_to_class
        for i in range(n_clusters):
            # Klasa przypisana do danego klastru
            class_label = cluster_to_class[i]
            
            # Kolor dla klasy centroidu
            color = colors[class_label % len(colors)]
            
            # Rysowanie centroidu
            plt.scatter(cntr_pca[i, 0], cntr_pca[i, 1], marker='x', s=200, c=color, label=f'Centroid (Class {class_label})')
        
        plt.title('Fuzzy C-Means Clustering (PCA Reduced Data)')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend()
        plt.show()

def custom_plot(X, cntr, validation_y, cluster_to_class, data):
    #cluster_labels = np.argmax(fuzzy_labels, axis=0)
    # Sprawdzamy czy można redukować wymiar, czy wystarczy narysować wykres bez zmian
    #data_type = simple_plot(X, cntr, cluster_labels, 'pca')

    n_clusters = cntr.shape[0]
    print(n_clusters)
    # Redukcja wymiarowości za pomocą PCA do 2 wymiarów
    if True:
        pca = PCA(n_components=2)
        pca.fit(data)
        data_pca = pca.transform(X)
        description = []
        scatter_plots = []

        # Redukcja wymiarowości centrów klastrów
        cntr_pca = pca.transform(cntr)
        fig, ax = plt.subplots(figsize=(10, 8))


        # for i in range(n_clusters):
        #     plt.scatter(data_pca[cluster_labels == i, 0], data_pca[cluster_labels == i, 1], label=f'Cluster {i+1}')
        colors = plt.cm.get_cmap('viridis', 4)
        #print("fuzzy_labels, ", fuzzy_labels)

        # for i in range(validation_y.shape[0]):
        #     validation_y[i] = cluster_to_class[validation_y[i]]

        #print("klasy, ", validation_y)
        # Rysowanie danych dla każdej klasy osobno
        for class_idx in range(4):
            #print("---------------klasa-----------", class_idx)
            class_mask = validation_y == class_idx
            #for i in range(fuzzy_labels.shape[1]):
            #    if class_mask[i]:  # Sprawdzamy, czy maska dla tego indeksu jest prawdziwa
                    #for value in fuzzy_labels_val[:, i]:
                    #description.append(fuzzy_labels[:, i])
                    #print("labele: ",fuzzy_labels[:, i] )
                    #print("klasa: ", cluster_to_class[np.argmax(fuzzy_labels[:, i])])

    
            scatter = ax.scatter(data_pca[class_mask, 0], data_pca[class_mask, 1], 
                       color=colors(class_idx), label=f'Klasa {class_idx}')
            scatter_plots.append(scatter)
    
        

        if cluster_to_class is not None:
            for idx, (centroid, class_idx) in enumerate(zip(cntr_pca, cluster_to_class)):
                ax.scatter(centroid[0], centroid[1], 
                           color=colors(class_idx), edgecolor='black', 
                           marker='o', s=200, linewidths=2, label=f'Centroid {idx} (Klasa {class_idx})')
        # def hover_label(sel):
        #     text = ''
        #     #for i in description[sel.index]:
        #     #print(description[sel.index])
        #     k = 0
        #     for i in description[sel.index]:
        #         text += (f"u do centorida {k} wynosi: {i:.3f}\n")
        #         k += 1
        #     sel.annotation.set_text(text)
           
        # mplcursors.cursor(scatter_plots, hover=True).connect("add", hover_label)
        # Dodanie centrów klastrów do wykresu
        plt.title('Fuzzy C-Means Clustering (PCA Reduced Data)')
        ax.scatter(cntr_pca[:, 0], cntr_pca[:, 1], marker='x', s=200, c='black', label='PCA Cluster Centers')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend()
        plt.show()

def plot_pca_standard(X, y):
    # Tworzymy obiekt PCA i redukujemy do 2 wymiarów
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # Tworzymy wykres punktów, kolorując je zgodnie z klasami y
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', s=100, edgecolor='k')
    
    # Dodajemy kolorowy pasek legendy
    legend = plt.legend(*scatter.legend_elements(), title="Klasy")
    plt.gca().add_artist(legend)
    
    # Dodajemy tytuł i etykiety osi
    plt.title("Wizualizacja danych za pomocą PCA")
    plt.xlabel("Główna składowa 1")
    plt.ylabel("Główna składowa 2")
    
    # Wyświetlamy wykres
    plt.show()

    
def plot_heatmap(data, centroids, fuzzy_labels):   
    cluster_labels = np.argmax(fuzzy_labels, axis=0)
    # Sprawdzamy czy można redukować wymiar, czy wystarczy narysować wykres bez zmian
    data_type = simple_plot(data, centroids, cluster_labels, 'heatmap')

    if(data_type):
        combined_data = np.vstack([data, centroids])
        combined_labels = np.hstack([cluster_labels, [-1] * len(centroids)])  # -1 dla centroidów
    
        plt.figure(figsize=(12, 8))
        sns.heatmap(combined_data, cmap='coolwarm', xticklabels=False, yticklabels=False)
        plt.title('Heatmap of Data and Centroids')
        plt.show()

def plot_tsne(data, centroids, fuzzy_labels):
    cluster_labels = np.argmax(fuzzy_labels, axis=0)
    # Sprawdzamy czy można redukować wymiar, czy wystarczy narysować wykres bez zmian
    data_type = simple_plot(data, centroids, cluster_labels, 'tsne')
    if(data_type):
        combined_data = np.vstack([data, centroids])
        tsne = TSNE(n_components=2, random_state=42)
        tsne_results = tsne.fit_transform(combined_data)
    
        plt.figure(figsize=(10, 6))
        plt.scatter(tsne_results[:-len(centroids), 0], tsne_results[:-len(centroids), 1], c=cluster_labels, cmap='viridis', label='Data')
        plt.scatter(tsne_results[-len(centroids):, 0], tsne_results[-len(centroids):, 1], c='black', label='Centroids', marker='x', s=200)
        plt.legend()
        plt.colorbar()
        plt.title('t-SNE Plot of Data and Centroids')
        plt.show()


def plot_mds(data, centroids, fuzzy_labels):
    cluster_labels = np.argmax(fuzzy_labels, axis=0)
    # Sprawdzamy czy można redukować wymiar, czy wystarczy narysować wykres bez zmian
    data_type = simple_plot(data, centroids, cluster_labels, 'mds')

    if(data_type):
        combined_data = np.vstack([data, centroids])
        mds = MDS(n_components=2, random_state=42)
        mds_results = mds.fit_transform(combined_data)
    
        plt.figure(figsize=(10, 6))
        plt.scatter(mds_results[:-len(centroids), 0], mds_results[:-len(centroids), 1], c=cluster_labels, cmap='viridis', label='Data')
        plt.scatter(mds_results[-len(centroids):, 0], mds_results[-len(centroids):, 1], c='black', label='Centroids', marker='x', s=200)
        plt.legend()
        plt.colorbar()
        plt.title('MDS Plot of Data and Centroids')
        plt.show()

#################################################################################

                            ##Overview Plots##

#################################################################################

def plot_centroids(centroids):
    # Przeprowadzenie PCA
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(centroids)

    colors_list = list(plt.cm.tab10.colors)
    
    # Wykres PCA
    plt.figure(figsize=(10, 5))
    plt.scatter(data_pca[:, 0], data_pca[:, 1], c = colors_list[:data_pca.shape[0]], cmap='viridis')
    plt.colorbar(label='Czas')
    plt.xlabel('Główna składowa 1')
    plt.ylabel('Główna składowa 2')
    plt.title('Wizualizacja centroidów startowych PCA')
    plt.grid(True)
    plt.show()

def plot_two(data, centroids, cluster_labels, plot1, plot2):
    plot1(data, cntr, cluster_labels)
    plot2(data, cntr, cluster_labels)

def visualize_all(data, centroids, fuzzy_labels):
    # Wizualizacja dla pierwszej iteracji
    print('PCA plot')
    plot_pca(data, centroids, fuzzy_labels)
    print('heatmap plot')
    plot_heatmap(data, centroids, fuzzy_labels)
    print('TSNE plot')
    plot_tsne(data, centroids, fuzzy_labels)
    print('MDS plot')
    plot_mds(data, centroids, fuzzy_labels)

def prepare_diagnosis_data_for_plotting(diagnosis_chunk, diagnosis_iterations):
    chunk_lists = diagnosis_chunk.get_lists()
    
    lists_amount =len(diagnosis_iterations[0].get_lists())
    
    iter_lists = []
    iter_lists_concatenate = []
    
    for i in range(0,lists_amount):
        iter_lists.append([])
        iter_lists_concatenate.append([])
        
    for di in diagnosis_iterations:
        lists = di.get_lists()
        for i, list_ in enumerate(lists):
            iter_lists[i].append(list_)
            iter_lists_concatenate[i] = iter_lists_concatenate[i] + list_
    return chunk_lists, iter_lists, iter_lists_concatenate
                        
# Funkcja wyświetla historię rozwoju algorytmu IFCM (i jego odmian)
def overview_plot(diagnosis_chunk, diagnosis_iterations, n_centroids_history=5):
    print('Historia danych ze względu na kolejne chunki')
    diagnosis_chunk.plot_lists('Historia danych ze względu na kolejne chunki')
    
    print('Historia statystyk ze względu na kolejne chunki')
    diagnosis_chunk.plot_statistics('Historia danych ze względu na kolejne chunki')
    
    print('Historia danych ze względu na kolejne chunki + historia rozwoju wewnątrz oblczeń dla pojedyńczego chunku')
    chunk_lists, iter_lists, iter_lists_concatenate = prepare_diagnosis_data_for_plotting(diagnosis_chunk, diagnosis_iterations)
    plot_lists_inside_lists(chunk_lists, iter_lists, 'Historia danych ze względu na kolejne chunki + historia rozwoju wewnątrz oblczeń dla pojedyńczego chunku')
    
    print('Historia fpc dla wszystkich chunków, i rozwojem wewnatrz chunku')
    plot_multiple_functions([(iter_lists_concatenate[0],'fpc')], 'Historia fpc dla wszystkich chunków, i rozwojem wewnatrz chunku')
    
    print('Historia centroidów ze względu na chunki')
    diagnosis_chunk.plot_centroid_history(n_centroids_history)
    
    print('fpc last', diagnosis_chunk.fpc_data[-1])
    print('rand last', diagnosis_chunk.rand_data[-1])

# Funkcje wyświetla wiele funkcji na jednym wykresie. 
# Input: args to lista par (lista, nazwa)
def plot_multiple_functions(args, title='multiple functions'):

    plt.figure(figsize=(10, 6))
    
    for y_values, label in args:
        x_values = list(range(len(y_values)))  # Zakładamy, że x to indeksy listy y
        plt.plot(x_values, y_values, label=label)
    
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

# Funkcja do tworzenia wykresu
def plot_lists_inside_lists(first_class_lists, second_class_lists, title='Plot of First Class and Second Class Lists'):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Wykres list pierwszej klasy
    for i, first_list in enumerate(first_class_lists):
        ax.plot(first_list, marker='o')
        
        # Wykresy list drugiej klasy dla każdej listy pierwszej klasy

    for i in range(len(second_class_lists)):
        for j in range(len(first_class_lists[0]) - 1):
            second_list = second_class_lists[i][j]
            x_vals = np.linspace(j, j+1, len(second_list))
            y_vals = np.array(second_list)

            ax.plot(x_vals, y_vals)
    
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title(title)
    plt.show()


#################################################################################

                            ##Plot labeled data##

#################################################################################


# Visualise Loaded Data
def visualise_labeled_data_all_dimensions(data, y, dim):
    # Create a 2x3 grid of subplots
    c = data.shape[1]
    fig, axs = plt.subplots(int(math.ceil(dim/3)) , 3, figsize=(15, 10))
    
    # Ustalamy unikalne etykiety
    unique_labels = np.unique(y)
    
    # Wybieramy kolory dla każdej klasy
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    for j in range(dim):
        for i, label in enumerate(unique_labels):
            # Wybieranie punktów dla danej klasy
            ix = np.array(np.where(label == y)).reshape(-1)
            mask = (y == label)
            axs[int(j/3), j%3].plot(ix, data[mask, j], color=colors[label], label=f'Klasa {label}', alpha=0.5)
            axs[int(j/3), j%3].set_title(f'param {j}')

#################################################################################

                            ##Compare fcm & knn##

#################################################################################


def create_set_for_stats(silhouette_avg, davies_bouldin_avg, rand, fpc, statistics):
    return {
        'Silhouette Score': silhouette_avg,
        #'Davies-Bouldin Score': davies_bouldin_avg,
        'Rand Score': rand,
        'Tested fpc': fpc,
        'Accuracy': statistics['Accuracy'],
        'Precision': statistics['Precision'],
        'Recall': statistics['Recall']
    }

def compare_models_statistics(statistics):
    # Wyciągamy wszystkie nazwy modeli
    models = list(statistics.keys())
    
    # Wyciągamy wszystkie nazwy metryk (przyjmujemy, że wszystkie modele mają te same metryki)
    metrics = list(statistics[models[0]].keys())
    
    # Tworzymy macierz wyników
    results = np.array([[statistics[model][metric] for model in models] for metric in metrics])
    
    # Ustawienia wykresu
    x = np.arange(len(metrics))  # Pozycje na osi X
    width = 0.15  # Szerokość słupków
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Tworzymy słupki dla każdego modelu
    for i, model in enumerate(models):
        bars = ax.bar(x + i * width, results[:, i], width, label=model)
        
        # Dodawanie wartości nad słupkami
        for bar in bars:
            yval = bar.get_height()  # Pobierz wysokość (wartość) słupka
            ax.text(
                bar.get_x() + bar.get_width() / 2,  # Pozycja X (środek słupka)
                yval,  # Pozycja Y (na górze słupka)
                f'{yval:.2f}',  # Tekst do wyświetlenia
                ha='center', va='bottom'  # Wyrównanie tekstu
            )
    
    # Dodajemy etykiety i tytuły
    ax.set_xlabel('Metryki')
    ax.set_ylabel('Wartość')
    ax.set_title('Porównanie statystyk różnych modeli')
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(metrics)
    ax.legend()
    
    # Wyświetlamy wykres
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()