import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
import random
from tqdm import tqdm
import numpy as np

#################################################################################

                            ##Dimentional Reduction Plots##

#################################################################################


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
    print(n_clusters)
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
    print(n_clusters)

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
    
    print('Historia danych wewnatrz iteracji dla pierwszego chunka')
    diagnosis_iterations[0].plot_lists('Historia danych wewnatrz iteracji dla pierwszego chunka')
    
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
def visualise_labeled_data_all_dimensions(data, y, n_classes):
    # Create a 2x3 grid of subplots
    c = data.shape[1]
    fig, axs = plt.subplots(int(c/3) + 1, 3, figsize=(15, 10))
    
    # Ustalamy unikalne etykiety
    unique_labels = np.unique(y)
    
    # Wybieramy kolory dla każdej klasy
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    for j in range(n_classes):
        for i, label in enumerate(unique_labels):
            # Wybieranie punktów dla danej klasy
            ix = np.array(np.where(label == y)).reshape(-1)
            mask = (y == label)
            axs[int(j/3), j%3].plot(ix, data[mask, 0], color=colors[j], label=f'Klasa {label}', alpha=0.5)
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
        ax.bar(x + i * width, results[:, i], width, label=model)
    
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


