from tqdm import tqdm
from IPython.display import clear_output
import time
import sys
import statistics
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA

# Rysowanie wykresów wielu funkcji na jednym rysunku
# Input:
#       args - lista krotek, gdzie każda krotka zawiera:
#              y_values - lista lub tablica wartości y do narysowania
#              label - etykieta dla danej funkcji
# Output:
#       Brak (funkcja wyświetla wykres)
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

class DiagnosisTools:
    # Klasa zawiera jako pola listy z statystkami zebranymi przy każdej iteracji fuzzy cmeans (iteracja - kolejny chunk)
    # Zbiera dane, liczy średnie i wartości skrajne, tworzy wykresy, wypisuje statystki.
    # Zawiera również historię rozwoju/zmian centroidów. Potrafi narysować jak centroidy zmieniały się z kolejnymi iteracjami.
    
    def __init__(self):
        # Inicjalizacja trzech pustych list
        self.silhouette_avg_data = []
        self.davies_bouldin_avg_data = []
        self.fpc_data = []
        self.rand_data = []
        
        # Centroidy
        self.centroids = []

# Funkcje dodające
    def add_elements(self, silhouette_avg, davies_bouldin_avg, fpc, rand):
        self.silhouette_avg_data.append(silhouette_avg) 
        self.davies_bouldin_avg_data.append(davies_bouldin_avg)
        self.fpc_data.append(fpc)
        self.rand_data.append(rand)
        
    def add_centroids(self, centroids):
        self.centroids.append(centroids)

# Funkcje rysujące wykresy
    def plot_lists(self,title='multiple plot'):
        plot_multiple_functions([(self.silhouette_avg_data, 'silhouette_avg'), (self.davies_bouldin_avg_data, 'davies_bouldin_avg'), (self.fpc_data, 'fpc'), (self.rand_data,'rand')], title)
    
    def plot_centroid_history(self, k):
        # Metoda pokazuje historię zmian centroidów z każdą iteracją/ z każdym dodanym chunkiem.
        
        # Redukcja wymiaru do 2 za pomocą PCA
        centroids = self.centroids[:k]

        # Łączenie wszystkich centroidów w jedną tablicę dla PCA
        all_centroids = np.vstack(centroids)
        
        # Redukcja wymiaru do 2 za pomocą PCA
        pca = PCA(n_components=2)
        reduced_centroids = pca.fit_transform(all_centroids)
        
        # Inicjalizacja wykresu
        plt.figure(figsize=(10, 8))
        
        # Kolorowanie punktów
        n_colors = centroids[0].shape[0]
        colors = plt.cm.get_cmap('tab10', n_colors)  # Wybieramy colormap z odpowiednią liczbą kolorów
        
        # Rysowanie centroidów dla każdej grupy
        start_idx = 0
        for i, centroids in enumerate(centroids):
            end_idx = start_idx + centroids.shape[0]
            reduced = reduced_centroids[start_idx:end_idx]
            
            for j, point in enumerate(reduced):
                plt.scatter(point[0], point[1], color=colors(j))
                plt.text(point[0], point[1], str(i+1), fontsize=12, ha='right')
            
            start_idx = end_idx
        
        plt.title('Centroidy zredukowane za pomocą PCA')
        plt.xlabel('Pierwsza składowa główna')
        plt.ylabel('Druga składowa główna')
        plt.grid(True)
        
        # Tworzenie legendy dla kolorów
        legend_labels = [f'Centroid {i+1}' for i in range(n_colors)]
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors(i), markersize=10) for i in range(n_colors)]
        plt.legend(handles, legend_labels, title='Legenda')
    
        plt.show()

# Gettery
    def get_lists(self):
        return self.silhouette_avg_data, self.davies_bouldin_avg_data, self.fpc_data, self.rand_data
        
    # Dostaje na wejściu funkcję której korzysta na każdej z list, następnie funkcja zwraca 1 wartość dla każdej listy.
    def get_func_statistic(self, func):
        sil = func(self.silhouette_avg_data)
        davies = func(self.davies_bouldin_avg_data)
        fpc = func(self.fpc_data)
        rand = func(self.rand_data)
        return sil, davies, fpc, rand
        
    def get_avg(self):
        return self.get_func_statistic(statistics.mean)
    def get_max(self):
        return self.get_func_statistic(max)
    def get_min(self):
        return self.get_func_statistic(min)

    def get_statistics(self):
        sil_avg, dav_avg, fpc_avg, rand_avg = self.get_avg()
        sil_min, dav_min, fpc_min, rand_min = self.get_min()
        sil_max, dav_max, fpc_max, rand_max = self.get_max()
        return np.array([[sil_avg, dav_avg, fpc_avg, rand_avg], [sil_min, dav_min, fpc_min, rand_min], [sil_max, dav_max, fpc_max, rand_max]])

# Funkcje wypisujace wartości
    def print_statistics(self):
        array = self.get_statistics()
        print(array)
        
    def __str__(self):
        return f'silhouette_avg_data: {self.silhouette_avg_data}\ndavies_bouldin_avg_data: {self.davies_bouldin_avg_data}\nfpc_data: {self.fpc_data}'


class Multilist:
    
    def __init__(self, names):
        # Inicjalizacja trzech pustych list
        self.lists = []
        self.names = names
        for i in range(len(names)):
            self.lists.append([])

# Funkcje dodające
    def add_elements(self, elements):
        for i, element in enumerate(elements):
            self.lists[i].append(element)

# Funkcje rysujące wykresy
    def plot_lists(self, title='multiple plot'):
        plot_multiple_functions([(self.lists[i], self.names[i]) for i in range(len(self.lists))], title)
    
    
# Gettery
    def get_lists(self):
        return self.lists
        
    # Dostaje na wejściu funkcję której korzysta na każdej z list, następnie funkcja zwraca 1 wartość dla każdej listy.
    def get_func_statistic(self, func):
        ans = []
        for list_ in self.lists:
            ans.append(func(list_))
        return ans
        
    def get_avg(self):
        return self.get_func_statistic(statistics.mean)
    def get_max(self):
        return self.get_func_statistic(max)
    def get_min(self):
        return self.get_func_statistic(min)

    def get_statistics(self):
        ans = []

        ans.append(self.get_avg())
        ans.append(self.get_min())
        ans.append(self.get_max())
        return np.array(ans)

# Funkcje wypisujace wartości
    def print_statistics(self):
        array = self.get_statistics()
        print(array)
        
    def __str__(self):
        return f'silhouette_avg_data: {self.silhouette_avg_data}\ndavies_bouldin_avg_data: {self.davies_bouldin_avg_data}\nfpc_data: {self.fpc_data}'
    