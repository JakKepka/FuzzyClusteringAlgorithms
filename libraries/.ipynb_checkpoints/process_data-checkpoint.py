import numpy as np
from tslearn.datasets import UCR_UEA_datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def knn_with_library(X_train, y_train, k=3):
    # Tworzymy model k-NN
    knn = KNeighborsClassifier(n_neighbors=k)
    
    # Trenujemy model na danych treningowych
    knn.fit(X_train, y_train)
    
    return knn
    
def select_subset(X_train, y_train, p):
    # Obliczamy rozmiar podzbioru (p% danych)
    subset_size = int(p * len(X_train))
    
    # Wybieramy p% danych z X_train i y_train z zachowaniem proporcji klas
    X_subset, _, y_subset, _ = train_test_split(X_train, y_train, 
                                                train_size=subset_size, 
                                                stratify=y_train, 
                                                random_state=42)
    return X_subset, y_subset

def stratify_data(X, y, percentage):
    # Usuwamy wiersze z NaN w X lub odpowiadające etykiety w y
    mask = ~np.isnan(X).any(axis=1)
    X_clean = X[mask]
    y_clean = y[mask]
    
    # Obliczamy ile próbek chcemy zachować
    strat_size = percentage
    
    # Używamy metody train_test_split z argumentem 'stratify', który zapewnia stratyfikację
    X_stratified, _, y_stratified, _ = train_test_split(X_clean, y_clean, train_size=strat_size, stratify=y_clean)
    
    return X_stratified, y_stratified

def shuffle_dataset(X_train, y_train):
    # Losowo przetasowujemy X_train i y_train w sposób spójny
    X_train_shuffled, y_train_shuffled = shuffle(X_train, y_train, random_state=42)
    return X_train_shuffled, y_train_shuffled
    
def merge_chunks(chunks, chunks_y):
    # Inicjalizacja pustych list na połączone dane
    data_set = []
    y = []

    # Iteracja przez wszystkie segmenty i etykiety
    for chunk, chunk_y in zip(chunks, chunks_y):
        # Rozszerzenie listy data_set o elementy z bieżącego segmentu
        data_set.extend(chunk)
        # Rozszerzenie listy y o elementy z bieżących etykiet
        y.extend(chunk_y)

    # Konwersja data_set i y na numpy.array (opcjonalne)
    data_set = np.array(data_set)
    y = np.array(y)

    return data_set, y


def convert_to_dataframe(X, y):
    # Tworzenie DataFrame, gdzie każda kolumna to szereg czasowy (w formie listy)
    X_df = pd.DataFrame({i: [pd.Series(X[j, :, i]) for j in range(X.shape[0])] for i in range(X.shape[2])})
    y_df = pd.Series(y)
    return X_df, y_df

def reshape_data(X, y, n_features):
    n_length, m_length = X.shape

    y_reshaped = np.hstack([np.repeat(y[i], len(X.loc[i, 0])) for i in range(n_length)])  

    columns = []
    for column_id in range(m_length):

        column = []
        for row_id in range(n_length):
            list_length = len(X.loc[row_id, column_id])
            column += list(X.loc[row_id, column_id])
            
        column = np.array(column)
        columns.append(column)
        
    X_reshaped = np.column_stack(columns)
    
    return X_reshaped, y_reshaped


def extend_list(lista, n):
    wynik = []
    for element in lista:
        wynik.extend([element] * n)
    return wynik

def map_strings_to_ints(strings):
    # Utwórz słownik do mapowania stringów na inty
    string_to_int = {}
    current_int = 0
    
    # Wynikowa lista z intami
    result = []
    
    # Przejdź przez listę stringów
    for string in strings:
        # Jeśli string nie jest jeszcze w słowniku, dodaj go
        if string not in string_to_int:
            string_to_int[string] = current_int
            current_int += 1
        # Dodaj odpowiadający int do wynikowej listy
        result.append(string_to_int[string])
    
    return result


#################################################################################

                          ##Create time series from data set##

#################################################################################


def sort_by_class(X, y):
    # Sortowanie według etykiet w y
    sorted_indices = np.argsort(y)
    
    # Zastosowanie posortowanych indeksów do X i y
    X_sorted = X[sorted_indices]
    y_sorted = y[sorted_indices]
    
    return X_sorted, y_sorted

# Dzieli dane dla każdej klasy na odcinki średniej długości mean  z odchyleniem standardowym std_var. Następnie tasuje tak stworzone segmenty.
def shuffle_dataset_with_chunk_sizes(X, y, mean, std_var, seed=42):
    # Ustawiamy seed
    np.random.seed(seed)
    
    # Liczba klas
    num_classes = np.unique(y).size
    y = np.array(y)
    # Listy do przechowywania chunków i etykiet
    X_chunks = []
    y_chunks = []
    chunk_sizes_list = []

    # Ustalanie rozmiaru chunków i mieszanie w obrębie każdej klasy
    start = 0
    for i in range(num_classes):
        # Wyodrębnij dane dla danej klasy
        class_indices = np.where(y == i)[0]
        class_size = len(class_indices)
        
        # Losowanie chunków
        chunk_sizes = []
        while sum(chunk_sizes) < class_size:
            chunk_size = int(np.abs(np.random.normal(mean, std_var)))
            if sum(chunk_sizes) + chunk_size > class_size:
                chunk_size = class_size - sum(chunk_sizes)
            chunk_sizes.append(chunk_size)
        
        # Dzielenie i mieszanie chunków
        class_X_chunks = np.array_split(X[class_indices], np.cumsum(chunk_sizes[:-1]))
        class_y_chunks = np.array_split(y[class_indices], np.cumsum(chunk_sizes[:-1]))
        
        X_chunks.extend(class_X_chunks)
        y_chunks.extend(class_y_chunks)
        chunk_sizes_list.extend(chunk_sizes)

    # Mieszanie w skali całego datasetu
    combined = list(zip(X_chunks, y_chunks, chunk_sizes_list))
    np.random.shuffle(combined)
    
    # Rozdzielenie pomieszanych chunków
    X_shuffled, y_shuffled, shuffled_chunk_sizes = zip(*combined)
    
    # Spłaszczenie listy chunków
    X_shuffled = np.concatenate(X_shuffled)
    y_shuffled = np.concatenate(y_shuffled)

    return X_shuffled, y_shuffled, list(shuffled_chunk_sizes)
