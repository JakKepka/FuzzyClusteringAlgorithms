import numpy as np
from tslearn.datasets import UCR_UEA_datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedShuffleSplit




#################################################################################

                          ##Padding and stratification##

#################################################################################


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

def pad_or_trim_list(lst, padding_length):
    list_length = len(lst)
    
    if list_length < padding_length:
        # Padding - dodaj elementy zgodne z rozkładem normalnym (średnia i std)
        mean = np.mean(lst)
        std = np.std(lst)
        padding_size = padding_length - list_length
        padding_values = np.random.normal(loc=mean, scale=std, size=padding_size).tolist()
        lst.extend(padding_values)
    
    elif list_length > padding_length:
        # Przycinanie - losowo usuń elementy
        lst = np.random.choice(lst, padding_length, replace=False).tolist()
    
    return lst
# Dopasuj każdą listę w DataFrame do zadanej długości padding_length
def adjust_dataframe(df, padding_length):

    return df.applymap(lambda x: pad_or_trim_list(x.tolist(), padding_length))
    
# Funkcja grupuje dane według etykiet z DataFrame y i tworzy nowe obiekty składające się z połączonych elementów z tej samej klasy.
def aggregate_by_class(df, y, l):

    # Upewnij się, że y to jedna kolumna
    if y.shape[1] != 1:
        raise ValueError("y musi być DataFrame z jedną kolumną.")

    # Jeżeli l <= 0 to nie pozostaw elemnty bez zmian
    if(l <= 0):
        chunks_sizes = []
        for i in range(df.shape[0]):
            length = 1
            chunks_sizes.append(length)
            
        return df, y, chunks_sizes 
    df_with_labels = pd.concat([df, y], axis=1)

    # Lista do przechowywania nowych danych
    new_data = []
    new_labels = []
    chunk_sizes = []
    
    # Grupowanie według klas (etykiet)
    for label, group in df_with_labels.groupby(y.iloc[:, 0]):
        group = group.iloc[:, :-1]
        
        # Liczba obiektów w grupie
        k = len(group)

        # Jeżeli liczba obiektów jest mniejsza lub równa l, zachowaj oryginalne obiekty
        if k <= l:
            new_data.append(group)
            new_labels.extend([label] * len(group.values))
            chunk_sizes.append(len(group.values))
        else:
            # Oblicz, ile obiektów połączyć
            step = int(np.ceil(k / l))
            
            # Dzielenie i łączenie obiektów
            for i in range(0, k, step):
                chunk = group.iloc[i:i+step]
                
                # Uśrednianie wartości w obrębie "chunku" (grupy)
                combined = chunk#.apply(lambda col: np.mean(col.values, axis=0))
                
                # Dodawanie nowego obiektu i jego etykiety
                new_data.append(combined)
                new_labels.extend([label] * len(combined))
                chunk_sizes.append(len(combined))
                
    # Zwróć nowe DataFrame z połączonymi obiektami i etykietami
    new_df = pd.concat(new_data, ignore_index=True)
    new_labels_df = pd.DataFrame(new_labels, columns=[y.columns[0]])
    return new_df, new_labels_df, chunk_sizes

#################################################################################

                          ##Diffrent##

#################################################################################


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

def shuffle_dataset(X_train, y_train):
    # Losowo przetasowujemy X_train i y_train w sposób spójny
    X_train_shuffled, y_train_shuffled = shuffle(X_train, y_train, random_state=42)
    return X_train_shuffled, y_train_shuffled


def stratified_chunks(X, y, chunks_length):
    # Sprawdzamy czy suma chunks_length jest równa liczbie punktów danych
    if sum(chunks_length) != len(X):
        raise ValueError("Suma wartości chunks_length musi być równa liczbie punktów w X i y.")
    
    # Zmieniamy dane na numpy arrays dla łatwiejszej manipulacji
    X = np.array(X)
    y = np.array(y)

    # Przechowujemy wyniki
    X_chunks = []
    y_chunks = []

    # Kopie danych do przetwarzania
    X_remaining = X.copy()
    y_remaining = y.copy()

    # Algorytm stratyfikowanego podziału na chunki
    for chunk_size in chunks_length:
        # Stosujemy StratifiedShuffleSplit, aby wylosować próbkę o wielkości chunk_size
        sss = StratifiedShuffleSplit(n_splits=1, train_size=chunk_size, random_state=None)
        for chunk_idx, _ in sss.split(X_remaining, y_remaining):
            X_chunk, y_chunk = X_remaining[chunk_idx], y_remaining[chunk_idx]
            
            # Zapisujemy chunk
            X_chunks.append(X_chunk)
            y_chunks.append(y_chunk)
            
            # Usuwamy wylosowane dane, aby podzielić resztę
            X_remaining = np.delete(X_remaining, chunk_idx, axis=0)
            y_remaining = np.delete(y_remaining, chunk_idx, axis=0)

    return X_chunks, y_chunks
    
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
    y_df = pd.DataFrame(y)
    return X_df, y_df

def reshape_data(X, y, chunks_sizes):
    n_length, m_length = X.shape

    y_reshaped = np.hstack([np.repeat(y[i], len(X.iloc[i, 0])) for i in range(n_length)])  

    for i, chunk_size in enumerate(chunks_sizes):
        chunks_sizes[i] *= len(X.iloc[i,0])

    columns = []
    for column_id in range(m_length):
        
        column = []
        for row_id in range(n_length):
            column += list(X.iloc[row_id, column_id])
            
        column = np.array(column)
        columns.append(column)
        
    X_reshaped = np.column_stack(columns)
    
    return X_reshaped, y_reshaped, chunks_sizes


def extend_list(lista, n):
    wynik = []
    for element in lista:
        wynik.extend([element] * n)
    return wynik

def map_strings_to_ints(strings, string_to_int=None):
    # Utwórz słownik do mapowania stringów na inty
    strings = strings[0]

    if(string_to_int is None):
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
    
    return np.array(result), string_to_int


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
