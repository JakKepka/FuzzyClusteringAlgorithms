import numpy as np

# Tworzeniu chunków z danych, rozmiary chunków pochodza z listy chunk_sizes
def create_chunks(chunk_sizes, matrix):
    # Rozmiary kolejnych chunków, mogą mieć różne wielkości. Następnie ze względu na te liczby dzielone są chunki
    chunks = []
    start = 0
    for size in chunk_sizes:
        end = start + size
        chunks.append(matrix[start:end])
        start = end

    return chunks

# Utwórz zestaw chunków dla danych
def create_dataset_chunks(chunk_sizes, X, y, y_matrix=None):
    # Rozmiary kolejnych chunków, mogą mieć różne wielkości. Następnie ze względu na te liczby dzielone są chunki
    chunks = create_chunks(chunk_sizes, X)
    chunks_y = create_chunks(chunk_sizes, y)
    if y_matrix is not None:
        chunks_y_matrix  = create_chunks(chunk_sizes, y_matrix)
    else:
        chunks_y_matrix = None

    return chunks, chunks_y, chunks_y_matrix

# Łączenie chunków w dataset
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