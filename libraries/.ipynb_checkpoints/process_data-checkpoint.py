import numpy as np


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