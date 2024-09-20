import numpy as np

def create_chunks(chunk_sizes, matrix):
    # Rozmiary kolejnych chunków, mogą mieć różne wielkości. Następnie ze względu na te liczby dzielone są chunki
    chunks = []
    start = 0
    for size in chunk_sizes:
        end = start + size
        chunks.append(matrix[start:end])
        start = end

    return chunks

def create_dataset_chunks(chunk_sizes, X, y, y_matrix=None):
    # Rozmiary kolejnych chunków, mogą mieć różne wielkości. Następnie ze względu na te liczby dzielone są chunki
    chunks = create_chunks(chunk_sizes, X)
    chunks_y = create_chunks(chunk_sizes, y)
    if y_matrix is not None:
        chunks_y_matrix  = create_chunks(chunk_sizes, y_matrix)
    else:
        chunks_y_matrix = None

    return chunks, chunks_y, chunks_y_matrix