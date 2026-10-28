# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

def get_dict(file_name):
    """
    This function returns the english to french dictionary given a file where the each column corresponds to a word.
    Check out the files this function takes in your workspace.
    """
    my_file = pd.read_csv(file_name, delimiter=' ')
    etof = {}  # the english to french dictionary to be returned
    for i in range(len(my_file)):
        # indexing into the rows.
        # en = my_file.loc[i][0]
        # fr = my_file.loc[i][1]
        en = my_file.loc[i].iloc[0]
        fr = my_file.loc[i].iloc[1]
        etof[en] = fr

    return etof


def cosine_similarity(A, B):
    '''
    Input:
        A: a numpy array which corresponds to a word vector
        B: A numpy array which corresponds to a word vector
    Output:
        cos: numerical number representing the cosine similarity between A and B.
    '''
    # you have to set this variable to the true label.
    cos = -10    
    dot = np.dot(A, B)
    normb = np.linalg.norm(B)
    
    if len(A.shape) == 1: # If A is just a vector, we get the norm
        norma = np.linalg.norm(A)
        cos = dot / (norma * normb)
    else: # If A is a matrix, then compute the norms of the word vectors of the matrix (norm of each row)
        norma = np.linalg.norm(A, axis=1)
        epsilon = 1.0e-9 # to avoid division by 0
        cos = dot / (norma * normb + epsilon)
        
    return cos
