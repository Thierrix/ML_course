import numpy as np
from poly import *
from utils import apply_pca_given_components, build_poly
from normalization import normalize_data


def clean_test_data(x_te, labels, features, median_and_most_probable_class, mean, W, degree) :
    
    #Keep only the features we kept on the training set 
    mask = [feature in features.keys() for feature in labels]
    x_te = x_te[:, mask]
    
    #Replace the nan in the same way we did on the training set
    for feature in features :
        nan = median_and_most_probable_class[feature]
        x_te[:, features[feature]] = np.nan_to_num(x_te[:,features[feature]], nan = median_and_most_probable_class[feature])

    #Normalize the testing data independtly from the training set
    x_te = normalize_data(x_te)
    
    #Apply the pca given the training PCA
    x_te = apply_pca_given_components(x_te, mean, W)
  
    return x_te

