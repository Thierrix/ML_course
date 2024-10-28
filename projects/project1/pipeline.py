from implementations import *
from functions
def pipeline() :
    print('Starting pipeline')
    print("Beginning data loading")

    # Load data
    DATA_PATH = 'C:/Users/clavo\OneDrive - epfl.ch/EPFL/Cours/Semester 1/CS-433/Project/dataset/dataset' # Change to path for dataset.zip extracted folder
    x_train, x_test, y_train, train_ids, test_ids, labels =  load_csv_data(DATA_PATH, sub_sample=False)
    labels.pop(0)    

    print("Loading complete")
    #Split the data into training and testing
    x_tr, x_te, y_tr, y_te = split_data(x_train, y_train, 0.8, seed= 2)


    lambdas = [0.01]
    up_sampling_percentages = [0.2]
    degrees = [1]
    variances_threshold = [0.995]
    decision_threshold = [0.47]
    acceptable_nan_percentages = [1]
    max_iters = [300]
    outliers_row_limit = [0.7]
    gammas = [0.5]
    nan_handlers = ['numeric']
    k_fold = 4

    
    models = ['mean_squared_error_gd', 'mean_squared_error_sgd']
    best_gamma, best_up_sampling_percentage, best_degree, best_variance_threshold, best_lambda, \
    best_max_iter, best_f1_score, best_threshold, best_nan_percentage, best_outlier_limit, \
    best_nan_handler, best_model = grid_search_k_fold(
        y_tr, 
        x_tr, 
        k_fold, 
        lambdas, 
        gammas, 
        up_sampling_percentages,
        degrees, 
        variances_threshold, 
        max_iters,
        decision_threshold, 
        acceptable_nan_percentages,
        labels, 
        outliers_row_limit, 
        nan_handlers
    )

    return 0






pipeline()