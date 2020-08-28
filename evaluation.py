import numpy as np

def predict(data, user_features_, item_features_, max_rating = 2, min_rating = -2):
#    (data, user_features_, item_features_) =  (train.take([0, 1], axis=1), user_features_origin_, item_features_origin_)
    data = data.astype(int)
    u_features = user_features_.take(data.take(0, axis=1), axis=0) 
    i_features = item_features_.take(data.take(1, axis=1), axis=0)
    preds = np.sum(u_features * i_features, 1)
    if max_rating:
        preds[preds > max_rating] = max_rating
    if min_rating:
        preds[preds < min_rating] = min_rating
    return preds

def RMSE(estimation, truth):
    """Root Mean Square Error"""
    estimation = np.float64(estimation)
    truth = np.float64(truth)
    num_sample = estimation.shape[0]
    
    # sum square error
    sse = np.sum(np.square(truth - estimation))
    return np.sqrt(np.divide(sse, num_sample - 1))
