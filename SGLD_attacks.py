# -*- coding: utf-8 -*-
"""
Created on Tue May 19 15:22:04 2020

@author: zxx
"""
import random
import time
import copy

#from six.moves import xrange
import numpy as np
from numpy.random import RandomState


from dataset import load_movielens_ratings
from dataset import build_user_item_matrix
from ALS_optimize import ALS
from ALS_optimize_origin import ALS_origin
from evaluation import predict
from evaluation import RMSE

from compute_grad import compute_grad_SGLD


def random_mal_ratings(mal_user,n_item,mal_item,seed = None):
    # random generator malicious users data
    assert mal_item < n_item
    mal_ratings = []
    for u in range(mal_user):
        mal_user_idx = u
        mal_item_idx = random.sample(range(n_item), mal_item)
        for i in range(mal_item):
            mal_movie_idx = mal_item_idx[i]
            RandomState(seed).rand()/0.2
            mal_rating = int(5 * RandomState(seed).rand()) + 1
            mal_ratings.append([mal_user_idx, mal_movie_idx, mal_rating])
    return np.array(mal_ratings)



#数据类型转换的函数
def arraytorating(malarray, mal_user, n_item):
    malrating = []
    for u in range(mal_user):
        for i in range(n_item):
            if malarray[u,i] != 0 :
                malrating.append([u, i, malarray[u,i]])
    return np.array(malrating)


############################################################################################################




#train origin model
def optimize_model_origin(converge, n_user, n_item, n_feature, train, mean_rating_, lamda_u, lamda_v, user_features_origin_, item_features_origin_):
    
    print("Start training model without data poisoning attacks!")
    last_rmse = None
    n_iters = 100
    for iteration in range(n_iters):
        t1 = time.time()
        user_features_origin_, item_features_origin_ = ALS_origin(n_user, n_item, n_feature, train, mean_rating_, lamda_u, lamda_v, user_features_origin_, item_features_origin_)
        train_preds = predict(train.take([0, 1], axis=1), user_features_origin_, item_features_origin_)
        train_rmse = RMSE(train_preds, train.take(2, axis=1) - 3)
        t2 = time.time()
        print("The %d th iteration \t time: %ds \t RMSE: %f " % (iteration + 1, t2 - t1, train_rmse))
        # stop when converge
        if last_rmse and abs(train_rmse - last_rmse) < converge:
            break
        else:
            last_rmse = train_rmse
    return last_rmse

#train added attack data model
def optimize_model(converge, n_user, n_item, n_feature, mal_user, train, mean_rating_, mal_mean_rating_, mal_ratings, lamda_u, lamda_v, \
        user_features_, mal_user_features_, item_features_):
    print("Start training model with data poisoning attacks!")
    last_rmse = None
    n_iters = 100
    for iteration in range(n_iters):
        t1 = time.time()

        user_features_, mal_user_features_, item_features_ = ALS(n_user, n_item, n_feature, mal_user, train, \
                                mean_rating_, mal_mean_rating_, mal_ratings, lamda_u, lamda_v, \
                                user_features_, mal_user_features_, item_features_)
        train_preds = predict(train.take([0, 1], axis=1), user_features_, item_features_)
        train_rmse = RMSE(train_preds, train.take(2, axis=1) - 3)
        t2 = time.time()
        print("The %d th iteration \t time: %ds \t RMSE: %f " % (iteration + 1, t2 - t1, train_rmse))
        # stop when converge
        if last_rmse and abs(train_rmse - last_rmse) < converge:
            break
        else:
            last_rmse = train_rmse
    return last_rmse



def main_SGLD(data_size,attack_size = 0.05,fill_size = 0.05,target_item = 22):
    '''
    parameters:
    lamda_u: the regularization parameter of user
    lamda_v: the regularization parameter of item
    alpha: the proportion of malicious users
    mal_item: the items of malicious users rating
    n_iter: number of iteration
    converge: the least RMSE between two iterations
    train_pct: the proportion of train dataset
    '''
    
    lamda_u = 5e-2
    lamda_v = 5e-2
#    alpha = 0.01
#    n_iters = 100
    n_feature = 64
    converge = 1e-5
#    mal_item = 84
#    target_item = 22
    if data_size == '100K':
        ratings_file = 'ratings_ml.csv'
        ratings = load_movielens_ratings(ratings_file)
    if data_size == '1M':
        ratings = np.load('ratings_1m.npy')


    
    #断言评分的最大值为5，最小值为1
    max_rating = max(ratings[:, 2])
    min_rating = min(ratings[:, 2])
    assert max_rating == 5
    assert min_rating == 1
                                                     
    

    train = ratings
    n_user = max(train[:, 0]) + 1
    n_item = max(train[:, 1]) + 1
    mal_user = int(attack_size * n_user) 
#    mal_user = 47
    mal_item = int(fill_size * n_item) 
    
    # add malicious users data
    mal_ratings = random_mal_ratings(mal_user,n_item,mal_item)
    
    #initialize the matrix U U~ and V 
    seed = None
    user_features_ = 0.1 * RandomState(seed).rand(n_user, n_feature)
    mal_user_features_ = 0.1 * RandomState(seed).rand(mal_user, n_feature)
    item_features_ = 0.1 * RandomState(seed).rand(n_item, n_feature)
    mean_rating_ = np.mean(train.take(2, axis=1))
    mal_mean_rating_ = np.mean(mal_ratings.take(2, axis=1))
    user_features_origin_ = 0.1 * RandomState(seed).rand(n_user, n_feature)
    item_features_origin_ = 0.1 * RandomState(seed).rand(n_item, n_feature)

    


    #using the algorithm of SGLD to optimize the utility function
    '''
    s_iters: number of iteration in SGLD
    s_t: step size 
    Lamda: the contraint of vector
    '''
    
    print('*'*40)
    #m_iters = 10
    s_iters = 10
    s_t = 0.2 * np.ones([s_iters])

#    last_rmse = optimize_model_origin()
    last_rmse = optimize_model_origin(converge, n_user, n_item, n_feature, train, mean_rating_, \
                          lamda_u, lamda_v, user_features_origin_, item_features_origin_)
    print(last_rmse)
    
    
    
    
    #n_user = max(train[:, 0]) + 1
    #n_item = max(train[:, 1]) + 1
    
    train_matrix = np.zeros((n_user, n_item))
    for i in range(train.shape[0]):
        train_matrix[train[i,0], train[i,1]] = train[i,2]
    
    item_mean = np.zeros([n_item,])
    item_variance = np.zeros([n_item,])
    for j in range(n_item):
    #    j_num = sum(1 * (train_matrix[:,j] != 0))
    #    j_sum = sum(train_matrix[:,j])
    #    item_mean[j] = j_sum/j_num
    #    item_variance[j] = sum(pow(train_matrix[:,j][train_matrix[:,j] != 0] - item_mean[j] , 2))
    #    
        j_num = n_user
        j_sum = sum(train_matrix[:,j])
        item_mean[j] = j_sum/j_num
        item_variance[j] = sum(pow(train_matrix[:,j] - item_mean[j] , 2))/j_num
    
    #mal_data = build_user_item_matrix(mal_user,n_item,mal_ratings).toarray()
    
    mal_data = np.zeros([mal_user,n_item])
    #count = np.zeros([mal_user,n_item])
    for u in range(mal_user):
    #    print(u)
        item_index = np.random.randint(1,n_item,mal_item)
        for j in range(mal_item):
            randomdata = np.random.normal(loc=item_mean[item_index[j]], scale=np.sqrt(item_variance[item_index[j]]))
            randomdata = randomdata - item_mean[item_index[j]] + 3
            if randomdata > 0:
                mal_data[u,item_index[j]] = round(randomdata)
            else:
                mal_data[u,item_index[j]] = 0
            
            if mal_data[u,item_index[j]] > 5:
                mal_data[u,item_index[j]] = 5
            
    #        count[u,item_index[j]] = 1
    #sum(sum((mal_data !=0 ) * 1))       
    #sum(sum(count))
    
    mal_ratings = arraytorating(mal_data, mal_user, n_item).astype(np.int32)
    mal_mean_rating_ = np.mean(mal_ratings.take(2, axis=1))
    
    
    beta = 0.6
    sate = np.zeros([mal_user,n_item])
    
    inverse_diag = np.zeros([n_item,n_item])
    diag_sqrt = np.zeros([n_item,n_item])
    for j in range(n_item):
        sate[:,j] = item_mean[j]
        inverse_diag[j,j] = 1/item_variance[j]
        diag_sqrt[j,j] = np.sqrt(item_variance[j])
    
    zero_index = np.where(item_variance == 0)[0]
    for index in range(zero_index.shape[0]):
        inverse_diag[zero_index[index],zero_index[index]] = 0
    
    #inverse_diag[0,0] = 0 
    mal_data_sum = {0:mal_data}
    
    mal_data_index = copy.deepcopy(mal_data)
    
    mal_data_index_dic = {}
    for i in range(mal_user):
        mal_data_index_dic[i] = np.where(mal_data[i,:] != 0)
    
    last_rmse = optimize_model(converge, n_user, n_item, n_feature, mal_user, train, \
                       mean_rating_, mal_mean_rating_, mal_ratings, lamda_u, \
                       lamda_v, user_features_, mal_user_features_, item_features_)

    
    for t in range(s_iters):
        print(t)
        t1 = time.time()
#        optimize_model(converge, n_user, n_item, n_feature, mal_user, train, \
#                       mean_rating_, mal_mean_rating_, mal_ratings, lamda_u, \
#                       lamda_v, user_features_, mal_user_features_, item_features_)
#        w_j0 = 2, u1 = 0.5, u2 = 0.5
        grad_total = compute_grad_SGLD(mal_data_index_dic, n_user, n_item, mal_user, mal_ratings, train, user_features_, mal_user_features_, \
                            item_features_, diag_sqrt, n_feature, user_features_origin_, item_features_origin_, target_item)
#        mal_data = np.dot(mal_user_features_, item_features_.T)
#        grad_total = 0.01*np.ones([mal_user,n_item])
#        grad_total = np.load('grad.npy')
        temp = copy.deepcopy(mal_data)
        deta = beta * grad_total - np.dot((mal_data - sate) ,inverse_diag)
        gauss_noise = random.gauss(0,s_t[t])
        mal_data_new = mal_data + (s_t[t]/2) * deta + gauss_noise
        mal_data_new[mal_data_new > 5] = 5
        mal_data_new[mal_data_new < 1] = 1
        mal_data = np.multiply(mal_data_new,1 * (mal_data_index != 0 )) #projection
    #    mal_data +=  grad_total * s_t[t]
    
        mal_ratings = arraytorating(mal_data, mal_user, n_item).astype(np.int32)
        mal_mean_rating_ = np.mean(mal_ratings.take(2, axis=1))
        mal_data_sum[t+1] = mal_data
        np.save('attack_SGLD_05_%d.npy'%t,(mal_data + 0.5).astype(np.int32))
        print('PGA saved %d'%t) 
        
        
        rmse = optimize_model(converge, n_user, n_item, n_feature, mal_user, train, \
                       mean_rating_, mal_mean_rating_, mal_ratings, lamda_u, \
                       lamda_v, user_features_, mal_user_features_, item_features_)

        t2 = time.time()
        print("The %d th iteration \t time: %ds \t RMSE: %f " % (t + 1, t2 - t1, rmse))

    
    values = np.zeros([s_iters+1,])
    for t in range(s_iters+1):
        
        now_mal_data = mal_data_sum[t]
    
        for i in range(s_iters+1):
            value =sum(sum( (now_mal_data -  mal_data_sum[i]) * (now_mal_data -  mal_data_sum[i])))
            values[t] = values[t] + value
    
    final_mal_data = mal_data_sum[(np.where(values == np.min(values)))[0][0]]
    
    np.save('attack_SGLD_05.npy',(final_mal_data + 0.5).astype(np.int32))
    print('SGLD saved')
    
    
    
    return (final_mal_data + 0.5).astype(np.int32)

#test = np.load('attack_PGA.npy')
data_size = '1M'
#mal_SGLD = main_SGLD(data_size)

###########################################################################################################



##########################################################################
#from attack_evaluation import attack_df_rmse
#
# 
#mal_data_df = mal_data.astype(np.int32) + 3
#attack_df = np.concatenate((train_matrix,mal_data_df),axis = 0)
#
#def attack_r(attack_df):
#    attack_rating = []
#    for u in range(attack_df.shape[0]):
#        for i in range(attack_df.shape[1]):
#            if attack_df[u,i] != 0 :
#                attack_rating.append([u, i, attack_df[u,i]])
#    attack_rating = np.array(attack_rating).astype(np.int32)
#    return attack_rating
#
#attack_rating = attack_r(attack_df)
#attack_df_rmse(attack_rating,attack_df.shape[0],attack_df.shape[1])


