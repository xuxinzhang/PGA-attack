# -*- coding: utf-8 -*-
"""
Created on Tue May 19 15:22:04 2020

@author: zxx
"""
import random
import time
import copy

import numpy as np
from numpy.random import RandomState


from dataset import load_movielens_ratings
from dataset import build_user_item_matrix
from ALS_optimize import ALS
from ALS_optimize_origin import ALS_origin
from evaluation import predict
from evaluation import RMSE

from compute_grad import compute_grad_PGA

#from attack_evaluation import availability_rmse

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

        user_features_, mal_user_features_, item_features_ = ALS(n_user, n_item, n_feature, mal_user, \
                                train, mean_rating_, mal_mean_rating_, mal_ratings, lamda_u, lamda_v, \
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



def main_PGA(data_size,attack_size = 0.05,fill_size = 0.05,target_item = 22):
    '''
    parameters:
    lamda_u: the regularization parameter of user
    lamda_v: the regularization parameter of item
    attack_size: the proportion of malicious users
    mal_item: the items of malicious users rating
    n_iter: number of iteration
    converge: the least RMSE between two iterations
    train_pct: the proportion of train dataset
    '''
    
    n_feature = 64
    lamda_u = 5e-2
    lamda_v = 5e-2
    converge = 1e-5
    
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
#    mal_user = 2
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

    
    
    #using the algorithm of PGA to optimize the utility function
    '''
    m_iters: number of iteration in PGA
    s_t: step size 
    Lamda: the contraint of vector
    '''


    m_iters = 10
    s_t = 0.2 * np.ones([m_iters])

    last_rmse = None
    last_rmse = optimize_model_origin(converge, n_user, n_item, n_feature, train, mean_rating_, \
                          lamda_u, lamda_v, user_features_origin_, item_features_origin_)
    print(last_rmse)
    

    mal_data = build_user_item_matrix(mal_user,n_item,mal_ratings).toarray()
    
    mal_data_index_dic = {} #这个好像是为了只更新有值的部分
    for i in range(mal_user):
        mal_data_index_dic[i] = np.where(mal_data[i,:] != 0)
    
    
    last_rmse = optimize_model(converge, n_user, n_item, n_feature, mal_user, train, \
                       mean_rating_, mal_mean_rating_, mal_ratings, lamda_u, \
                       lamda_v, user_features_, mal_user_features_, item_features_)
    
    
    for t in range(m_iters):
        t1 = time.time()
        
#        grad_total = np.zeros([mal_user, n_item])
##        w_j0 = 2, u1 = 0.5, u2 = 0.5
#        for i in range(mal_user):
#            print('Computing the %dth malicious user' %i)
#            mal_use_index = mal_data_index_dic[i][0]
#            for j in range(mal_use_index.shape[0]):
#    #            print(j)
#                mal_item = mal_use_index[j]
#                
#                grad_total[i, mal_item] = 0.2
        
        grad_total = compute_grad_PGA(mal_data_index_dic, n_user, n_item, mal_user, mal_ratings, train, user_features_, mal_user_features_, \
                            item_features_, lamda_v, n_feature, user_features_origin_, item_features_origin_, target_item)
#        grad_total = 0.01*np.ones([mal_user,n_item])
#        grad_total = np.load('grad.npy')
        temp = copy.deepcopy(mal_data)
        mal_data_new = mal_data + grad_total * s_t[t]
        mal_data_new[mal_data_new > 5] = 5
        mal_data_new[mal_data_new < 1] = 1    
        mal_data = np.multiply(mal_data_new,1 * (temp != 0 ))
        
        np.save('attack_PGA_05_801_%d.npy'%t,(mal_data + 0.5).astype(np.int32))
        print('PGA saved %d'%t)         
        
    #    mal_data[mal_data > 5] = 5
    #    mal_data[mal_data < 0] = 0
        mal_ratings = arraytorating(mal_data, mal_user, n_item).astype(np.int32)
        mal_mean_rating_ = np.mean(mal_ratings.take(2, axis=1))
#        rmse = RMSE(mal_data, temp)
        rmse = optimize_model(converge, n_user, n_item, n_feature, mal_user, train, \
                       mean_rating_, mal_mean_rating_, mal_ratings, lamda_u, \
                       lamda_v, user_features_, mal_user_features_, item_features_)

        t2 = time.time()
        print("The %d th iteration \t time: %ds \t RMSE: %f " % (t + 1, t2 - t1, rmse))
        if last_rmse and abs(rmse - last_rmse) < converge:
            break
        else:
            last_rmse = rmse
    
    np.save('attack_PGA_05.npy',(mal_data + 0.5).astype(np.int32))
    print('PGA saved')
    
    
    return (mal_data + 0.5).astype(np.int32)

#test = np.load('attack_PGA_05.npy')
#data_size = '100K'
data_size = '1M'
#rmse_av, rmse_seen = availability_rmse(train, mal_data, converge, lamda_u, lamda_v)
#mal_PGA = main_PGA(data_size)



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


