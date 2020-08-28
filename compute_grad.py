import random
import numpy as np
from dataset import build_user_item_matrix
from numpy.linalg import inv
#compute the gradient of the hyrid utility function

    


def compute_utility_grad(n_user, n_item, train, user_features_, item_features_,user_features_origin_, item_features_origin_, \
    target_item, w_j0 = 2, u1 = 0.5, u2 = 0.5):
    ratings_csr_ = build_user_item_matrix(n_user, n_item, train)
    grad_av = 2 * (np.dot(user_features_, item_features_.T) - np.dot(user_features_origin_, item_features_origin_.T))
    for i in range(n_user):
        _, item_idx = ratings_csr_[i, :].nonzero()
        grad_av[i, item_idx] = 0
    avg_rating = np.mean(np.dot(user_features_, item_features_.T), axis = 0)
    perfer_index = np.where(avg_rating > 0.02)        #####这个需要改进
#    J0 = random.sample(list(perfer_index[0]), 1)
    J0 = target_item
    grad_in = np.zeros([n_user, n_item])
    grad_in[:, J0] = w_j0 
    grad_hy = u1 * grad_av + u2 * grad_in
    
    return grad_hy

#def compute_grad_PGA(n_user, n_item, mal_user, mal_ratings, train, user_features_, mal_user_features_, \
#    item_features_, lamda_v, n_feature, user_features_origin_, item_features_origin_, target_item):
#    '''
#    A : inv(lamda_v * Ik + sum(u_i* u_i))   (for u_i of item j)  k * k
#    u_i : 1 * k
#    grad_model: d(u_i * v_j.T)/d(M_ij) = u_i * A * u_i.T
#    '''
#    grad_R = compute_utility_grad(n_user, n_item, train, user_features_, \
#            item_features_, user_features_origin_, item_features_origin_, target_item)
#    ratings_csr_ = build_user_item_matrix(n_user, n_item, train)
#    ratings_csc_ = ratings_csr_.tocsc()
#    mal_ratings_csr_ = build_user_item_matrix(mal_user, n_item, mal_ratings)
#    mal_ratings_csc_ = mal_ratings_csr_.tocsc()
#    grad_total = np.zeros([mal_user, n_item])
#    for i in range(mal_user):
#        for j in range(n_item):
#            if j % 1000 == 0:
#                print('Computing the %dth malicious user, the %d item(total users: %d, total items: %d)' % (i, j, n_user, n_item))
#            user_idx, _ = ratings_csc_[:, j].nonzero()
#            mal_user_idx, _ = mal_ratings_csc_[:, j].nonzero()
#            user_features = user_features_.take(user_idx, axis=0)
#            mal_user_features = mal_user_features_.take(mal_user_idx, axis=0)
#            U = np.vstack((user_features, mal_user_features))  
#            u_i = user_features_.take(i, axis = 0)
#            A = np.dot(U.T, U) + lamda_v * np.eye(n_feature)  
##            A_u = np.dot(A, u_i.T)
#            grad_model = np.zeros([n_user, n_item])
#            for m in range(n_user):
#                u_m = user_features_.take(i, axis = 0)
#                grad_model[m, j] = np.dot(u_m, np.dot(inv(A), u_i.T))
#            grad_total[i, j] = sum(sum(grad_model * grad_R))
#    return grad_total
            

def compute_grad_PGA(mal_data_index_dic, n_user, n_item, mal_user, mal_ratings, train, user_features_, mal_user_features_, \
    item_features_, lamda_v, n_feature, user_features_origin_, item_features_origin_, target_item):
    '''
    A : inv(lamda_v * Ik + sum(u_i* u_i))   (for u_i of item j)  k * k
    u_i : 1 * k
    grad_model: d(u_i * v_j.T)/d(M_ij) = u_i * A * u_i.T
    '''
    grad_R = compute_utility_grad(n_user, n_item, train, user_features_, \
            item_features_, user_features_origin_, item_features_origin_, target_item)
    ratings_csr_ = build_user_item_matrix(n_user, n_item, train)
    ratings_csc_ = ratings_csr_.tocsc()
    mal_ratings_csr_ = build_user_item_matrix(mal_user, n_item, mal_ratings)
    mal_ratings_csc_ = mal_ratings_csr_.tocsc()
    grad_total = np.zeros([mal_user, n_item])
    import time
    t1= time.time()
    for i in range(mal_user):
        print('Computing the %dth malicious user' %i)
        mal_use_index = mal_data_index_dic[i][0]
        for j in range(mal_use_index.shape[0]):
#            print(j)
            mal_item = mal_use_index[j]
            user_idx, _ = ratings_csc_[:, mal_item].nonzero()
            mal_user_idx, _ = mal_ratings_csc_[:, mal_item].nonzero()
            user_features = user_features_.take(user_idx, axis=0)
            mal_user_features = mal_user_features_.take(mal_user_idx, axis=0)
            U = np.vstack((user_features, mal_user_features))  
            u_i = user_features_.take(i, axis = 0)
            A = np.dot(U.T, U) + lamda_v * np.eye(n_feature)  
#            A_u = np.dot(A, u_i.T)
            grad_model = np.zeros([n_user, n_item])
            for m in range(n_user):
                u_m = user_features_.take(i, axis = 0)
                grad_model[m, mal_item] = np.dot(u_m, np.dot(inv(A), u_i.T))
            grad_total[i, mal_item] = sum(sum(grad_model * grad_R))
        t2 = time.time()
        print(t2-t1)
        t1= time.time()
    return grad_total



#
def compute_grad_SGLD(mal_data_index_dic, n_user, n_item, mal_user, mal_ratings, train, user_features_, mal_user_features_, \
    item_features_, diag_sqrt, n_feature, user_features_origin_, item_features_origin_, target_item):
    '''
    B: 
    A : inv(lamda_v * Ik + sum(u_i* u_i))   (for u_i of item j)  k * k
    u_i : 1 * k
    grad_model: d(u_i * v_j.T)/d(M_ij) = u_i * A * u_i.T
    '''
    lamda = 0.05
    tao = 0.01
#    W = np.zeros([944,1683])  #需要 优化
#    sum(1*(mal_ratings.take(1, axis=1) == 47))
#    ou[0] = mal_ratings.take(0, axis=1)
#    r = 
    grad_R = compute_utility_grad(n_user, n_item, train, user_features_, \
            item_features_, user_features_origin_, item_features_origin_, target_item)
    ratings_csr_ = build_user_item_matrix(n_user, n_item, train)
    ratings_csc_ = ratings_csr_.tocsc()
    mal_ratings_csr_ = build_user_item_matrix(mal_user, n_item, mal_ratings)
    mal_ratings_csc_ = mal_ratings_csr_.tocsc()
    grad_total = np.zeros([mal_user, n_item])
    diag_sqrt = diag_sqrt[0:n_feature,0:n_feature]
    import time
    
    t1= time.time()
    for i in range(mal_user):
        print('Computing the %dth malicious user' %i)
        mal_use_index = mal_data_index_dic[i][0]
        for j in range(mal_use_index.shape[0]):
#            print(j)
            mal_item = mal_use_index[j]
            user_idx, _ = ratings_csc_[:, mal_item].nonzero()
            item_idx, _ = ratings_csc_[i, :].nonzero()
            mal_user_idx, _ = mal_ratings_csc_[:, mal_item].nonzero()
            user_features = user_features_.take(user_idx, axis=0)
            item_features = item_features_.take(item_idx, axis=0)
            mal_user_features = mal_user_features_.take(mal_user_idx, axis=0)
            U = np.vstack((user_features, mal_user_features))  
            u_i = user_features_.take(i, axis = 0)
            v_j = item_features_.take(mal_item, axis = 0)
            B = np.dot(U, diag_sqrt + lamda * np.eye(n_feature))  
            Bi = np.dot(B.T , B) + tao * np.eye(n_feature)
#            A_u = np.dot(A, u_i.T)
            grad_model = np.zeros([n_user, n_item])
            for m in range(n_user):
                u_m = user_features_.take(i, axis = 0)
                v_grad = np.dot(np.dot(inv(Bi), diag_sqrt + lamda * np.eye(n_feature)), u_m)
                e_grad = 1/np.dot(mal_user_features, item_features.T)
                grad_model[m, mal_item] = np.dot(u_m, np.dot(v_grad, diag_sqrt)) + np.sum(sum(np.dot(np.dot(u_i,v_j), e_grad)))
            grad_total[i, mal_item] = sum(sum(grad_model * grad_R))
            
            
        t2 = time.time()
        print(t2-t1)
        t1= time.time()
    
    
    return grad_total   

#from sympy import *
#x = Symbol('x')
#y = Symbol('y')
#solve([2 * x - y - 3, 3 * x + y - 7],[x, y])
#
#A = user_features_.T
#b = np.zeros([8,1683])
#x = np.linalg.solve(A, b)
#
#np.linalg.norm(x2,ord=2) #矩阵的2范数
