"""
title           :SVD_main.py
description     :the computation can be done using directly a SVD (Singular Value Decomposition) algorithm
author          :DevHyung
date            :2018.01.09
version         :1.0.0
python_version  :3.6
required module :tensorflow, numpy
develope env    : Mac OS X High Sierra, intel i5 (2Ghz), using cpu
[ref]
1. https://www.bonaccorso.eu/2017/08/02/svd-recommendations-using-tensorflow/
2.
"""
import numpy as np
import tensorflow as tf
import time
# Set random seed for reproducibility
np.random.seed(1000)
#####_____ Set traning variable _____#####
nb_users = 5000
nb_products = 2000
nb_factors = 500
max_rating = 5
nb_rated_products = 500
top_k_products = 10
##########################################
if __name__ == "__main__":
    # Calculate runtime
    start_time = time.time()
    # Create a random User-Item matrix
    uim = np.zeros((nb_users, nb_products), dtype=np.float32)
    for i in range(nb_users):
        nbp = np.random.randint(0, nb_products, size=nb_rated_products)
        for j in nbp:
            uim[i, j] = np.random.randint(1, max_rating + 1)
    # Create a Tensorflow graph
    graph = tf.Graph()
    with graph.as_default():
        # User-item matrix
        user_item_matrix = tf.placeholder(tf.float32, shape=(nb_users, nb_products))
        # SVD
        St, Ut, Vt = tf.svd(user_item_matrix)
        # Compute reduced matrices
        Sk = tf.diag(St)[0:nb_factors, 0:nb_factors]
        Uk = Ut[:, 0:nb_factors]
        Vk = Vt[0:nb_factors, :]
        # Compute Su and Si
        Su = tf.matmul(Uk, tf.sqrt(Sk))
        Si = tf.matmul(tf.sqrt(Sk), Vk)
        # Compute user ratings
        ratings_t = tf.matmul(Su, Si)
        # Pick top k suggestions
        best_ratings_t, best_items_t = tf.nn.top_k(ratings_t, top_k_products)
    # Create Tensorflow session
    session = tf.InteractiveSession(graph=graph)
    # Compute the top k suggestions for all users
    feed_dict = {
        user_item_matrix: uim
    }
    best_items = session.run([best_items_t], feed_dict=feed_dict)
    # Suggestions for user 1000, 1010
    for i in range(1000, 1010):
        print('User {}: {}'.format(i, best_items[0][i]))
    print("--- %s seconds ---" % (time.time() - start_time))
    """
    my develope SPEC take time show below 
    --- 19.178043842315674 seconds ---
    """
