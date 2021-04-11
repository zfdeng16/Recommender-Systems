#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Last modified on 2021/04/10
# RS hw_1 for PMF, coded by DZF

import numpy as np
import matplotlib.pyplot as plt


class PMF():
    
    '''
    [# Function description]
    PMF training.
    '''
    def train (self, num_user, num_item, train, test, learning_rate, D, regu_u, regu_v, maxiteration):
        # Initialize U and V
        U = np.random.normal(0, 0.1, (num_user, D))
        V = np.random.normal(0, 0.1, (num_item, D))
        
        # If has already converged, then stop earlier, see the codes below
        # 2.5 is set according to obseravation only!
        pre_rmse = 2.5
        endure_count = 3
        patience = 0
        max_rmse = 2.5
        
        # Plotting
        iteration_index = []
        RMSE_record = []
        
        # Iterate for 50 epoches (For the sake of saving time, I choose 50 instead of 100)
        for iter in range(maxiteration):
            # Calculate loss function at the beginning of each iteration
            loss = 0.0
            
            for data in train:
                user = data[0]
                item = data[1]
                rating = data[2]
                
                predict_rating = np.dot(U[user],V[item].T)
                error = rating - predict_rating
                
                # The square of the first term (error) in the loss function
                loss += error ** 2
                
                # Using GD to update feature vectors in U and V
                U[user] += learning_rate * (error * V[item] - regu_u * U[user])
                V[item] += learning_rate * (error * U[user] - regu_v * V[item])

                loss+=regu_u * np.square(U[user]).sum() + regu_v * np.square(V[item]).sum()

            loss = 0.5 * loss
            # By now, we have computed the current loss according to the equation shown in 
            #   the article.
            
            # Calculate RMSE according to the test set.
            rmse = self.eval_rmse(U, V, test)
            
            print('iteration:%d loss:%.3f rmse:%.5f'%(iter,loss,rmse))
            
            max_rmse = min(rmse, max_rmse)
            
            # If has already converged, then stop earlier
            if rmse < pre_rmse:
                pre_rmse = rmse
                patience = 0
            else:
                patience += 1
            if patience >= endure_count:
                break
            
            # Storing the datas generated in each itearation for plotting
            iteration_index.append((iter + 1))
            RMSE_record.append(rmse)
        
        # Plotting            
        plt.plot(iteration_index, RMSE_record)
        x_ticks = np.arange(0, maxiteration, 2)
        
        # The upper bound and lower bound are set according to  my observation only!
        y_ticks = np.arange(0.6, 2.1, 0.1)
        plt.xticks(x_ticks)
        plt.yticks(y_ticks)
        plt.title("10D-RMSE")
        plt.xlabel("epoches")
        plt.ylabel("RMSE")
        plt.show()

    '''
    [# Function description]
    Calculate the RMSE for each iteration according to U, V for all items in the test set.
    '''
    def eval_rmse(self, U, V, test):
        test_count = len(test)
        tmp_rmse = 0.0
        for te in test:
            user = te[0]
            item = te[1]
            real_rating = te[2]
            predict_rating = np.dot(U[user], V[item].T)
            tmp_rmse += np.square(real_rating - predict_rating)
        rmse = np.sqrt(tmp_rmse / test_count)
        return rmse


'''
[# Function description]
Load data from the file "u.data".

[# Parameters description]
path: the relative path of the data file, which is u.data.
train_ratio: the size of training set, using hold-out method to get it from the original
                data set.
'''
def read_data(path, train_ratio):
    user_set = {}
    item_set = {}
    u_idx = 0
    i_idx = 0
    data = []
    with open(path) as f:
        for line in f.readlines():
            u, i, r, _ = line.split('::')
            
            # In case of redundancy, update indexes when new object is encountered
            if u not in user_set:
                user_set[u] = u_idx
                u_idx += 1
            if i not in item_set:
                item_set[i] = i_idx
                i_idx += 1
            
            data.append([user_set[u],item_set[i],float(r)])
    
    # Shuffle and divide the dataset into two seperate parts (hold-out)
    np.random.shuffle(data)
    train = data[0 : int(len(data) * train_ratio)]
    test = data[int(len(data) * train_ratio) :]
    return u_idx, i_idx, train, test


'''
[# Function discription]
Load and train.
'''
if __name__=='__main__':
    num_user, num_item, train, test = read_data(path = 'data/ratings.dat', train_ratio = 0.8)
    pmf = PMF()
    pmf.train(num_user, num_item, train, test,
              learning_rate = 0.005, 
              D = 10, 
              regu_u = 0.01, 
              regu_v = 0.001, 
              maxiteration = 50)
