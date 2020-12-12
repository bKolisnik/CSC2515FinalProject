#Custom implementation of Simon Funk's SVD algorithm. Try to use all recommended hyperparams.

import pandas as pd
import numpy as np
import scipy
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
from sklearn.metrics import mean_squared_error

np.random.seed(42)

class CustomSVD():
    def __init__(self, df,user_col, item_col, rating_col, hidden_factors=100, init_mean=0, init_std_dev=0.1):
        self.train_df = df
        self.user_col = user_col
        self.item_col = item_col
        self.rating_col = rating_col
        self.hidden_factors = hidden_factors

        self.num_users = len(self.train_df[self.user_col].unique())
        self.num_items = len(self.train_df[self.item_col].unique())
        self.user_dict = {key: index for index, key in enumerate(self.train_df[user_col].unique())}
        self.item_dict = {key: index for index, key in enumerate(self.train_df[item_col].unique())}


        self.global_mean = self.train_df.loc[:,self.rating_col].mean()
        self.user_bias = np.zeros(self.num_users)
        self.item_bias = np.zeros(self.num_items)

        #Matrices corresponding to each user and items hidden factor.
        self.user_hidden_factor = np.random.normal(init_mean, init_std_dev,(self.num_users,self.hidden_factors))
        self.item_hidden_factor = np.random.normal(init_mean, init_std_dev,(self.num_items,self.hidden_factors))

    def predict_rating_test(self,row):
        user = row[0]
        item = row[1]
        if user not in self.user_dict:
            b_u = 0
            p_u = np.zeros(self.hidden_factors)
        else:
            user_ind = self.user_dict[user]
            b_u = self.user_bias[user_ind]
            p_u = self.user_hidden_factor[user_ind]

        if item not in self.item_dict:
            b_i = 0
            q_i = np.zeros(self.hidden_factors)
        else:
            item_ind = self.item_dict[item]
            b_i = self.item_bias[item_ind]
            q_i = self.item_hidden_factor[item_ind]

        prediction = self.global_mean + b_u + b_i + np.dot(q_i,p_u)

        if(prediction > 5):
            return 5
        elif(prediction < 1):
            return 1

        return prediction

    def predict_rating_train(self, row, learning_rate, regularization_parameter):
        #prediction and SGD in one step. Only return prediction
        user = row[0]
        item = row[1]
        if user not in self.user_dict:
            b_u = 0
            p_u = np.zeros(self.hidden_factors)
        else:
            user_ind = self.user_dict[user]
            b_u = self.user_bias[user_ind]
            p_u = self.user_hidden_factor[user_ind]

        if item not in self.item_dict:
            b_i = 0
            q_i = np.zeros(self.hidden_factors)
        else:
            item_ind = self.item_dict[item]
            b_i = self.item_bias[item_ind]
            q_i = self.item_hidden_factor[item_ind]

        prediction = self.global_mean + b_u + b_i + np.dot(q_i,p_u)

        error = row[2] - prediction

        if user in self.user_dict:
            self.user_bias[user_ind] = b_u + learning_rate*(error - regularization_parameter*b_u)
            self.user_hidden_factor[user_ind] = p_u + learning_rate*(error*q_i - regularization_parameter*p_u)

        if item in self.item_dict:
            self.item_bias[item_ind] = b_i + learning_rate*(error - regularization_parameter*b_i)
            self.item_hidden_factor[item_ind] = q_i + learning_rate*(error*p_u - regularization_parameter*q_i)
        
        if(prediction > 5):
            return 5
        elif(prediction < 1):
            return 1

        return prediction

    def train(self, n_epochs= 20, learning_rate=0.005, regularization_parameter=0.02):
        #just pass in the subset of the dataframe with the necessary columns
        train_matrix = self.train_df[[self.user_col,self.item_col,self.rating_col]].to_numpy()
        num_preds = len(train_matrix)
        answers = train_matrix[:,2].flatten()
        preds = np.zeros(num_preds)

        for e in range(n_epochs):
            for index, row in enumerate(train_matrix):
                prediction = self.predict_rating_train(row,learning_rate,regularization_parameter)
                preds[index] = prediction

            mse = mean_squared_error(answers,preds)
            print(f"Epoch {e}, training mse: {mse}")

    def predict(self,df):
        prediction_features = df[[self.user_col,self.item_col]].to_numpy()
        num_preds = len(prediction_features)

        preds = np.zeros(num_preds)

        for index, row in enumerate(prediction_features):
            prediction = self.predict_rating_test(row)
            #print(prediction)
            preds[index] = prediction

        return preds


class Hypothesis1(CustomSVD):
    def __init__(self, df,user_col, item_col, rating_col, expertise_col,hidden_factors=100, init_mean=0, init_std_dev=0.1):
        super().__init__(df, user_col, item_col, rating_col, hidden_factors, init_mean, init_std_dev)
        
        self.expertise = np.empty(self.num_users)

        for reviewer in self.train_df[user_col].unique():
            self.expertise[self.user_dict[reviewer]] = self.train_df.loc[self.train_df[user_col]==reviewer,expertise_col].mean()

        #self.expertise = np.diag(self.expertise)

    def predict_rating_train(self, row, learning_rate, regularization_parameter):
        #prediction and SGD in one step. Only return prediction
        user = row[0]
        item = row[1]
        if user not in self.user_dict:
            b_u = 0
            p_u = np.zeros(self.hidden_factors)
            ex = 0
        else:
            user_ind = self.user_dict[user]
            b_u = self.user_bias[user_ind]
            p_u = self.user_hidden_factor[user_ind]
            ex = self.expertise[user_ind]

        if item not in self.item_dict:
            b_i = 0
            q_i = np.zeros(self.hidden_factors)
        else:
            item_ind = self.item_dict[item]
            b_i = self.item_bias[item_ind]
            q_i = self.item_hidden_factor[item_ind]

        prediction = self.global_mean + b_u + b_i + np.dot(q_i,ex*p_u)

        error = row[2] - prediction

        if user in self.user_dict:
            self.user_bias[user_ind] = b_u + learning_rate*(error - regularization_parameter*b_u)
            self.user_hidden_factor[user_ind] = ex*p_u + learning_rate*(error*q_i - regularization_parameter*ex*p_u)

        if item in self.item_dict:
            self.item_bias[item_ind] = b_i + learning_rate*(error - regularization_parameter*b_i)
            self.item_hidden_factor[item_ind] = ex*q_i + learning_rate*(error*p_u - regularization_parameter*q_i)
        
        if(prediction > 5):
            return 5
        elif(prediction < 1):
            return 1

        return prediction

    def predict_rating_test(self,row):
        user = row[0]
        item = row[1]
        if user not in self.user_dict:
            b_u = 0
            p_u = np.zeros(self.hidden_factors)
            ex = 0
        else:
            user_ind = self.user_dict[user]
            b_u = self.user_bias[user_ind]
            p_u = self.user_hidden_factor[user_ind]
            ex = self.expertise[user_ind]

        if item not in self.item_dict:
            b_i = 0
            q_i = np.zeros(self.hidden_factors)
        else:
            item_ind = self.item_dict[item]
            b_i = self.item_bias[item_ind]
            q_i = self.item_hidden_factor[item_ind]

        prediction = self.global_mean + b_u + b_i + np.dot(q_i,ex*p_u)

        if(prediction > 5):
            return 5
        elif(prediction < 1):
            return 1

        return prediction