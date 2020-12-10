import pandas as pd
import numpy as np
import scipy
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
from sklearn.metrics.pairwise import cosine_similarity

class CustomKNN:

    def __init__(self,train_sample):
        self.train_sample = train_sample
        self.num_reviewers = len(train_sample['reviewerID'].unique())
        self.num_movies = len(train_sample['movieID'].unique())

        self.rev_dict = {key: index for index, key in enumerate(train_sample['reviewerID'].unique())}
        self.movie_dict = {key: index for index, key in enumerate(train_sample['movieID'].unique())}

        #build user-item matrix
        self.rev_mov = lil_matrix((self.num_reviewers, self.num_movies))
        for index, row in train_sample[['reviewerID','movieID','overall']].iterrows():
            rev = self.rev_dict[row['reviewerID']]
            mov = self.movie_dict[row['movieID']]
            self.rev_mov[rev,mov] = row['overall']
        
        self.rev_mov = csr_matrix(self.rev_mov)

    def train(self):
        self.global_mean = self.train_sample['overall'].mean()

        self.mean_vector = np.array([self.train_sample.loc[self.train_sample['reviewerID']==ele,'overall'].mean() for ele in self.train_sample['reviewerID'].unique()])
        self.cosine_similarity_matrix = cosine_similarity(self.rev_mov,self.rev_mov)
        np.fill_diagonal(self.cosine_similarity_matrix,-1)

    def test_real(self, row, k):

        if row['reviewerID'] not in self.rev_dict:
            return self.global_mean

        reviewer_ind = self.rev_dict[row['reviewerID']]
        mu = self.mean_vector[reviewer_ind]

        if row['movieID'] not in self.movie_dict:
            #return reviewer mean
            return mu

        movie_ind = self.movie_dict[row['movieID']]

        #at this point the reviewer and movie both have historical data.

        #k is the max number of neighbours who have reviewed this item to consider.

        #get neighbours who have reviewed item
        unis = self.train_sample.loc[self.train_sample['movieID']==row['movieID'],'reviewerID'].to_numpy()
        indexes = np.array([self.rev_dict[i] for i in unis])

        if(len(indexes)==0):
            print('No reviewers have reviewed the movie in dataset')

        #print("mu: "+str(mu))

        #numpy row vector
        neighbors = self.cosine_similarity_matrix[reviewer_ind]
        have_reviewed = neighbors[indexes]

        #these are have reviewed item AND similarity >0.
        grab = np.where(have_reviewed>0)[0]
        #print(grab)
        #print(len(grab))
        if(len(grab)==0):
            return mu

        refined_indexes = indexes[grab]
        #print(refined_indexes)

        #if(refined_indexes is None):
            #return mu
        
        #if(len(refined_indexes) == 0):
            #return mu


        if(len(refined_indexes) < k):
            k = len(refined_indexes)

        #print("K" + str(k))
        k_ins = np.argpartition(neighbors[refined_indexes], -k)[-k:]
        #print(k_ins)
        final_ins = refined_indexes[k_ins]

        neighbor_means = self.mean_vector[final_ins]
        historic_reviews = self.rev_mov[final_ins,movie_ind].toarray().flatten()

        #print("Diff: "+ str(diff))
        #print("Historic reviews: "+ str(historic_reviews))

        diff = historic_reviews - neighbor_means
        #print("Diff: "+ str(diff))
        num = np.dot(neighbors[final_ins],diff)
        denom = neighbors[final_ins].sum()
        #print(f"Final sims: "+str(neighbors[final_ins]))

        #print(f"num: {num}")
        #print(f"denom: {denom}")
        output = mu + num/denom
        if(output < 1):
            output = 1
        elif(output > 5):
            output = 5
        return output

    def test(self, row, k):

        if row['reviewerID'] not in self.rev_dict:
            return self.global_mean

        reviewer_ind = self.rev_dict[row['reviewerID']]
        mu = self.mean_vector[reviewer_ind]

        if row['movieID'] not in self.movie_dict:
            #return reviewer mean
            return mu

        movie_ind = self.movie_dict[row['movieID']]

        #at this point the reviewer and movie both have historical data.

        #k is the max number of neighbours who have reviewed this item to consider.

        #get neighbours who have reviewed item
        unis = self.train_sample.loc[self.train_sample['movieID']==row['movieID'],'reviewerID'].to_numpy()
        indexes = np.array([self.rev_dict[i] for i in unis])

        if(len(indexes)==0):
            print('No reviewers have reviewed the movie in dataset')

        #print("mu: "+str(mu))

        #numpy row vector
        neighbors = self.cosine_similarity_matrix[reviewer_ind]
        have_reviewed = neighbors[indexes]

        #these are have reviewed item AND similarity >0.
        grab = np.where(have_reviewed>0)[0]
        #print(grab)
        #print(len(grab))
        if(len(grab)==0):
            return mu

        refined_indexes = indexes[grab]
        #print(refined_indexes)

        #if(refined_indexes is None):
            #return mu
        
        #if(len(refined_indexes) == 0):
            #return mu


        if(len(refined_indexes) < k):
            k = len(refined_indexes)

        #print("K" + str(k))
        k_ins = np.argpartition(neighbors[refined_indexes], -k)[-k:]
        #print(k_ins)
        final_ins = refined_indexes[k_ins]

        neighbor_means = self.mean_vector[final_ins]
        historic_reviews = self.rev_mov[final_ins,movie_ind].toarray().flatten()

        #print("Diff: "+ str(diff))
        #print("Historic reviews: "+ str(historic_reviews))

        diff = historic_reviews - neighbor_means
        #print("Diff: "+ str(diff))
        num = np.dot(neighbors[final_ins],diff)
        denom = neighbors[final_ins].sum()
        #print(f"Final sims: "+str(neighbors[final_ins]))

        #print(f"num: {num}")
        #print(f"denom: {denom}")
        output = mu + num/denom
        return output


class Hypothesis1(CustomKNN):
    def __init__(self,train_sample):
        super().__init__(train_sample)

        self.expertise = np.empty(self.num_reviewers)

        for reviewer in self.train_sample['reviewerID'].unique():
            self.expertise[self.rev_dict[reviewer]] = train_sample.loc[train_sample['reviewerID']==reviewer,'expertise'].mean()

    def test_real(self, row, k, alpha):

        if row['reviewerID'] not in self.rev_dict:
            return self.global_mean

        reviewer_ind = self.rev_dict[row['reviewerID']]
        mu = self.mean_vector[reviewer_ind]

        if row['movieID'] not in self.movie_dict:
            #return reviewer mean
            return mu

        movie_ind = self.movie_dict[row['movieID']]

        #at this point the reviewer and movie both have historical data.

        #k is the max number of neighbours who have reviewed this item to consider.

        #get neighbours who have reviewed item
        unis = self.train_sample.loc[self.train_sample['movieID']==row['movieID'],'reviewerID'].to_numpy()
        indexes = np.array([self.rev_dict[i] for i in unis])

        if(len(indexes)==0):
            print('No reviewers have reviewed the movie in dataset')

        #print("mu: "+str(mu))

        #numpy row vector
        neighbors = self.cosine_similarity_matrix[reviewer_ind]
        have_reviewed = neighbors[indexes]

        #these are have reviewed item AND similarity >0.
        grab = np.where(have_reviewed>0)[0]
        #print(grab)
        #print(len(grab))
        if(len(grab)==0):
            return mu

        refined_indexes = indexes[grab]


        if(len(refined_indexes) < k):
            k = len(refined_indexes)

        #print("K" + str(k))
        k_ins = np.argpartition(neighbors[refined_indexes], -k)[-k:]
        #print(k_ins)
        final_ins = refined_indexes[k_ins]

        neighbor_means = self.mean_vector[final_ins]
        historic_reviews = self.rev_mov[final_ins,movie_ind].toarray().flatten()

        agg_exper = self.expertise[final_ins]

        block_vec = alpha*agg_exper + (1-alpha)*neighbors[final_ins]

        #print("Diff: "+ str(diff))
        #print("Historic reviews: "+ str(historic_reviews))

        diff = historic_reviews - neighbor_means
        #print("Diff: "+ str(diff))
        num = np.dot(block_vec,diff)
        denom = block_vec.sum()
        #print(f"Final sims: "+str(neighbors[final_ins]))

        #print(f"num: {num}")
        #print(f"denom: {denom}")
        output = mu + num/denom
        if(output < 1):
            output = 1
        elif(output > 5):
            output = 5
        return output

class Hypothesis2(CustomKNN):
    def __init__(self,train_sample):
        super().__init__(train_sample)

        self.expertise = np.empty(self.num_reviewers)

        for reviewer in self.train_sample['reviewerID'].unique():
            self.expertise[self.rev_dict[reviewer]] = train_sample.loc[train_sample['reviewerID']==reviewer,'expertise'].mean()

    def test_real(self, row, k, alpha):

        if row['reviewerID'] not in self.rev_dict:
            return self.global_mean

        reviewer_ind = self.rev_dict[row['reviewerID']]
        mu = self.mean_vector[reviewer_ind]

        if row['movieID'] not in self.movie_dict:
            #return reviewer mean
            return mu

        movie_ind = self.movie_dict[row['movieID']]

        #at this point the reviewer and movie both have historical data.

        #k is the max number of neighbours who have reviewed this item to consider.

        #get neighbours who have reviewed item
        unis = self.train_sample.loc[self.train_sample['movieID']==row['movieID'],'reviewerID'].to_numpy()
        indexes = np.array([self.rev_dict[i] for i in unis])

        if(len(indexes)==0):
            print('No reviewers have reviewed the movie in dataset')

        #print("mu: "+str(mu))

        #numpy row vector
        neighbors = self.cosine_similarity_matrix[reviewer_ind]
        have_reviewed = neighbors[indexes]

        #these are have reviewed item AND similarity >0.
        grab = np.where(have_reviewed>0)[0]
        #print(grab)
        #print(len(grab))
        if(len(grab)==0):
            return mu

        refined_indexes = indexes[grab]


        if(len(refined_indexes) < k):
            k = len(refined_indexes)

        #print("K" + str(k))
        k_ins = np.argpartition(neighbors[refined_indexes], -k)[-k:]
        #print(k_ins)
        final_ins = refined_indexes[k_ins]

        neighbor_means = self.mean_vector[final_ins]
        historic_reviews = self.rev_mov[final_ins,movie_ind].toarray().flatten()

        agg_exper = self.expertise[final_ins]

        rev_exp = self.expertise[reviewer_ind]
        rev_exp = rev_exp*np.ones(len(agg_exper))

        #1 will broadcast
        exp_dist = 1 - np.abs(rev_exp-agg_exper)

        block_vec = alpha*exp_dist + (1-alpha)*neighbors[final_ins]

        #print("Diff: "+ str(diff))
        #print("Historic reviews: "+ str(historic_reviews))

        diff = historic_reviews - neighbor_means
        #print("Diff: "+ str(diff))
        num = np.dot(block_vec,diff)
        denom = block_vec.sum()
        #print(f"Final sims: "+str(neighbors[final_ins]))

        #print(f"num: {num}")
        #print(f"denom: {denom}")
        output = mu + num/denom
        if(output < 1):
            output = 1
        elif(output > 5):
            output = 5
        return output