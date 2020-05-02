#!/usr/bin/env python
# coding: utf-8

# In[286]:


import numpy as np
import pandas as pd
import matplotlib
import math
from sklearn.preprocessing import StandardScaler as sc
from sklearn.decomposition import PCA as pca
from scipy.sparse.linalg import svds


# In[287]:


test_data = pd.read_csv("1571391072_1720269_test.dat", sep="\n", header=None)


# In[288]:


test_df = test_data[0].str.split(" ",n=1,expand =  True)


# In[289]:


test_df = test_df.drop([0])


# In[290]:


test_df.columns = ['userID','movieID']


# In[291]:


train_data = pd.read_csv('train.txt', sep=" ", header=None)


# In[292]:


train_data = train_data.drop([0])


# In[293]:


train_data = train_data.reset_index(drop=True)


# In[294]:


train_data.columns = ['userID','movieID','rating']


# In[295]:


movie_genres_data = pd.read_csv("movie_genres.txt", sep="\t", header=None)


# In[296]:


movie_genres_data = movie_genres_data.drop([0])
movie_genres_data = movie_genres_data.reset_index(drop=True)
movie_genres_data.columns = ['movieID','genre']


# In[297]:


movieID_List = movie_genres_data.movieID.unique() 


# In[298]:


userID_List = train_data.userID.unique()


# In[299]:


# user_movie_crossplosion = train_data.pivot(index = 'userID', columns ='movieID', values = 'rating')


# In[300]:


#Initialize the utility matrix that will hold the known movie ratings for each user
#rows are users, columns are movies
user_movie_util_df = pd.DataFrame(index=userID_List, columns=movieID_List)
user_movie_util_df = user_movie_util_df.fillna(0)


# In[301]:


# This function fills in all of the known movie ratings for each user
def fill_ratings(row):
    incUserID = row['userID']
    incMovieID = row['movieID']
    incRating = row['rating']
    user_movie_util_df.at[str(incUserID),str(incMovieID)] = float(incRating)
    return


# In[302]:


train_data.apply(fill_ratings, axis=1)


# In[303]:


# Need to find the average rating that a user gives in order to account for biases
means = []
x = float('nan')
def create_user_ratings_means(row):
    sum = 0
    mean = 0
    count = 0
    i = 0
    for i in range(len(row)):
        if row[i] != 0 and not math.isnan(row[i]):
            count +=1
            sum += row[i]
    if(count == 0):
        #Case where user has no ratings already, probably a new user
        mean = 0
    else:
        mean = sum/count
    means.append(mean)
    return
user_movie_util_df.apply(create_user_ratings_means, axis=1)


# In[304]:


user_movie_util_df = user_movie_util_df.fillna(0)


# In[305]:


#Convert utility dataframe to a matrix for matrix operations
util_matrix = user_movie_util_df.as_matrix()


# In[306]:


#Recalculation of means, had odd issue where the expected number of averages did not meet the number of users
means = []
user_movie_util_df.apply(create_user_ratings_means, axis=1)
util_matrix_mean = np.asarray(means)


# In[307]:


util_matrix_mean = np.asarray(means)
# subtract the mean of the user's ratings from each of that user's ratings
util_adjusted = user_movie_util_df.sub(util_matrix_mean,axis=0)


# In[308]:


#SVD is using a matrix as a linear transformation of the data
u, sigma, vt = svds(util_adjusted, k = 50)
sigma = np.diag(sigma)


# In[309]:


complete_ratings = np.dot(np.dot(u, sigma), vt) + util_matrix_mean.reshape(-1, 1)
predictions_df = pd.DataFrame(complete_ratings, columns = user_movie_util_df.columns)


# In[310]:


test_ratings = []
userIndex = user_movie_util_df.index.values.tolist()
predictions_df.index = userIndex

def make_test_ratings(row):
    incUserID = row['userID']
    incMovieID = row['movieID']
    try:
        test_ratings.append(predictions_df.at[incUserID,incMovieID])
        return
    except:
        test_ratings.append(0)
        return
    return
test_df.apply(make_test_ratings, axis=1)


# In[311]:


test_ratings


# In[312]:


def create_ratings_file():
    f= open("HW4_ratings.dat","w+")
    for rating in test_ratings:
        f.write(str(rating)+"\n")
    return

create_ratings_file()


# In[313]:


len(test_ratings)


# In[ ]:




