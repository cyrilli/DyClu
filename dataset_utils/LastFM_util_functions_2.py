import pickle # Save model 
import matplotlib.pyplot as plt
import re 			# regular expression library
from random import random, choice 	# for random strategy
from operator import itemgetter
import numpy as np
from scipy.sparse import csgraph
from scipy.spatial import distance
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import TruncatedSVD
def generateUserFeature(W):
    svd = TruncatedSVD(n_components=25)
    result = svd.fit(W).transform(W)
    return result
def vectorize(M):
	temp = []
	for i in range(M.shape[0]*M.shape[1]):
		temp.append(M.T.item(i))
	V = np.asarray(temp)
	return V

def matrixize(V, C_dimension):
	temp = np.zeros(shape = (C_dimension, len(V)/C_dimension))
	for i in range(len(V)/C_dimension):
		temp.T[i] = V[i*C_dimension : (i+1)*C_dimension]
	W = temp
	return W

def readFeatureVectorFile(FeatureVectorsFileName):
    FeatureVectors = {}
    with open(FeatureVectorsFileName, 'r') as f:        
        f.readline()
        for line in f:
            line = line.split("\t")            
            vec = line[1].strip('[]').strip('\n').split(';')
            FeatureVectors[int(line[0])] = np.array(vec)
    return FeatureVectors

# This code simply reads one line from the source files of Yahoo!
def parseLine(line):
        userID, tim, pool_articles = line.split("\t")
        userID, tim = int(userID), int(tim)
        pool_articles = np.array(pool_articles.strip('[').strip(']').strip('\n').split(','))
        #print pool_articles
      
        '''
        tim, articleID, click = line[0].strip().split("")
        tim, articleID, click = int(tim), int(articleID), int(click)
        user_features = np.array([float(x.strip().split(':')[1]) for x in line[1].strip().split(' ')[1:]])
        
        pool_articles = [l.strip().split(" ") for l in line[2:]]
        pool_articles = np.array([[int(l[0])] + [float(x.split(':')[1]) for x in l[1:]] for l in pool_articles])
        '''
        return userID, tim, pool_articles

def save_to_file(fileNameWrite, recordedStats, tim):
    with open(fileNameWrite, 'a+') as f:
        f.write('data') # the observation line starts with data;
        f.write(',' + str(tim))
        f.write(',' + ';'.join([str(x) for x in recordedStats]))
        f.write('\n')