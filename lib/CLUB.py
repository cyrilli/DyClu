import numpy as np
import math
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix


class LinUCBUserStruct:
    def __init__(self, featureDimension, userID, lambda_, RankoneInverse=False):
        self.userID = userID
        self.A = lambda_ * np.identity(n=featureDimension)
        self.b = np.zeros(featureDimension)
        self.AInv = np.linalg.inv(self.A)
        self.UserTheta = np.zeros(featureDimension)
        self.RankoneInverse = RankoneInverse

    def updateParameters(self, articlePicked, click):
        featureVector = articlePicked.featureVector
        self.A += np.outer(featureVector, featureVector)
        self.b += featureVector * click
        if self.RankoneInverse:
            temp = np.dot(self.AInv, featureVector)
            self.AInv = self.AInv - (np.outer(temp, temp)) / (1.0 + np.dot(np.transpose(featureVector), temp))
        else:
            self.AInv = np.linalg.inv(self.A)
        self.UserTheta = np.dot(self.AInv, self.b)

    def getTheta(self):
        return self.UserTheta

    def getA(self):
        return self.A

    def getProb(self, alpha, users, article):
        featureVector = article.featureVector
        mean = np.dot(self.getTheta(), featureVector)
        var = np.sqrt(np.dot(np.dot(featureVector, np.linalg.inv(self.getA())), featureVector))
        pta = mean + alpha * var
        return pta

class CLUBUserStruct(LinUCBUserStruct):
    def __init__(self, featureDimension, lambda_, userID):
        LinUCBUserStruct.__init__(self, featureDimension=featureDimension, userID=userID, lambda_=lambda_)
        self.reward = 0
        self.CA = self.A
        self.Cb = self.b
        self.CAInv = np.linalg.inv(self.CA)
        self.CTheta = np.dot(self.CAInv, self.Cb)
        self.I = lambda_ * np.identity(n=featureDimension)
        self.counter = 0
        self.CBPrime = 0
        self.d = featureDimension

    def updateParameters(self, articlePicked_FeatureVector, click, alpha_2):
        # LinUCBUserStruct.updateParameters(self, articlePicked_FeatureVector, click)
        # alpha_2 = 1
        if type(articlePicked_FeatureVector) != np.ndarray:
            articlePicked_FeatureVector = articlePicked_FeatureVector.contextFeatureVector[:self.d]
        self.A += np.outer(articlePicked_FeatureVector, articlePicked_FeatureVector)
        self.b += articlePicked_FeatureVector * click
        self.AInv = np.linalg.inv(self.A)
        self.UserTheta = np.dot(self.AInv, self.b)
        self.counter += 1
        self.CBPrime = alpha_2 * np.sqrt(float(1 + math.log10(1 + self.counter)) / float(1 + self.counter))

    def updateParametersofClusters(self, clusters, userID, Graph, users):
        self.CA = self.I
        self.Cb = np.zeros(self.d)
        # print type(clusters)

        for i in range(len(clusters)):
            if clusters[i] == clusters[userID]:
                self.CA += float(Graph[userID, i]) * (users[i].A - self.I)
                self.Cb += float(Graph[userID, i]) * users[i].b
        self.CAInv = np.linalg.inv(self.CA)
        self.CTheta = np.dot(self.CAInv, self.Cb)

    def getProb(self, alpha, article_FeatureVector, time):
        mean = np.dot(self.CTheta, article_FeatureVector)
        var = np.sqrt(np.dot(np.dot(article_FeatureVector, self.CAInv), article_FeatureVector))
        pta = mean + alpha * var * np.sqrt(math.log10(time + 1))
        return pta


class LinUCBAlgorithm:
    def __init__(self, dimension, alpha, lambda_, n, RankoneInverse=False):  # n is number of users
        self.users = []
        # algorithm have n users, each user has a user structure
        for i in range(n):
            self.users.append(LinUCBUserStruct(dimension, i, lambda_, RankoneInverse))

        self.dimension = dimension
        self.alpha = alpha

        self.CanEstimateCoUserPreference = True
        self.CanEstimateUserPreference = False
        self.CanEstimateW = False

    def decide(self, pool_articles, userID):
        maxPTA = float('-inf')
        articlePicked = None

        for x in pool_articles:
            x_pta = self.users[userID].getProb(self.alpha, self.users, x)
            # pick article with highest Prob
            if maxPTA < x_pta:
                articlePicked = x
                maxPTA = x_pta

        return articlePicked

    def updateParameters(self, articlePicked, click, userID):
        self.users[userID].updateParameters(articlePicked, click)

    def getCoTheta(self, userID):
        return self.users[userID].UserTheta

class CLUBAlgorithm(LinUCBAlgorithm):
    def __init__(self, dimension, alpha, lambda_, n, alpha_2, cluster_init="Complete"):
        self.time = 0
        # N_LinUCBAlgorithm.__init__(dimension = dimension, alpha=alpha,lambda_ = lambda_,n=n)
        self.users = []
        # algorithm have n users, each user has a user structure
        for i in range(n):
            self.users.append(CLUBUserStruct(dimension, lambda_, i))

        self.dimension = dimension
        self.alpha = alpha
        self.alpha_2 = alpha_2
        if (cluster_init == "Erdos-Renyi"):
            p = 3 * math.log(n) / n
            self.Graph = np.random.choice([0, 1], size=(n, n), p=[1 - p, p])
            self.clusters = []
            g = csr_matrix(self.Graph)
            N_components, components = connected_components(g)
        else:
            self.Graph = np.ones([n, n])
            self.clusters = []
            g = csr_matrix(self.Graph)
            N_components, components = connected_components(g)

        self.CanEstimateCoUserPreference = False
        self.CanEstimateUserPreference = False
        self.CanEstimateUserCluster = False
        self.CanEstimateW = False

        self.userID2userIndex = {}
        self.user_counter = 0

    def decide(self, pool_articles, userID):
        if userID not in self.userID2userIndex:
            self.userID2userIndex[userID] = self.user_counter
            self.user_counter += 1
        userID = self.userID2userIndex[userID]
        self.users[userID].updateParametersofClusters(self.clusters, userID, self.Graph, self.users)
        maxPTA = float('-inf')

        for x in pool_articles:
            x_pta = self.users[userID].getProb(self.alpha, x.featureVector, self.time)
            # pick article with highest Prob
            if maxPTA < x_pta:
                featureVectorPicked = x.featureVector
                picked = x
                maxPTA = x_pta
        self.time += 1

        return picked

    def updateParameters(self, featureVector, click, userID):
        userID = self.userID2userIndex[userID]
        self.users[userID].updateParameters(featureVector, click, self.alpha_2)

    def updateGraphClusters(self, userID, binaryRatio):
        userID = self.userID2userIndex[userID]
        n = len(self.users)
        for j in range(n):
            ratio = float(np.linalg.norm(self.users[userID].UserTheta - self.users[j].UserTheta, 2)) / float(
                self.users[userID].CBPrime + self.users[j].CBPrime)
            # print float(np.linalg.norm(self.users[userID].UserTheta - self.users[j].UserTheta,2)),'R', ratio
            if ratio > 1:
                ratio = 0
            elif binaryRatio == 'True':
                ratio = 1
            elif binaryRatio == 'False':
                ratio = 1.0 / math.exp(ratio)
            # print 'ratio',ratio
            self.Graph[userID][j] = ratio
            self.Graph[j][userID] = self.Graph[userID][j]
        N_components, component_list = connected_components(csr_matrix(self.Graph))
        # print 'N_components:',N_components
        self.clusters = component_list
        return N_components

    def getLearntParameters(self, userID):
        userID = self.userID2userIndex[userID]
        return self.users[userID].UserTheta
