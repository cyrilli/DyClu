import copy
import numpy as np
import random
from random import sample, shuffle, choice
from scipy.sparse import csgraph
import datetime
import os.path
import matplotlib.pyplot as plt
import argparse
from sklearn.decomposition import TruncatedSVD
from sklearn import cluster
from sklearn.decomposition import PCA
# local address to save simulated users, simulated articles, and results
from conf import sim_files_folder, save_address
from util_functions import featureUniform, gaussianFeature, get_new_theta, save_stat
from Articles_nonstationary import ArticleManager
from Users_nonstationary import UserManager
#Stationary Bandit algorithms
from lib.LinUCB import N_LinUCBAlgorithm, Uniform_LinUCBAlgorithm,Hybrid_LinUCBAlgorithm, N_LinUCBAlgorithm_restart, N_LinUCBAlgorithm_oracle
from lib.hLinUCB import HLinUCBAlgorithm
from lib.factorUCB import FactorUCBAlgorithm
from lib.CoLin import AsyCoLinUCBAlgorithm
from lib.CLUB import *
from lib.PTS import PTSAlgorithm
from lib.UCBPMF import UCBPMFAlgorithm
from lib.UCB1 import UCB1Algorithm
#nonStationary Bandit algorithms
#nonStationary Bandit baselines
from lib.WMDUCB1 import WMDUCB1Algorithm
from lib.AdaptiveThompson import AdaptiveThompsonAlgorithm
from lib.MetaBandit import  MetaBanditAlgorithm
from lib.dLinUCB import dLinUCBAlgorithm, dLinUCBAlgorithm_noDiscard, dLinUCBAlgorithm_share, dLinUCBAlgorithm_AlwaysShare
from lib.ContextDependent_dLinUCB_update import ContextDependent_dLinUCBAlgorithm_update, ContextDependent_dLinUCBAlgorithm_update_avg
from lib.DenBand import *
from lib.DenBand_update import *
from lib.Den_Oracle import *
from lib.DenBand_onlyUpdateNew import *
from lib.DenBand_OracelUpdate import *
from lib.DenBand_OracleReuse import *
from lib.DenBand_share import ContextDependent_dLinUCBAlgorithm_frozen
#from lib.DenBand_share import ContextDependent_dLinUCBAlgorithm_frozen
from lib.dTS import dTSAlgorithm
from lib.dTS_sampling import dTSAlgorithm_Sample
from lib.dLinUCB_likelihood import dLinUCB_LikelihoodAlgorithm, dLinUCBAlgorithm_likelihoodShare, dLinUCBAlgorithm_likelihoodShare_POP, dLinUCBAlgorithm_likelihoodShare_POP_ALL, dLinUCBAlgorithm_likelihood_eliminate, dLinUCBAlgorithm_likelihood_new
from lib.dLinUCB_DP import dLinUCB_DP
from lib.CoDBand import CoDBand, CoDBand_bak, CoDBand_NonPop
from lib.dLinUCB_DP_DP import dLinUCB_DP_Log_DP
from lib.dLinUCB_DP_Log_Reset import dLinUCB_DP_Log_Reset, dLinUCB_DP_Log_bak_Reset
from lib.dLinUCB_DP_Log_bak import dLinUCB_DP_Log_org
from lib.dLinUCB_DP_Log_Sample import dLinUCB_DP_Log_S
from lib.dTS_DP import dTS_DP_Log
from lib.TS import TSAlgorithm, TSAlgorithm_M, TSAlgorithm_UCB
from lib.dLogUCB_DP_Thres import CoDBand_thres
from lib.Lin_func import *

import pickle
#python simu_generative.py --alg dLinUCBs --stationary_ratio 0.5 --minWindow 50 --sigma_0 0.1 --sigma 0.001

#Command line: python simu_nonstationary.py  --alg dLinUCBs --stationary_ratio 0.3
class simulateOnlineData(object):
	def __init__(self, namelabel, context_dimension, latent_dimension, training_iterations, testing_iterations, testing_method, plot, articles_dic, users, 
					batchSize = 1,
					noise = lambda : 0,
					matrixNoise = lambda:0,
					type_ = 'UniformTheta', 
					signature = '', 
					poolArticleSize = 10, 
					NoiseScale = 0,
					sparseLevel = 0,  
					epsilon = 1, Gepsilon = 1, stationary_ratio = None):

		self.simulation_signature = signature
		self.type = type_

		self.context_dimension = context_dimension
		self.latent_dimension = latent_dimension
		self.training_iterations = training_iterations
		self.testing_iterations = testing_iterations
		self.testing_method = testing_method
		self.plot = plot

		self.noise = noise
		self.matrixNoise = matrixNoise # noise to be added to W
		self.NoiseScale = NoiseScale
		
		#self.articles = articles 
		self.articles_dic = articles_dic
		self.users = users
		self.sparseLevel = sparseLevel

		self.poolArticleSize = poolArticleSize
		self.batchSize = batchSize
		
		#self.W = self.initializeW(epsilon)
		#self.GW = self.initializeGW(Gepsilon)
		self.W, self.W0 = self.constructAdjMatrix(sparseLevel)
		W = self.W.copy()
		self.GW = self.constructLaplacianMatrix(W, Gepsilon)
		self.articlePool = []
		self.stationary_ratio = stationary_ratio

		self.change_num = 0
		self.optimal_arm_changeSensitiveRatio = 0.0
		
	def constructGraph(self):
		n = len(self.users)	

		G = np.zeros(shape = (n, n))
		for ui in self.users:
			for uj in self.users:
				G[ui.id][uj.id] = np.dot(ui.theta, uj.theta) # is dot product sufficient
		return G
		
	def constructAdjMatrix(self, m):
		n = len(self.users)	

		G = self.constructGraph()
		W = np.zeros(shape = (n, n))
		W0 = np.zeros(shape = (n, n)) # corrupt version of W
		for ui in self.users:
			for uj in self.users:
				W[ui.id][uj.id] = G[ui.id][uj.id]
				sim = W[ui.id][uj.id] + self.matrixNoise() # corrupt W with noise
				if sim < 0:
					sim = 0
				W0[ui.id][uj.id] = sim
				
			# find out the top M similar users in G
			if m>0 and m<n:
				similarity = sorted(G[ui.id], reverse=True)
				threshold = similarity[m]				
				
				# trim the graph
				for i in range(n):
					if G[ui.id][i] <= threshold:
						W[ui.id][i] = 0;
						W0[ui.id][i] = 0;
					
			W[ui.id] /= sum(W[ui.id])
			W0[ui.id] /= sum(W0[ui.id])

		return [W, W0]

	def constructLaplacianMatrix(self, W, Gepsilon):
		G = W.copy()
		#Convert adjacency matrix of weighted graph to adjacency matrix of unweighted graph
		for i in self.users:
			for j in self.users:
				if G[i.id][j.id] > 0:
					G[i.id][j.id] = 1	

		L = csgraph.laplacian(G, normed = False)
		print L
		I = np.identity(n = G.shape[0])
		GW = I + Gepsilon*L  # W is a double stochastic matrix
		print 'GW', GW
		return GW.T

	def getW(self):
		return self.W
	def getW0(self):
		return self.W0
	def getFullW(self):
		return self.FullW
	
	def getGW(self):
		return self.GW

	def getTheta(self):
		Theta = np.zeros(shape = (self.dimension, len(self.users)))
		for i in range(len(self.users)):
			Theta.T[i] = self.users[i].theta
		return Theta
	def generateUserFeature(self,W):
		svd = TruncatedSVD(n_components=20)
		result = svd.fit(W).transform(W)
		return result

	def CoTheta(self):
		for ui in self.users:
			ui.CoTheta = np.zeros(self.context_dimension+self.latent_dimension)
			for uj in self.users:
				ui.CoTheta += self.W[uj.id][ui.id] * np.asarray(uj.theta)
			print 'Users', ui.id, 'CoTheta', ui.CoTheta	

	def getClick(self, theta, featureVector, noise):
		reward = np.dot(theta, featureVector) + noise
		clickProb = self.sigmoid(reward)
		randomNum = random.uniform(0,1)
		#click_threshold = 0.7
		
		if (randomNum )<= clickProb:
			click = 1
		else:
			click = 0
		return click  #Binary

	def batchRecord(self, iter_):
		#pass
		print "Iteration %d"%iter_, "Pool", len(self.articlePool)," Elapsed time", datetime.datetime.now() - self.startTime, "Iteration %d"%iter_

	def regulateArticlePool(self, user_id = 0):

		# Randomly generate articles
		#self.articlePool = sample(self.articles, self.poolArticleSize)   
		
		self.articlePool = []
		
		stationary_arm_num = int(round(1.0*self.poolArticleSize*self.stationary_ratio/self.change_num)) if self.change_num > 0 else 0
		nonstationary_arm_num = self.poolArticleSize - stationary_arm_num*self.change_num

		# print stationary_arm_num, nonstationary_arm_num, len(self.articles_dic[user_id]['nonstationary'])

		# stationary_pool = sample(self.articles_dic[user_id]['stationary'], stationary_arm_num)
		nonstationary_pool = sample(self.articles_dic[user_id]['nonstationary'][self.change_num-1 if self.change_num > 0 else 0], nonstationary_arm_num)

		stationary_pool = []
		for i in range(self.change_num):
			stationary_pool.extend(sample(self.articles_dic[user_id]['stationary'][self.change_num][self.change_num-1], stationary_arm_num))

		# u = self.users[0]
		# old_theta = u.theta
		# new_theta = u.new_theta
		# for a in stationary_pool:
		# 	print 'stationary_pool' , a.id
		# 	print np.dot((new_theta- old_theta ), a.featureVector)
		# for a in nonstationary_pool:
		# 	print 'non', a.id
			
		# 	print np.dot((new_theta- old_theta ), a.featureVector)
		


		self.articlePool = nonstationary_pool + stationary_pool
		# print (typeof(self.articlePool[0].id))
		shuffle(self.articlePool)
		'''
		random_num = np.random.uniform(0,1)
		if random_num <= stationary_ratio:
			self.articlePool = sample(self.articles_dic[user_id]['stationary'], self.poolArticleSize)
		else:
			self.articlePool = sample(self.articles_dic[user_id]['nonstationary'], self.poolArticleSize)
		shuffle(self.articlePool)
		'''

		# print len(self.articlePool)
		

	def getReward(self, user, pickedArticle):
		return np.dot(user.theta, pickedArticle.featureVector)

	def GetOptimalReward(self, user, articlePool):		
		maxReward = float('-inf')
		maxx = None
		for x in articlePool:	 
			reward = self.getReward(user, x)
			#print reward, maxReward
			if reward > maxReward:
				maxReward = reward
				maxx = x

		return maxReward, x
	
	def getL2Diff(self, x, y):
		return np.linalg.norm(x-y) # L2 norm

	def runAlgorithms(self, algorithms):
		SAVE = True
		self.startTime = datetime.datetime.now()
		timeRun = self.startTime.strftime('_%m_%d_%H_%M') 
		filenameWriteRegret = os.path.join(save_address, 'AccRegret' + timeRun + '.csv')
		filenameWritePara = os.path.join(save_address, 'ParameterEstimation' + timeRun + '.csv')
		filenameWrite_actualChanges = os.path.join(save_address, 'ActualChanges_List' + timeRun + '.list')
		filenameWrite_DenBandDetectedChanges = os.path.join(save_address, 'DetectedChanges_List' + timeRun + '.list')


		# compute co-theta for every user
		self.CoTheta()
		TrueThetaList = []

		tim_ = []
		BatchCumlateRegret = {}
		AlgRegret = {}

		BatchCumlateRegret_Beta = {}
		AlgRegret_Beta = {}

		ThetaDiffList = {}
		BetaDiffList = {}
		CoThetaDiffList = {}
		WDiffList = {}
		VDiffList = {}
		CoThetaVDiffList = {}
		RDiffList ={}
		RVDiffList = {}

		ThetaDiff = {}
		BetaDiff = {}
		CoThetaDiff = {}
		WDiff = {}
		VDiff = {}
		CoThetaVDiff = {}
		RDiff ={}
		RVDiff = {}

		Var = {}
		
		# Initialization
		userSize = len(self.users)
		for alg_name, alg in algorithms.items():
			AlgRegret[alg_name] = []
			BatchCumlateRegret[alg_name] = []

			AlgRegret_Beta[alg_name] = []
			BatchCumlateRegret_Beta[alg_name] = []


			if alg.CanEstimateUserPreference:
				ThetaDiffList[alg_name] = []
			if alg.CanEstimateBeta:
				BetaDiffList[alg_name] = []
			if alg.CanEstimateCoUserPreference:
				CoThetaDiffList[alg_name] = []
			if alg.CanEstimateW:
				WDiffList[alg_name] = []
			if alg.CanEstimateV:
				VDiffList[alg_name] = []
				CoThetaVDiffList[alg_name] = []
				RVDiffList[alg_name] = []
				RDiffList[alg_name] = []
			Var[alg_name] = []

		if SAVE:
			with open(filenameWriteRegret, 'w') as f:
				f.write('Time(Iteration)')
				f.write(',' + ','.join( [str(alg_name) for alg_name in algorithms.iterkeys()]))
				f.write('\n')
		
		# with open(filenameWritePara, 'w') as f:
		# 	f.write('Time(Iteration)')
		# 	f.write(',' + ','.join([str(alg_name)+'CoTheta' for alg_name in CoThetaDiffList.iterkeys()]))
		# 	f.write(','+ ','.join([str(alg_name)+'Theta' for alg_name in ThetaDiffList.iterkeys()]))
		# 	f.write(','+ ','.join([str(alg_name)+'W' for alg_name in WDiffList.iterkeys()]))
		# 	f.write(','+ ','.join([str(alg_name)+'V' for alg_name in VDiffList.iterkeys()]))
		# 	f.write(',' + ','.join([str(alg_name)+'CoThetaV' for alg_name in CoThetaVDiffList.iterkeys()]))
		# 	f.write(','+ ','.join([str(alg_name)+'R' for alg_name in RDiffList.iterkeys()]))
		# 	f.write(','+ ','.join([str(alg_name)+'RV' for alg_name in RVDiffList.iterkeys()]))
		# 	f.write('\n')
		
		print(algorithms)

		# Training
		#shuffle(self.articles)
		for iter_ in range(self.training_iterations):
			article = self.articles[iter_]										
			for u in self.users:
				noise = self.noise()	
				reward = self.getReward(u, article)
				reward += noise										
				for alg_name, alg in algorithms.items():
					alg.updateParameters(article, reward, u.id)	

			if 'syncCoLinUCB' in algorithms:
				algorithms['syncCoLinUCB'].LateUpdate()	

		#Testing
		actual_changes = [0]
		actual_changes_fix = [0]
		actual_changes_value = {}
		ThetaList = {}
		ThetaList_index = {}
		actual_changes_dic = {}
		change_schedule = {}
		user_count = {}

		for u in self.users:
			actual_changes_value[u.id] = [1]
			ThetaList[u.id] = [u.theta]
			ThetaList_index[u.id] = []

			actual_changes_dic[u.id] = []
			actual_changes_dic[u.id].append(0)

			change_schedule[u.id] = 200 + 50*int(u.id)  #200 +
			#change_schedule[u.id] = 1000
			user_count[u.id] = 0

		TrueThetaList.append(self.users[0].theta)

		opt_stationary = 0
		opt_nonstationary = 0
		alg_stationary = 0
		alg_nonstationary= 0

		theta_list = [self.users[0].theta]
		gamma = 3.00
		for u in self.users:
			self.users[u.id].theta = theta_list[0]
		#theta_list.append(self.users[0].theta)
		popularity_list = [random.uniform(0,1)]
		# for i in range(len(theta_list)):
		# 	popularity_list.append(random.uniform(0,1))
		usedEnvironment_id = {}
		usedEnvironment_id[0] = 1
		usedEnvironment_list = [0]

		for iter_ in range(self.testing_iterations):
			noise = self.noise()


			for alg_name, alg in algorithms.items():
				if alg.CanEstimateUserPreference:
					ThetaDiff[alg_name] = 0
				if alg.CanEstimateBeta:
					BetaDiff[alg_name] = 0
				if alg.CanEstimateCoUserPreference:
					CoThetaDiff[alg_name] = 0
				if alg.CanEstimateW:
					WDiff[alg_name] = 0
				if alg.CanEstimateV:
					VDiff[alg_name]	= 0	
					CoThetaVDiff[alg_name] = 0	
					RVDiff[alg_name]	= 0	
				RDiff[alg_name]	= 0	

			#if iter_ % 800 == 0 and iter_ > 0:
			change_flag = False
			user_change_flag = {}
			for u in self.users:
				user_change_flag[u.id] = False
			
			for u in self.users:
			#u = choice(self.users)
				user_count[u.id] +=1
				#if iter_ > (actual_changes[-1] + 200 + (len(actual_changes)*50)):	
				#if iter_ > (actual_changes[-1] + 300):
				
				if user_count[u.id] > (actual_changes_dic[u.id][-1] + change_schedule[u.id]):
					#Update Theta Popularity
					for m in range(len(theta_list)):
						for temp_u in self.users:
							if np.array_equal(theta_list[m], temp_u.theta):
								popularity_list[m] +=1

					opt_stationary = 0
					opt_nonstationary = 0
					alg_stationary = 0
					alg_nonstationary= 0
					actual_changes_fix.append(iter_)
					# for u in self.users:
					# 	#new_theta = get_new_theta(self.context_dimension, u.theta, 0.9)
					# 	# new_theta = u.new_theta
					# 	# u.theta = new_theta
					# 	u.theta = u.theta_hist[self.change_num]
					# 	print u.theta
					# 	TrueThetaList.append(u.theta)
					# 	#actual_changes_value[u.id].append(np.linalg.norm( old_theta - u.theta))
					# 	actual_changes_value[u.id].append(1)
					old_theta = u.theta
					#Sample new theta
					random_num = random.uniform(0,1)
					ChoseOld_Flag = False
					change_to_id = None
					chosen_index = None
					for i in range(len(theta_list)):
						popu = sum(popularity_list[:i])/(sum(popularity_list) + gamma)
						if random_num <= popu:
							change_to_id = i
							new_theta = theta_list[i]
							ChoseOld_Flag = True
							#popularity_list[i] = popularity_list[i] + gamma
							usedEnvironment_list.append(i)
							if i not in usedEnvironment_id:
								usedEnvironment_id[i] = 1
							else:
								usedEnvironment_id[i] +=1
							actual_changes_dic[u.id].append(user_count[u.id])
							chosen_index = i
							break

					if not ChoseOld_Flag:
						new_theta = get_new_theta(self.context_dimension, old_theta, 0.9)
						theta_list.append(new_theta)
						popularity_list.append(gamma)
						actual_changes_dic[u.id].append(user_count[u.id])
						usedEnvironment_id[len(theta_list)-1] = 1
						usedEnvironment_list.append(len(theta_list)-1)
						chosen_index = len(theta_list)-1

					if not np.array_equal(new_theta, old_theta):
						actual_changes.append(iter_)
						change_flag = True
						self.change_num += 1
						user_change_flag[u.id] = True


					self.users[u.id].theta = new_theta
					ThetaList[u.id].append(new_theta)
					ThetaList_index[u.id].append(chosen_index)
					#actual_changes_global.append(iter_)
					#actual_changes_dic[u.id].append(iter_)
				self.regulateArticlePool(u.id) # select random articles

				#noise = self.noise()
				#get optimal reward for user x at time t
				OptimalReward, OptimalArticle = self.GetOptimalReward(u, self.articlePool) 
				if OptimalArticle.stationary == True:
					opt_stationary += 1
				else:
					opt_nonstationary += 1
					self.optimal_arm_changeSensitiveRatio +=1
				OptimalReward += noise
							
				for alg_name, alg in algorithms.items():

					if 'DenBandit' in alg_name:
						pickedArticle = alg.decide(self.articlePool, u.id, u.theta,  user_change_flag[u.id])
						#change_flag = False
					elif 'restart' in alg_name  or 'Oracle' in alg_name:
						pickedArticle = alg.decide(self.articlePool, u.id,  user_change_flag[u.id])
					else:
						pickedArticle = alg.decide(self.articlePool, u.id)
					if 'c_dLinUCB' in alg_name or 'DenBand' in alg_name:
						if pickedArticle.stationary == True:
							alg_stationary += 1
						else:
							alg_nonstationary += 1
					# print (alg_stationary, alg_nonstationary)
					reward = self.getReward(u, pickedArticle) + noise
					# print pickedArticle.id, reward, OptimalArticle.id, OptimalReward

					if (self.testing_method=="online"): # for batch test, do not update while testing
						if 'DenBandit_Oracle' in alg_name:
							alg.updateParameters(pickedArticle, reward, u.id, u.theta)
						else:
							alg.updateParameters(pickedArticle, reward, u.id)
						if alg_name =='CLUB':
							n_components= alg.updateGraphClusters(u.id,'False')

					regret = OptimalReward - reward
					if 'CoDBan_WeightedUpdate' in alg_name:
						fileNameWrite_stat = os.path.join(save_address +'/stat/', 'Stat' + timeRun + str(u.id) + '.txt')  
						save_stat(fileNameWrite_stat, iter_, alg.MASTER_Global.SLAVEs_prob_all)


					#print regret
					AlgRegret[alg_name].append(regret)
					if alg.CanEstimateBeta:
						predicted_reward = np.dot(alg.getTheta(u.id), pickedArticle.featureVector)
						prediction_error = abs(predicted_reward - reward)
						estimated_prediction_error = np.dot(alg.getBeta(u.id), pickedArticle.featureVector)
						#estimated_prediction_error = 0
						beta_regret = abs(prediction_error - estimated_prediction_error )
						AlgRegret_Beta[alg_name].append(beta_regret)

					#update parameter estimation record
					if alg.CanEstimateUserPreference:
						#ThetaDiff[alg_name] += self.getL2Diff(u.theta, alg.getTheta(u.id))
						ThetaDiff[alg_name] = alg.getLikeli()

					if alg.CanEstimateBeta:
						BetaDiff[alg_name] += self.getL2Diff(u.theta- alg.getTheta(u.id), alg.getBeta(u.id))
					if alg.CanEstimateCoUserPreference:
						CoThetaDiff[alg_name] += self.getL2Diff(u.CoTheta[:self.context_dimension], alg.getCoTheta(u.id)[:self.context_dimension])
					if alg.CanEstimateW:
						WDiff[alg_name] += self.getL2Diff(self.W.T[u.id], alg.getW(u.id))	
					if alg.CanEstimateV:
						VDiff[alg_name]	+= self.getL2Diff(self.articles[pickedArticle.id].featureVector, alg.getV(pickedArticle.id))
						CoThetaVDiff[alg_name]	+= self.getL2Diff(u.CoTheta[self.context_dimension:], alg.getCoTheta(u.id)[self.context_dimension:])
						RVDiff[alg_name] += abs(u.CoTheta[self.context_dimension:].dot(self.articles[pickedArticle.id].featureVector[self.context_dimension:]) - alg.getCoTheta(u.id)[self.context_dimension:].dot(alg.getV(pickedArticle.id)[self.context_dimension:]))
						RDiff[alg_name] += reward-noise -  alg.getCoTheta(u.id).dot(alg.getV(pickedArticle.id))
			
			for alg_name, alg in algorithms.items():
				if alg.CanEstimateUserPreference:
					ThetaDiffList[alg_name] += [ThetaDiff[alg_name]/userSize]
					print len(ThetaDiffList[alg_name])
				if alg.CanEstimateBeta:
					BetaDiffList[alg_name] += [BetaDiff[alg_name]/userSize]
				if alg.CanEstimateCoUserPreference:
					CoThetaDiffList[alg_name] += [CoThetaDiff[alg_name]/userSize]
				if alg.CanEstimateW:
					WDiffList[alg_name] += [WDiff[alg_name]/userSize]	
				if alg.CanEstimateV:
					VDiffList[alg_name] += [VDiff[alg_name]/userSize]	
					CoThetaVDiffList[alg_name] += [CoThetaVDiff[alg_name]/userSize]
					RVDiffList[alg_name] += [RVDiff[alg_name]/userSize]
					RDiffList[alg_name] += [RDiff[alg_name]/userSize]				
			if iter_%self.batchSize == 0:
				self.batchRecord(iter_)
				tim_.append(iter_)
				for alg_name, alg in algorithms.items():
					BatchCumlateRegret[alg_name].append(sum(AlgRegret[alg_name]))
					if alg.CanEstimateBeta:
						BatchCumlateRegret_Beta[alg_name].append(sum(AlgRegret_Beta[alg_name]))

				if SAVE:
					with open(filenameWriteRegret, 'a+') as f:
						f.write(str(iter_))
						f.write(',' + ','.join([str(BatchCumlateRegret[alg_name][-1]) for alg_name in algorithms.iterkeys()]))
						f.write('\n')
				# with open(filenameWritePara, 'a+') as f:
				# 	f.write(str(iter_))
				# 	f.write(',' + ','.join([str(CoThetaDiffList[alg_name][-1]) for alg_name in CoThetaDiffList.iterkeys()]))
				# 	f.write(','+ ','.join([str(ThetaDiffList[alg_name][-1]) for alg_name in ThetaDiffList.iterkeys()]))
				# 	f.write('\n')


		print("Actual changes: " + str(actual_changes))
		print 'OptimalArticle Non-stationary Ratio', opt_nonstationary/float(testing_iterations)

		for alg_name in algorithms.iterkeys():
			if 'dLinUCB' in alg_name:
				print alg_name,'Switch Points:', str(algorithms[alg_name].users[0].SwitchPoints)
				#print alg_name, 'Switch Points UCB', algorithms[alg_name].users[0].SwitchPoints_UCB 
				print alg_name, 'Selected UCB', algorithms[alg_name].users[0].selectedAlgList
				#print alg_name, 'Pool', algorithms[alg_name].users[0].SwitchPoints_pool
				print( str(alg_name)+ "New UCBS: " + str(algorithms[alg_name].users[0].newUCBs))
				print(str(alg_name) + "Discarded UCBS: " + str(algorithms[alg_name].users[0].discardUCBs))
				print(str(alg_name) + "Discarded UCBID: " + str(algorithms[alg_name].users[0].discardUCBIDs ))
				'''
				totaldelay = 0.0
				totalchange = len(actual_changes) -1
				for i in range(len(algorithms[alg_name].users[0].newUCBs)):
					detected_change = algorithms[alg_name].users[0].newUCBs[i]
					#print i, len(actual_changes)
					totaldelay += (detected_change - actual_changes[i])
				#print 'average delay', totaldelay/totalchange
				'''

		

		Alg_Changes_List = {}
		Alg_newUCBs_List = {}
		Alg_discardUCBs_List = {}



		if (self.plot==True): # only plot
		 	linestyles = ['o-', 's-', '*-','>-','<-','g-', '.-', 'o-', 's-', '*-']

			#f, axa = plt.subplots(2, sharex=True)
			markerlist = ['*', 's', 'o', '*', 's']
			m = 0
			#axa[1].scatter(actual_changes, ActualChanges_List, marker = 'o', color = 'b', label = 'Actual Changes')
			
			# for arm in arm_trueReward:
			# 	if arm < 5:
			# 		print arm
			# 		axa[0].plot(arm_trueReward[arm], label = 'arm ' +str(arm))
			# 	elif arm <10:
			# 		axa[0].plot(arm_trueReward[arm], linestyle = '--', label = 'arm ' +str(arm))
			'''

			for alg_name, alg in algorithms.items():
				if 'Master' in alg_name:
					if len(alg.users[0].discardUCBs) > 0:
						labelName = 'dLinUCB'
						print len(alg.users[0].discardUCBs), len(Alg_discardUCBs_List[alg_name])
						#axa[0].scatter(alg.users[0].discardUCBs, Alg_discardUCBs_List[alg_name],  marker = 's', color = 'red',  label = 'dynamic-LinUCB:' + 'Discard Old Model')
						axa[0].axvline(alg.users[0].newUCBs[0], linestyle='-', color = 'green', linewidth = 1.5, label = 'dLinUCB: create new slave model')  #, label = 'dLinUCB:' + 'Create new slave model'
						for j in alg.users[0].newUCBs:
							axa[0].axvline(j, linestyle='-', color = 'green', linewidth = 1.5)
						axa[0].axvline(alg.users[0].discardUCBs[0],  linestyle='-', color = 'red', linewidth = 1.5, label = 'dLinUCB: discard old slave model') #, label = 'dLinUCB:' + 'Discard old slave model'
						for j in alg.users[0].discardUCBs:
							axa[0].axvline(j,  linestyle='-', color = 'red', linewidth = 1.5)
					# elif 'Context' in alg_name:
					# 	labelName = 'ContextdLinUCB'
					# 	#alg_name = 'dynamic LinUCB'
					# 	#axa[0].plot(alg.users[0].ActiveLinUCBNum,  label =  str(alg_name).split('_')[1] +'_ActiveLinUCBNum')
					# 	#axa[0].scatter(alg.users[0].newUCBs, Alg_newUCBs_List[alg_name], marker = 's', color = 'green',  label = 'dynamic-LinUCB:' + 'Create New Model')
					# 	#print len(alg.users[0].discardUCBs), len(Alg_discardUCBs_List[alg_name])
					# 	#axa[0].scatter(alg.users[0].discardUCBs, Alg_discardUCBs_List[alg_name],  marker = 's', color = 'red',  label = 'dynamic-LinUCB:' + 'Discard Old Model')
					# 	axa[0].axvline(alg.users[0].newUCBs[0], linestyle='-', color = 'b', linewidth = 1.5, label = str(labelName)+ ':Create new slave model')
					# 	for j in alg.users[0].newUCBs:
					# 		axa[0].axvline(j, linestyle='-', color = 'b', linewidth = 1.5)
					# 	axa[0].axvline(alg.users[0].discardUCBs[0],  linestyle='-', color = 'c', linewidth = 1.5, label = str(labelName)+ ':Discard old slave model')
					# 	for j in alg.users[0].discardUCBs:
					# 		axa[0].axvline(j,  linestyle='-', color = 'c', linewidth = 1.5)

						axa[0].legend(loc='upper left',prop={'size':8}, ncol=5)
						#axa[0].set_xlabel("Iteration")
						#axa.set_ylabel("Changes")
						axa[0].set_title("Slave Model Creation and Abandonment")
						#axa[0].set_ylim([-1,8])
						#plt.ylim([0,2])
						axa[0].set_yticks([])
						#plt.show()


						#axa[1].scatter(alg.users[0].SwitchPoints, Alg_Changes_List[alg_name],marker = markerlist[m] , color = 'c', label = 'dynamic-LinUCB: Detected Changes')
						m +=1
			#f, axa = plt.subplots(1, sharex=True)
			
			axa[1].legend(loc='upper left',prop={'size':13})
			#axa[1].set_xlabel("Iteration",fontsize = 20, fontweight='bold')
			#axa.set_ylabel("Changes")
			axa[1].set_title("Change Detection")
			#axa[1].set_antialiased(False)
			#axa[1].set_ylim([0,6])
			axa[1].set_yticks([])
			#plt.show()
			'''

			# plot the results	
			f, axa = plt.subplots(1, sharex=True)
			count = 0
			linestyles = ['o-', 's-', '*-','>-','<-','g-', '.-', 'o-', 's-', '*-']
			markerslist = ['*', 's', 'o', '*', 's', 'o', '*']
			colors = [u'b', u'g', u'r', u'c', u'm', u'y', u'k']
			for alg_name, alg in algorithms.items():
				print 'select'
				# print 'alg', str(alg_name), alg.total, alg.selectNew
				if 'DenBandit_LCB' in alg_name:
					if alg.total !=0:
						print 'select new ratio:', str(alg_name), alg.selectNew/alg.total

				if 'Master_' in alg_name:
					#labelName = str(alg_name).split('_')[1]
					labelName = 'dLinUCB'
					alg = algorithms[alg_name]
				elif 'Context' in alg_name:
					#labelName = str(alg_name).split('_')[1]
					labelName = 'ContextdLinUCB'
					alg = algorithms[alg_name]
				else:
					labelName = alg_name

				print linestyles[count]
				print (alg_name, count, len(tim_), len(BatchCumlateRegret[alg_name]))
				axa.plot(tim_, BatchCumlateRegret[alg_name], linewidth = 2, marker = markerslist[count], color=colors[count], markevery = 400,  label = labelName)
				print '%s: %.2f' % (alg_name, BatchCumlateRegret[alg_name][-1])
				if alg.CanEstimateBeta:
					axa.plot(tim_, BatchCumlateRegret_Beta[alg_name], linewidth = 2, marker = markerslist[count], markevery = 400,  label = 'beta_regret'+labelName)

				count +=1
			axa.axvline(actual_changes[0], color='k', linestyle='-', linewidth=1.5 , label = 'Actual Changes')
			for k in actual_changes:
				axa.axvline(k, color='k', linestyle='-', linewidth=1.5 )
			if SAVE:
				pickle.dump(actual_changes, open(filenameWrite_actualChanges, 'wb'))

			for alg_name, alg in algorithms.items():	
				if 'CodBand-LinUCB' in alg_name:
					labelName = alg_name
					alg = algorithms[alg_name]
					#for u in self.users:
					axa.axvline(alg.users[0].newUCBs[0], color='b', linestyle='--', linewidth=1.5 , label = 'CodBand-LinUCB Detected Changes')
					for u in self.users:
						for j in alg.users[u.id].newUCBs:
							axa.axvline(j, color='b', linestyle='--', linewidth=1.5 )
				if 'dLinUCB' in alg_name:
					labelName = 'dLinUCB'
					alg = algorithms[alg_name]
					#for u in self.users:
					axa.axvline(alg.users[0].newUCBs[0], color='b', linestyle='--', linewidth=1.5 , label = 'dLinUCB Detected Changes')
					for u in self.users:
						for j in alg.users[u.id].newUCBs:
							axa.axvline(j, color='b', linestyle='--', linewidth=1.5 )
				if 'DenBandit_LCB' in alg_name:
					#pickle.dump(,  )
					if SAVE:
						pickle.dump(alg.users[0].newUCBs, open(filenameWrite_DenBandDetectedChanges, 'wb'))
					labelName = 'DenBand'
					alg = algorithms[alg_name]
					axa.axvline(alg.users[0].newUCBs[0], color='g', linestyle='--', linewidth=1.5 , label = 'DenBand Detected Changes')
					for u in self.users:
						for j in alg.users[u.id].newUCBs:
							axa.axvline(j, color='g', linestyle='--', linewidth=1.5 )

				if 'CoDBanD' in alg_name:
					#pickle.dump(,  )
					if SAVE:
						pickle.dump(alg.users[0].newUCBs, open(filenameWrite_DenBandDetectedChanges, 'wb'))
					labelName = 'new'
					alg = algorithms[alg_name]
					axa.axvline(alg.users[0].newUCBs[0], color='c', linestyle='--', linewidth=1.5 , label = 'CoDBanD Detected Changes')
					for u in self.users:
						for j in alg.users[u.id].newUCBs:
							axa.axvline(j, color='c', linestyle='--', linewidth=1.5 )
				if 'CoDBand_thres' in alg_name:
					labelName = 'new'
					alg = algorithms[alg_name]
					axa.axvline(alg.users[0].newUCBs[0], color='r', linestyle='--', linewidth=1.5 , label = 'CoDBanD-thres Detected Changes')
					for u in self.users:
						for j in alg.users[u.id].newUCBs:
							axa.axvline(j, color='r', linestyle='--', linewidth=1.5 )


			handles,labels = axa.get_legend_handles_labels()
			print handles,labels 
			# handles = [handles[0], handles[2], handles[1]]
			# labels = [labels[0], labels[2], labels[1]]

			axa.legend(handles,labels ,loc='upper left',prop={'size':13}, ncol = 2)

			# axa.legend(loc='upper left',prop={'size':13}, ncol = 2)
			#axa[2].set_xlabel("Iteration", fontsize = 20, fontweight='bold')
			axa.set_ylabel("Regret", fontsize = 22, fontweight='bold')
			# axa.set_title("Accumulated Regret")
			

			plt.xlabel("Iteration", fontsize = 22, fontweight='bold')
			#plt.xlim([0,2000])

			#plt.savefig('./results/'  + str(namelabel) + str(timeRun) + '.pdf')
			plt.show()
			
			# plot the estimation error of co-theta
			f, axa = plt.subplots(1, sharex=True)
			time = range(self.testing_iterations)
			for alg_name, alg in algorithms.items():

				if alg.CanEstimateUserPreference:
					axa.plot(time, ThetaDiffList[alg_name], label = alg_name)
				if alg.CanEstimateBeta:
					axa.plot(time, BetaDiffList[alg_name], label = alg_name + '_Beta')
				if alg.CanEstimateCoUserPreference:
					axa.plot(time, CoThetaDiffList[alg_name], label = alg_name + '_CoTheta')
				# if alg.CanEstimateV:
				# 	axa.plot(time, VDiffList[alg_name], label = alg_name + '_V')			
				# 	axa.plot(time, CoThetaVDiffList[alg_name], label = alg_name + '_CoThetaV')	
				# 	axa.plot(time, RVDiffList[alg_name], label = alg_name + '_RV')	
				# 	axa.plot(time, RDiffList[alg_name], label = alg_name + '_R')	

			axa.legend(loc='upper right',prop={'size':6})
			axa.set_xlabel("Iteration")
			# axa.set_ylabel("L2 Diff")
			# #axa.set_yscale('log')
			# axa.set_title("Parameter estimation error")

			axa.set_ylabel("LogLikelihood")
			#axa.set_yscale('log')
			axa.set_title("Real Time LogLikelihood")

			plt.show()


			f, axa = plt.subplots(1, sharex=True)
			time = range(self.testing_iterations)
			for alg_name, alg in algorithms.items():

				if 'DenBandit_LCB' in alg_name:
					axa.plot(alg.reward_estimation_error_new, label = 'New')
					axa.plot(alg.reward_estimation_error_LCB, label = 'LCB')
					false_positive_list_value = alg.false_positive_list.keys()
					b3 = [val for val in false_positive_list_value if val in alg.reward_estimation_error_diff_list]
					#axa.axvline(alg.reward_estimation_error_diff_list[0], linestyle='-', color = 'b', linewidth = 1.5, label = str(alg_name)+ 'False Positive')
					print 'num of false_positive', len(false_positive_list_value)
					print 'num of diff error', len(alg.reward_estimation_error_diff_list)
					print 'num of over alp', len(b3)
					print 'opt_stationary', opt_stationary
					#print 'Optimal Ratio in Change-invariant arms', opt_stationary/float(alg.change_invariant_arms)
					#for j in alg.reward_estimation_error_diff_list:
						#axa.axvline(j, linestyle='-', color = 'b', linewidth = 1.5)
	
			axa.legend(loc='upper right',prop={'size':6})
			axa.set_xlabel("Iteration")
			axa.set_ylabel("L2 Diff")
			#axa.set_yscale('log')
			axa.set_title("Reward estimation error")
			plt.show()


			f, axa = plt.subplots(1, sharex=True)
			time = range(self.testing_iterations)
			for alg_name, alg in algorithms.items():
				if 'DenBandit_LCB' in alg_name:
					axa.plot(alg.selected_arm_estimation_error_new, label = 'New')
					axa.plot(alg.selected_arm_estimation_error_LCB, label = 'LCB')
					false_positive_list_value = alg.false_positive_list.keys()
					b3 = [val for val in false_positive_list_value if val in alg.reward_estimation_error_diff_list]
					#axa.axvline(alg.reward_estimation_error_diff_list[0], linestyle='-', color = 'b', linewidth = 1.5, label = str(alg_name)+ 'False Positive')
					print 'num of false_positive', len(false_positive_list_value)
					print 'num of diff error', len(alg.reward_estimation_error_diff_list)
					print 'num of over alp', len(b3)
					#for j in alg.reward_estimation_error_diff_list:
					#	axa.axvline(j, linestyle='-', color = 'b', linewidth = 1.5)
	
			axa.legend(loc='upper right',prop={'size':6})
			axa.set_xlabel("Iteration")
			axa.set_ylabel("L2 Diff")
			axa.set_yscale('log')
			axa.set_title("Selected arm Reward estimation error")
			plt.show()

			#Plot Each bandit model's Theta Estimaiton Quality
			for alg_name, alg in algorithms.items():
				if 'DenBandit' in alg_name:
					f, axa = plt.subplots(1, sharex=True)
					for userID in range(len(self.users)):
						model_theta_estimation_error_dic ={}
						for bandit_expert in alg.users[userID].EXPERTs:
							for i in range(testing_iterations):
								#File blank intereaction with 0
								if i not in bandit_expert.theta_estimation_error:
									bandit_expert.theta_estimation_error[i] = 0
							sorted_dic = sorted(bandit_expert.theta_estimation_error.items())
							model_theta_estimation_error_dic[bandit_expert.model_id] = []
							for k, v in sorted_dic:
								model_theta_estimation_error_dic[bandit_expert.model_id].append(v)
							axa.plot(model_theta_estimation_error_dic[bandit_expert.model_id], label = str(bandit_expert.model_id))
						filenameWritePara_current = filenameWritePara + '__' + str(alg_name) + '.csv'
						print 'name', filenameWritePara_current
						if SAVE:
							with open(filenameWritePara_current, 'w') as f:
								f.write('Time(Iteration)')
								f.write(',' + ','.join(['model_id_' + str(id_) for id_ in model_theta_estimation_error_dic.keys() ]))
								f.write('\n')
								#for id_ in model_theta_estimation_error_dic.keys():
								for k in range(len(model_theta_estimation_error_dic[0])):
									f.write(str(iter_))
									f.write(',' + ','.join([str(model_theta_estimation_error_dic[id_][k]) for  id_ in model_theta_estimation_error_dic.keys()]))
									#f.write(',' + ','.join([str(alg_name)+'CoTheta' for alg_name in CoThetaDiffList.iterkeys()]))
									f.write('\n')
					axa.legend(loc='upper right',prop={'size':6})
					axa.set_xlabel("Iteration")
					axa.set_ylabel("L2 Diff")
					#axa.set_yscale('log')
					axa.set_title(str(alg_name) + "Model Theta estimation error")
					plt.show()
		print 'usedEnvironment_id', usedEnvironment_id
		print 'usedEnvironment_list', usedEnvironment_list
		print 'self.optimal_arm_changeSensitiveRatio:', 1- self.optimal_arm_changeSensitiveRatio/float(testing_iterations)
		for u in self.users:
			print 'user', u.id, ThetaList[u.id]
			print 'usser index', u.id, ThetaList_index[u.id]
		finalRegret = {}
		for alg_name, alg in algorithms.items():
			print '%s: %.2f' % (alg_name, BatchCumlateRegret[alg_name][-1])
			if alg_name == 'Context_dLinUCB':
				print 'beta observation'
				#alg.printBetaObservation(self.users[0].id)

		return BatchCumlateRegret

def pca_articles(articles, order):
	X = []
	for i, article in enumerate(articles):
		X.append(article.featureVector)
	pca = PCA()
	X_new = pca.fit_transform(X)
	# X_new = np.asarray(X)
	print('pca variance in each dim:', pca.explained_variance_ratio_) 

	print X_new
	#default is descending order, where the latend features use least informative dimensions.
	if order == 'random':
		np.random.shuffle(X_new.T)
	elif order == 'ascend':
		X_new = np.fliplr(X_new)
	elif order == 'origin':
		X_new = X
	for i, article in enumerate(articles):
		articles[i].featureVector = X_new[i]
	return


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description = '')
	parser.add_argument('--alg', dest='alg', help='Select a specific algorithm, could be CoLin, hLinUCB, factorUCB, LinUCB, etc.')
	parser.add_argument('--namelabel', dest='namelabel', help='Name')
	parser.add_argument('--stationary_ratio', dest = 'stationary_ratio' , help='Set dimension of context features.')
	parser.add_argument('--contextdim', type=int, help='Set dimension of context features.')
	parser.add_argument('--hiddendim', type=int, help='Set dimension of hidden features.')
	parser.add_argument('--AVG', action='store_true', help='run multiple times to get mean and std') 

	parser.add_argument('--minWindow', help='Feature dimension used for estimation.')
	parser.add_argument('--sigma_0', help='Feature dimension used for estimation.')                   
	parser.add_argument('--sigma',  help='Feature dimension used for estimation.')

	args = parser.parse_args()
	algName = str(args.alg)
	namelabel = str(args.namelabel)
	AVG = args.AVG


	if args.contextdim:
		context_dimension = args.contextdim
	else:
		context_dimension = 10
	if args.hiddendim:
		latent_dimension = args.hiddendim
	else:
		latent_dimension = 0

	if args.stationary_ratio:
		stationary_ratio = float(args.stationary_ratio)
	else:
		stationary_ratio = None

	if args.minWindow:
		minWindow = int(args.minWindow)
	else:
		minWindow = 100

	if args.sigma_0:
		sigma_0 = float(args.sigma_0)
	else:
		sigma_0 = 0.1

	if args.sigma:
		sigma = float(args.sigma)
	else:
		sigma = 0.01


	training_iterations = 0
	testing_iterations = 1000
	tau = minWindow
	
	NoiseScale = 0.05

	

	#alpha  = 0.3
	#alpha = 0.6
	lambda_ = 0.1   # Initialize A
	#alpha = 0.1
	alpha = 0.1
	#alpha = 30*NoiseScale + math.sqrt(lambda_)
	#lambda_ = 0.05
	epsilon = 0 # initialize W
	eta_ = 0.5
	alpha_2 = 0.04

	n_articles = 1000
	ArticleGroups = 0

	n_users = 5
	UserGroups = 5
	
	poolSize = 20
	batchSize = 1

	# Matrix parameters
	matrixNoise = 0.001
	sparseLevel = n_users  # if smaller or equal to 0 or larger or enqual to usernum, matrix is fully connected


	# Parameters for GOBLin
	G_alpha = alpha
	G_lambda_ = lambda_
	Gepsilon = 1



	articles_dic = {}
	#Non
	userFilename = os.path.join(sim_files_folder, "nonstationary_users_"+str(n_users)+"context_"+str(context_dimension)+"latent_"+str(latent_dimension)+ "Ugroups" + str(UserGroups)+'change10'+".json")
	
	#"Run if there is no such file with these settings; if file already exist then comment out the below funciton"
	# we can choose to simulate users every time we run the program or simulate users once, save it to 'sim_files_folder', and keep using it.
	UM = UserManager(context_dimension+latent_dimension, n_users, UserGroups = UserGroups, thetaFunc=gaussianFeature, argv={'l2_limit':1})
	#users = UM.simulateThetafromUsers()
	#UM.saveUsers(users, userFilename, force = True)
	users = UM.loadUsers(userFilename)

	
	for u in users:
		articlesFilename = os.path.join(sim_files_folder, "CoDBand_articles_"+str(n_articles) +'for_user' + str(u.id) +"d"+str(context_dimension)+'change10_nullspace'+".json")
		# Similarly, we can choose to simulate articles every time we run the program or simulate articles once, save it to 'sim_files_folder', and keep using it.
		AM = ArticleManager(context_dimension+latent_dimension, user = u, n_articles=n_articles, ArticleGroups = ArticleGroups,
				FeatureFunc=gaussianFeature,  argv={'l2_limit':1})
		
		#articles = AM.simulateArticlePool()
# 
		#AM.saveArticles(articles, articlesFilename, force=True)
		articles = AM.loadArticles(articlesFilename)

		articles_dic[u.id] = articles

		articles_expand = []
		change_num = 15
		for i in range(change_num-1):
			articles_expand += articles['nonstationary'][i]
		for i in range(change_num):
			for j in range(i):
				articles_expand += articles['stationary'][i][j]
	print len(articles_expand)
		
	
	
	# all_articlesFilename = os.path.join(sim_files_folder, "articles_"+str(n_articles) +"d"+str(context_dimension)+".p")
	# #AM.saveArticles(articles_dic, all_articlesFilename , force=False)
	# articles_dic = pickle.load(open(all_articlesFilename, 'rb'))
		


	if AVG:
		plot = False
	else:
		plot = True

	if AVG:
		run_num = 5
		alg_regret_list_dic = {}
	else:
		run_num = 1
		alg_regret_list_dic = {}

	for i in range(run_num):
		simExperiment = simulateOnlineData(namelabel = namelabel, context_dimension = context_dimension,
							latent_dimension = latent_dimension,
							training_iterations = training_iterations,
							testing_iterations = testing_iterations,
							testing_method = "online", # batch or online
							plot = plot,
							articles_dic =articles_dic,
							users = users,		
							noise = lambda : np.random.normal(scale = NoiseScale),
							matrixNoise = lambda : np.random.normal(scale = matrixNoise),
							batchSize = batchSize,
							type_ = "UniformTheta", 
							signature = AM.signature,
							sparseLevel = sparseLevel,
							poolArticleSize = poolSize, NoiseScale = NoiseScale, epsilon = epsilon, Gepsilon =Gepsilon, stationary_ratio= stationary_ratio)

		print "Starting for ", simExperiment.simulation_signature

		#algorithms = {}

		
		algorithms = {}
		if algName == 'dLinUCBs':
			# algorithms['WMDUCB1'] = WMDUCB1Algorithm(window= 100, checkInter = 50)
			# algorithms['MetaBandit'] = MetaBanditAlgorithm(arms = n_articles , articles = articles_expand)
		
			#algorithms['RLB'] = RLBAlgorithm(dimension = context_dimension)
			
			#algorithms['AdTS_likelihood'] = dTSAlgorithm(dimension = context_dimension, n = n_users, AdTS_Window = 80, AdTS_CheckInter = 10)
			#algorithms['AdTS_LikelihoodSampling'] = dTSAlgorithm_Sample(dimension = context_dimension, n = n_users, AdTS_Window = 80, AdTS_CheckInter = 10)
			#algorithms['TS'] = TSAlgorithm(dimension = context_dimension, n = n_users, AdTS_Window = 50, AdTS_CheckInter = 20)
			#algorithms['LinUCB'] = N_LinUCBAlgorithm(dimension = context_dimension, alpha = alpha, lambda_ = lambda_, n = n_users, NoiseScale =NoiseScale)
			#algorithms['LinUCB'] = LinUCB(dimension = context_dimension, alpha = alpha, lambda_ = lambda_ , userID = 0, pretrained_Theta= 'zero', pretrained_A = lambda_*np.identity(n = context_dimension), pretrained_b = np.zeros(context_dimension) )
			
			#algorithms['CoLinUCB'] = AsyCoLinUCBAlgorithm(dimension = context_dimension, alpha = alpha, lambda_ = lambda_, n = n_users, W = simExperiment.getW())
			#algorithms['CLUB'] = CLUBAlgorithm(dimension = context_dimension, alpha = alpha, lambda_ = lambda_, n = n_users, alpha_2 = alpha_2)
			
			algorithms['dLinUCB'] = dLinUCBAlgorithm(dimension = context_dimension, alpha = alpha, lambda_ = lambda_, tau = tau, NoiseScale =NoiseScale)
			#algorithms['AdTS'] = AdaptiveThompsonAlgorithm(dimension = context_dimension, n = n_users, AdTS_Window = 80, AdTS_CheckInter = 10, v = 0.1)
			##algorithms['dLinUCB_Nodiscard'] = dLinUCBAlgorithm_noDiscard(dimension = context_dimension, alpha = alpha, lambda_ = lambda_, NoiseScale =NoiseScale)
			#algorithms['dLinUCB_share'] = dLinUCBAlgorithm_share(dimension = context_dimension, alpha = alpha, lambda_ = lambda_, NoiseScale =NoiseScale)
			#algorithms['dLinUCB_likelihood__'] = dLinUCB_LikelihoodAlgorithm(dimension = context_dimension, alpha = alpha, lambda_ = lambda_, NoiseScale =NoiseScale)
			#algorithms['dLinUCB_likelihood_eliminate'] = dLinUCBAlgorithm_likelihood_eliminate(dimension = context_dimension, alpha = alpha, lambda_ = lambda_, NoiseScale =NoiseScale)
			#algorithms['dLinUCB_likelihood_new'] = dLinUCBAlgorithm_likelihood_new(dimension = context_dimension, alpha = alpha, lambda_ = lambda_, NoiseScale =NoiseScale)
			#algorithms['dLinUCB_likelihood_Sample'] = dLinUCB_DP(dimension = context_dimension, alpha = alpha, lambda_ = lambda_, minWindow = minWindow,  sigma_0 = sigma_0, sigma = sigma)
			#algorithms['CoDBan_SingleUpate'] = dLinUCB_DP_Log(dimension = context_dimension, alpha = alpha, lambda_ = lambda_, minWindow = minWindow, sigma_0 = sigma_0, sigma = sigma, Update_Method = 'Single')
			#algorithms['CoDBanD'] = CoDBand(dimension = context_dimension, alpha = alpha, lambda_ = lambda_, minWindow = minWindow, sigma_0 = sigma_0, sigma = sigma, Update_Method = 'Sample')
			#algorithms['CoDBanD_NonPOP'] = CoDBand_NonPop(dimension = context_dimension, alpha = alpha, lambda_ = lambda_, minWindow = minWindow, sigma_0 = sigma_0, sigma = sigma, Update_Method = 'Sample')
			algorithms['CoDBanD'] = CoDBand_bak(dimension = context_dimension, alpha = alpha, lambda_ = lambda_, minWindow = minWindow, sigma_0 = sigma_0, sigma = sigma, Update_Method = 'Sample')
			#algorithms['CoDBand_thres'] = CoDBand_thres(dimension = context_dimension, alpha = alpha, lambda_ = lambda_, minWindow = minWindow, sigma_0 = sigma_0, sigma = sigma, Update_Method = 'Single')
			#algorithms['CoDBand-Reset'] = dLinUCB_DP_Log_Reset(dimension = context_dimension, alpha = alpha, lambda_ = lambda_, minWindow = minWindow, sigma_0 = sigma_0, sigma = sigma, Update_Method = 'Sample')
			#algorithms['CoDBan_BakReset'] = dLinUCB_DP_Log_bak_Reset(dimension = context_dimension, alpha = alpha, lambda_ = lambda_, minWindow = minWindow, sigma_0 = sigma_0, sigma = sigma, Update_Method = 'Sample')
			#
			#
			#algorithms['CoDBan_org'] = dLinUCB_DP_Log_org(dimension = context_dimension, alpha = alpha, lambda_ = lambda_, minWindow = minWindow, sigma_0 = sigma_0, sigma = sigma, Update_Method = 'Weight')
			#algorithms['CoDBan_SampleMultiple'] = dLinUCB_DP_Log(dimension = context_dimension, alpha = alpha, lambda_ = lambda_, minWindow = minWindow, sigma_0 = sigma_0, sigma = sigma, Update_Method = 'Sample')
			#algorithms['CodBand-LinUCB'] = dLinUCB_DP_Log_DP(dimension = context_dimension, alpha = alpha, lambda_ = lambda_, minWindow = minWindow, sigma_0 = sigma_0, sigma = sigma, Update_Method = 'Weight')
			#algorithms['CodBand-TS'] = dTS_DP_Log(dimension = context_dimension, alpha = alpha, lambda_ = lambda_, minWindow = minWindow, sigma_0 = sigma_0, sigma = sigma, Update_Method = 'Weight')
			
			# #algorithms['dLinUCB_likelihood_M'] = dLinUCB_Likelihood_Multiple_Algorithm(dimension = context_dimension, alpha = alpha, lambda_ = lambda_, NoiseScale =NoiseScale)
			#algorithms['dLinUCB_likelihood_Share'] = dLinUCBAlgorithm_likelihoodShare(dimension = context_dimension, alpha = alpha, lambda_ = lambda_, NoiseScale =NoiseScale)
			#algorithms['dLinUCB_likelihood_Share_POP'] = dLinUCBAlgorithm_likelihoodShare_POP(dimension = context_dimension, alpha = alpha, lambda_ = lambda_, NoiseScale =NoiseScale)
			#algorithms['dLinUCB_likelihood_Share_POP_all'] = dLinUCBAlgorithm_likelihoodShare_POP_ALL(dimension = context_dimension, alpha = alpha, lambda_ = lambda_, NoiseScale =NoiseScale)
			# #algorithms['dLinUCB_AlwaysShare'] = dLinUCBAlgorithm_AlwaysShare(dimension = context_dimension, alpha = alpha, lambda_ = lambda_, NoiseScale =NoiseScale)
			
			# algorithms['LinUCB_restart'] = N_LinUCBAlgorithm_restart(dimension = context_dimension, alpha = alpha, lambda_ = lambda_, n = n_users, NoiseScale =NoiseScale)
			# #algorithms['DenBandit_Oracle'] =DenBand_Oracle(dimension = context_dimension, alpha = alpha, lambda_ = lambda_,   NoiseScale =NoiseScale, tau = tau)
			#algorithms['DenBandit_LCB'] =DenBand(dimension = context_dimension, alpha = alpha, lambda_ = lambda_,   NoiseScale =NoiseScale, tau = tau)
			# #algorithms['DenBandit_share'] =ContextDependent_dLinUCBAlgorithm_frozen(dimension = context_dimension, alpha = alpha, lambda_ = lambda_,   NoiseScale =NoiseScale, tau = tau)
			
			#algorithms['DenBandit_AVG'] = DenBand_avg(dimension = context_dimension, alpha = alpha, lambda_ = lambda_, NoiseScale =NoiseScale, tau = tau)

			#algorithms['DenBandit_Oracle'] =DenBand_Oracle(dimension = context_dimension, alpha = alpha, lambda_ = lambda_,   NoiseScale =NoiseScale, tau = 200)
			
			#algorithms['LinUCB_oralce_restart'] = N_LinUCBAlgorithm_oracle(dimension = context_dimension, alpha = alpha, lambda_ = lambda_, n = n_users, NoiseScale =NoiseScale)
			
			#algorithms['c_dLinUCB'] =ContextDependent_dLinUCBAlgorithm_update(dimension = context_dimension, alpha = alpha, lambda_ = lambda_, n = n_users, NoiseScale =NoiseScale, tau = 200)
			# algorithms['c_dLinUCB_beta_all'] =ContextDependent_dLinUCBAlgorithm_update(dimension = context_dimension, alpha = alpha, lambda_ = lambda_, n = n_users, NoiseScale =NoiseScale, tau = tau, beta='all')
			#algorithms['c_dLinUCB_beta_lifespan_600'] =ContextDependent_dLinUCBAlgorithm_update(dimension = context_dimension, alpha = alpha, lambda_ = lambda_, n = n_users, NoiseScale =NoiseScale, tau = tau, L = 5000, beta='original')
			#algorithms['c_dLinUCB_beta_orginial'] =ContextDependent_dLinUCBAlgorithm_update_avg(dimension = context_dimension, alpha = alpha, lambda_ = lambda_, n = n_users, NoiseScale =NoiseScale, tau = tau, L = 5000, beta='original')
			#algorithms['DenBandit_LCB'] =DenBand(dimension = context_dimension, alpha = alpha, lambda_ = lambda_,   NoiseScale =NoiseScale, tau = tau)
			#algorithms['DenBandit_AVG'] = DenBand_avg(dimension = context_dimension, alpha = alpha, lambda_ = lambda_, NoiseScale =NoiseScale, tau = tau)
			#algorithms['DenBandit_Random'] = DenBand_random(dimension = context_dimension, alpha = alpha, lambda_ = lambda_, NoiseScale =NoiseScale, tau = 200)
			#algorithms['DenBandit_SelectNew'] = DenBand_SelectNew(dimension = context_dimension, alpha = alpha, lambda_ = lambda_, NoiseScale =NoiseScale, tau = tau)
			#algorithms['DenBandit_DenBand_SelectOld'] = DenBand_SelectOld(dimension = context_dimension, alpha = alpha, lambda_ = lambda_, NoiseScale =NoiseScale, tau = 200)
			
			#algorithms['DenBandit_SeperateOldNew'] = DenBand_SeperateOldNew(dimension = context_dimension, alpha = alpha, lambda_ = lambda_, NoiseScale =NoiseScale, tau = 200)

			## Only Update New
			#algorithms['DenBandit_OnlyUpdateNew_seperate'] =DenBand_OnlyUpdateNew_SeperateOldNew(dimension = context_dimension, alpha = alpha, lambda_ = lambda_,   NoiseScale =NoiseScale, tau = 100)
			#algorithms['DenBandit_OnlyUpdateNew_LCB'] =DenBand_OnlyUpdateNew_LCB(dimension = context_dimension, alpha = alpha, lambda_ = lambda_,   NoiseScale =NoiseScale, tau = 100)
			#algorithms['DenBandit_OnlyUpdateNew_Random'] =DenBand_OnlyUpdateNew_Random(dimension = context_dimension, alpha = alpha, lambda_ = lambda_,   NoiseScale =NoiseScale, tau = 100)
			#algorithms['DenBandit_OnlyUpdateNew_New'] =DenBand_OnlyUpdateNew_New(dimension = context_dimension, alpha = alpha, lambda_ = lambda_,   NoiseScale =NoiseScale, tau = 100)
			
			## Oracle Update
			#algorithms['DenBandit_OracleUpdate_LCB'] =DenBand_OUpdate_LCB(dimension = context_dimension, alpha = alpha, lambda_ = lambda_,   NoiseScale =NoiseScale, tau = 100)
			#algorithms['DenBandit_OracleUpdate_Random'] =DenBand_OUpdate_Random(dimension = context_dimension, alpha = alpha, lambda_ = lambda_,   NoiseScale =NoiseScale, tau = 100)
			#algorithms['DenBandit_OracleUpdate_New'] =DenBand_OUpdate_New(dimension = context_dimension, alpha = alpha, lambda_ = lambda_,   NoiseScale =NoiseScale, tau = 100)
			
			## Oracle Reuse
			#algorithms['DenBandit_OraReuse_LCB'] =DenBand_OReuse_LCB(dimension = context_dimension, alpha = alpha, lambda_ = lambda_,   NoiseScale =NoiseScale, tau = 100)
			#algorithms['DenBandit_OrReuse_Random'] =DenBand_OReuse_Random(dimension = context_dimension, alpha = alpha, lambda_ = lambda_,   NoiseScale =NoiseScale, tau = 100)
			#algorithms['DenBandit_OrReuse_New'] =DenBand_OReuse_New(dimension = context_dimension, alpha = alpha, lambda_ = lambda_,   NoiseScale =NoiseScale, tau = 100)
			#algorithms['DenBandit_OrReuse_Old'] =DenBand_OReuse_Old(dimension = context_dimension, alpha = alpha, lambda_ = lambda_,   NoiseScale =NoiseScale, tau = 100)
			
			#Oracle Models
			#algorithms['DenBandit_OracleUpdate'] =DenBand_OracleUpdate(dimension = context_dimension, alpha = alpha, lambda_ = lambda_,   NoiseScale =NoiseScale, tau = 100)
			#algorithms['DenBandit_OracleUpdate_SeperateOldNew'] =DenBand_OracleUpdate_SeperateOldNew(dimension = context_dimension, alpha = alpha, lambda_ = lambda_,   NoiseScale =NoiseScale, tau = 100)
			
			#algorithms['DenBandit_ReuseOracle'] =DenBand_OracleReuse(dimension = context_dimension, alpha = alpha, lambda_ = lambda_,   NoiseScale =NoiseScale, tau = 200)
			#algorithms['DenBandit_OracleReuseOracle_Random'] =DenBand_OracleReuse_Random(dimension = context_dimension, alpha = alpha, lambda_ = lambda_,   NoiseScale =NoiseScale, tau = 200)
			#algorithms['DenBandit_OracleReuse_Old'] =DenBand_OracleReuse_Old(dimension = context_dimension, alpha = alpha, lambda_ = lambda_,   NoiseScale =NoiseScale, tau = 200)
			
			#algorithms['DenBandit_OracleReuseOracle_LCB'] =DenBand_OracleReuse_LCB(dimension = context_dimension, alpha = alpha, lambda_ = lambda_,   NoiseScale =NoiseScale, tau = 200)
			#algorithms['DenBandit_OracleReuseOracle_New'] =DenBand_OracleReuse_New(dimension = context_dimension, alpha = alpha, lambda_ = lambda_,   NoiseScale =NoiseScale, tau = 200)
			
			#algorithms['DenBandit_Oracle'] =DenBand_Oracle(dimension = context_dimension, alpha = alpha, lambda_ = lambda_,   NoiseScale =NoiseScale, tau = 200)
			
			#algorithms['DenBandit_RestartBeta'] =DenBand_RestartBeta(dimension = context_dimension, alpha = alpha, lambda_ = lambda_,   NoiseScale =NoiseScale, tau = 200)
			#algorithms['DenBandit_AlwaysUpdateRestartBeta'] =DenBand_AlwaysUpdteRestartBeta(dimension = context_dimension, alpha = alpha, lambda_ = lambda_,   NoiseScale =NoiseScale, tau = 200)
			

			
			#algorithms['OracleRestart'] = N_LinUCBAlgorithm_restart(dimension = context_dimension, alpha = alpha, lambda_ = lambda_, n = n_users, NoiseScale =NoiseScale)
			#algorithms['OracleReuse'] = N_LinUCBAlgorithm_oracle(dimension = context_dimension, alpha = alpha, lambda_ = lambda_, n = n_users, NoiseScale =NoiseScale)
			
			#algorithms['DenBandit_SelectUpdate'] =DenBand_selectUpdate(dimension = context_dimension, alpha = alpha, lambda_ = lambda_,   NoiseScale =NoiseScale, tau = 200)
			#algorithms['DenBandit_update'] = DenBand_Update(dimension = context_dimension, alpha = alpha, lambda_ = lambda_, NoiseScale =NoiseScale, tau = 200)
			#algorithms['DenBandit_revised'] =DenBand_ReviseBeta(dimension = context_dimension, alpha = alpha, lambda_ = lambda_,   NoiseScale =NoiseScale, tau = 200)
			#algorithms['DenBandit_updateall'] =DenBand_Update_Update(dimension = context_dimension, alpha = alpha, lambda_ = lambda_,   NoiseScale =NoiseScale, tau = 200)
			#algorithms['DenBandit_Revised_RestartBeta'] =DenBand_Revised_RestartBeta(dimension = context_dimension, alpha = alpha, lambda_ = lambda_,   NoiseScale =NoiseScale, tau = 200)
			#algorithms['DenBandit_SelectNew_revised'] = DenBand_SelectNew_revised(dimension = context_dimension, alpha = alpha, lambda_ = lambda_, NoiseScale =NoiseScale, tau = 200)
		if algName == 'All':
			algorithms['Master_dLinUCB'] = dLinUCBAlgorithm_time(dimension = context_dimension, alpha = alpha, lambda_ = lambda_, n = n_users, NoiseScale =NoiseScale)
			algorithms['LinUCB'] = N_LinUCBAlgorithm(dimension = context_dimension, alpha = alpha, lambda_ = lambda_, n = n_users, NoiseScale =NoiseScale)
			#algorithms['WMDUCB1'] = WMDUCB1Algorithm(window= 300, checkInter = 200)
			#algorithms['RLB'] = RLBAlgorithm(dimension = context_dimension)
			algorithms['AdTS'] = AdaptiveThompsonAlgorithm(dimension = context_dimension, n = n_users, AdTS_Window = 50, AdTS_CheckInter = 20)
			# algorithms['Master_LCBInterval'] = dLinUCBAlgorithm(dimension = context_dimension, alpha = alpha, lambda_ = lambda_, n = n_users, NoiseScale =NoiseScale)
			# algorithms['LCBIntervalContext'] =ContextDependent_dLinUCBAlgorithm(dimension = context_dimension, alpha = alpha, lambda_ = lambda_, n = n_users, NoiseScale =NoiseScale)
			#algorithms['DenBand-lcb'] =ContextDependent_dLinUCBAlgorithm_update(dimension = context_dimension, alpha = alpha, lambda_ = lambda_, n = n_users, NoiseScale =NoiseScale, tau = tau)
			#algorithms['DenBand-avg'] =ContextDependent_dLinUCBAlgorithm_update_avg(dimension = context_dimension, alpha = alpha, lambda_ = lambda_, n = n_users, NoiseScale =NoiseScale, tau = tau)
			#algorithms['MetaBandit'] = MetaBanditAlgorithm(arms = n_articles , articles = articles_expand)
		
		if algName == 'DenBand':
			algorithms['DenBand-avg'] =ContextDependent_dLinUCBAlgorithm_update_avg(dimension = context_dimension, alpha = alpha, lambda_ = lambda_, n = n_users, NoiseScale =NoiseScale, tau = tau, L = 1500, beta='original')
			algorithms['DenBand-lcb'] =ContextDependent_dLinUCBAlgorithm_update(dimension = context_dimension, alpha = alpha, lambda_ = lambda_, n = n_users, NoiseScale =NoiseScale, tau = tau, L = 1500, beta='original')
			algorithms['OracleRestart'] = N_LinUCBAlgorithm_restart(dimension = context_dimension, alpha = alpha, lambda_ = lambda_, n = n_users, NoiseScale =NoiseScale)
			algorithms['OracleReuse'] = N_LinUCBAlgorithm_oracle(dimension = context_dimension, alpha = alpha, lambda_ = lambda_, n = n_users, NoiseScale =NoiseScale)
			algorithms['AdTS'] = AdaptiveThompsonAlgorithm(dimension = context_dimension, n = n_users, AdTS_Window = 100, AdTS_CheckInter = 40)
			algorithms['LinUCB'] = N_LinUCBAlgorithm(dimension = context_dimension, alpha = alpha, lambda_ = lambda_, n = n_users, NoiseScale =NoiseScale)

		regret_dic = simExperiment.runAlgorithms(algorithms)
		for alg_name in regret_dic.keys():
			if alg_name not in alg_regret_list_dic:
				alg_regret_list_dic[alg_name] = []
			alg_regret_list_dic[alg_name].append(regret_dic[alg_name][-1])
			print alg_name, regret_dic[alg_name][-1]
	for alg in alg_regret_list_dic.keys():
		#print alg, alg_regret_list_dic[alg]
		regret_array = np.array(alg_regret_list_dic[alg])
		print alg, regret_array
		print alg, 'regret mean', np.mean(regret_array), 'std', np.std(regret_array)
		# print alg, 'regret mean', np.mean(regret_array), 'std', np.std(regret_array)
