import numpy as np
from random import sample, shuffle, choice
import datetime
import os.path
import matplotlib.pyplot as plt
from conf import *

# real dataset
from dataset_utils.LastFM_util_functions_2 import readFeatureVectorFile, parseLine


class Article():
    def __init__(self, aid, FV=None):
        self.article_id = aid
        self.contextFeatureVector = FV
        self.featureVector = FV


class experimentOnRealData(object):
    def __init__(self, namelabel, dataset, context_dimension, batchSize=1, plot=True, Write_to_File=False):

        self.namelabel = namelabel
        assert dataset in ["LastFM"]
        self.dataset = dataset
        self.context_dimension = context_dimension
        self.Plot = plot
        self.Write_to_File = Write_to_File
        self.batchSize = batchSize

        self.address = LastFM_address
        self.save_address = LastFM_save_address
        FeatureVectorsFileName = LastFM_FeatureVectorsFileName

        self.event_fileName = self.address + "/eventFile.dat"  # processed_event dataset

        # Read Feature Vectors from File
        self.FeatureVectors = readFeatureVectorFile(FeatureVectorsFileName)
        self.articlePool = []

    def getL2Diff(self, x, y):
        return np.linalg.norm(x - y)  # L2 norm

    def batchRecord(self, iter_):
        print("Iteration %d" % iter_, "Pool", len(self.articlePool), " Elapsed time",
              datetime.datetime.now() - self.startTime)

    def runAlgorithms(self, algorithms, startTime):
        self.startTime = startTime
        timeRun = self.startTime.strftime('_%m_%d_%H_%M_%S')

        filenameWriteReward = os.path.join(self.save_address, 'AccReward' + str(self.namelabel) + timeRun + '.csv')
        end_num = 0
        while os.path.exists(filenameWriteReward):
            filenameWriteReward = os.path.join(self.save_address,
                                               'AccReward' + str(self.namelabel) + timeRun + str(end_num) + '.csv')
            end_num += 1
        tim_ = []
        AlgReward = {}
        BatchCumlateReward = {}
        AlgReward["random"] = []
        BatchCumlateReward["random"] = []
        for alg_name, alg in algorithms.items():
            AlgReward[alg_name] = []
            BatchCumlateReward[alg_name] = []

        if self.Write_to_File:
            with open(filenameWriteReward, 'w') as f:
                f.write('Time(Iteration)')
                f.write(',' + ','.join([str(alg_name) for alg_name in algorithms.keys()]))
                f.write('\n')

        userIDSet = set()
        with open(self.event_fileName, 'r') as f:
            f.readline()
            iter_ = 0
            for _, line in enumerate(f, 1):
                userID, _, pool_articles = parseLine(line)
                if userID not in userIDSet:
                    userIDSet.add(userID)
                # ground truth chosen article
                article_id_chosen = int(pool_articles[0])
                # Construct arm pool
                self.article_pool = []
                for article in pool_articles:
                    article_id = int(article.strip(']'))
                    article_featureVector = self.FeatureVectors[article_id]
                    article_featureVector = np.array(article_featureVector, dtype=float)
                    assert type(article_featureVector) == np.ndarray
                    assert article_featureVector.shape == (self.context_dimension,)
                    self.article_pool.append(Article(article_id, article_featureVector))

                # Random strategy
                RandomPicked = choice(self.article_pool)
                if RandomPicked.article_id == article_id_chosen:
                    reward = 1
                else:
                    reward = 0  # avoid division by zero
                AlgReward["random"].append(reward)

                for alg_name, alg in algorithms.items():
                    # Observe the candiate arm pool and algoirhtm makes a decision
                    pickedArticle = alg.decide(self.article_pool, userID)

                    # Get the feedback by looking at whether the selected arm by alg is the same as that of ground truth
                    if pickedArticle.article_id == article_id_chosen:
                        reward = 1
                    else:
                        reward = 0
                    # The feedback/observation will be fed to the algorithm to further update the algorithm's model estimation
                    alg.updateParameters(pickedArticle, reward, userID)
                    if alg_name == 'CLUB':
                        n_components = alg.updateGraphClusters(userID, 'False')
                    # Record the reward
                    AlgReward[alg_name].append(reward)

                if iter_ % self.batchSize == 0:
                    self.batchRecord(iter_)
                    tim_.append(iter_)
                    BatchCumlateReward["random"].append(sum(AlgReward["random"]))
                    for alg_name in algorithms.keys():
                        BatchCumlateReward[alg_name].append(sum(AlgReward[alg_name]))

                    # for alg_name in BatchCumlateReward.keys():
                    #     print(BatchCumlateReward[alg_name][-1])

                    if self.Write_to_File:
                        with open(filenameWriteReward, 'a+') as f:
                            f.write(str(iter_))
                            f.write(',' + ','.join([str(BatchCumlateReward[alg_name][-1]) for alg_name in
                                                    list(algorithms.keys()) + ["random"]]))
                            f.write('\n')
                iter_ += 1

        cp_path = os.path.join(self.save_address, "detectedChangePoints" + str(self.namelabel) + str(timeRun) + '.txt')

        end_num = 0
        while os.path.exists(cp_path):
            cp_path = os.path.join(self.save_address,
                                   "detectedChangePoints" + str(self.namelabel) + str(timeRun) + str(end_num) + '.txt')
            end_num += 1

        with open(cp_path, "w") as text_file:

            for alg_name in algorithms.keys():
                if 'adTS' in alg_name:
                    print("============= adTS detected change =============", file=text_file)
                    for u in userIDSet:
                        print("User {} detected change points: {}".format(u, algorithms[alg_name].users[u].changes),
                              file=text_file)
                if 'dLinUCB' in alg_name:
                    print("============ dLinUCB detected change ===========", file=text_file)
                    for u in userIDSet:
                        print("User {} detected change points: {}".format(u, algorithms[alg_name].users[u].newUCBs),
                              file=text_file)
                if 'dCLUB' in alg_name:
                    print("============== {} detected change ==============".format(alg_name), file=text_file)
                    for u in userIDSet:
                        print("User {} detected change points: {}".format(u, algorithms[alg_name].users[
                            u].detectedChangePoints), file=text_file)

        if self.Plot:  # only plot
            linestyles = ['o-', 's-', '*-', '>-', '<-', 'g-', '.-', 'o-', 's-', '*-']
            markerlist = ['.', ',', 'o', 's', '*', 'v', '>', '<']

            fig, ax = plt.subplots()

            count = 0
            for alg_name, alg in algorithms.items():
                labelName = alg_name
                ax.plot(tim_, [x for x in BatchCumlateReward[alg_name]], linewidth=2, marker=markerlist[count],
                        markevery=400, label=labelName)
                count += 1
            count += 1
            ax.legend(loc='upper right')
            ax.set(xlabel='Iteration', ylabel='Reward',
                   title='Reward over iterations')
            ax.grid()
            plt_path = os.path.join(self.save_address, str(self.namelabel) + str(timeRun) + '.png')

            end_num = 0
            while os.path.exists(plt_path):
                plt_path = os.path.join(self.save_address, str(self.namelabel) + str(timeRun) + str(end_num) + '.png')
                end_num += 1

            plt.savefig(plt_path)
            plt.show()
        for alg_name in algorithms.keys():
            print('%s: %.2f' % (alg_name, BatchCumlateReward[alg_name][-1]))
        return
