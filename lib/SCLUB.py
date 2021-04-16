import numpy as np
from bidict import bidict

class Base:
    # Base agent for online clustering of bandits
    def __init__(self, d, NoiseScale, alpha, lambda_, delta_1=0.1):
        self.d = d
        self.NoiseScale = NoiseScale
        self.alpha = alpha
        self.lambda_ = lambda_
        self.delta_1 = delta_1
        # self.beta = np.sqrt(self.d * np.log(self.T / self.d)) # parameter for select item

    def _beta(self, N, t):
        return self.NoiseScale * np.sqrt(self.d * np.log(1 + N / (self.d * self.lambda_)) + 2 * np.log(1 / self.delta_1)) + np.sqrt(self.lambda_)
        # return np.sqrt(self.d * np.log(1 + N / self.d) + 4 * np.log(t) + np.log(2)) + 1

    def _select_item_ucb(self, S, Sinv, theta, items, N, t):
        if self.alpha is not None:
            return np.argmax(np.dot(items, theta) + self.alpha * (np.matmul(items, Sinv) * items).sum(axis = 1))
        else:
            return np.argmax(np.dot(items, theta) + self._beta(N, t) * (np.matmul(items, Sinv) * items).sum(axis = 1))

    def recommend(self, i, items, t):
        # items is of type np.array (L, d)
        # select one index from items to user i
        return

    def store_info(self, i, x, y, t):
        return

    def _update_inverse(self, S, b, Sinv, x, t):
        Sinv = np.linalg.inv(S)
        theta = np.matmul(Sinv, b)
        return Sinv, theta

    def update(self, t):
        return

    def run(self, envir):
        for t in range(self.T):
            if t % 5000 == 0:
                print(t // 5000, end = ' ')
            self.I = envir.generate_users()
            for i in self.I:
                items = envir.get_items()
                kk = self.recommend(i=i, items=items, t=t)
                x = items[kk]
                y= envir.feedback(i=i, k=kk)
                self.store_info(i=i, x=x, y=y, t=t)

            self.update(t)

        print()

class LinUCB_IND(Base):
    # each user is an independent LinUCB
    def __init__(self, nu, d, NoiseScale, alpha, lambda_):
        super(LinUCB_IND, self).__init__(d, NoiseScale, alpha, lambda_)
        self.S = {i:lambda_*np.eye(d) for i in range(nu)}
        self.b = {i:np.zeros(d) for i in range(nu)}
        self.Sinv = {i:(1/lambda_)*np.eye(d) for i in range(nu)}
        self.theta = {i:np.zeros(d) for i in range(nu)}

        self.N = np.zeros(nu)

    def recommend(self, i, items, t):
        return self._select_item_ucb(self.S[i], self.Sinv[i], self.theta[i], items, self.N[i], t)

    def store_info(self, i, x, y, t):
        self.S[i] += np.outer(x, x)
        self.b[i] += y * x
        self.N[i] += 1

        self.Sinv[i], self.theta[i] = self._update_inverse(self.S[i], self.b[i], self.Sinv[i], x, self.N[i])

class Cluster:
    def __init__(self, users, S, b, N, checks):
        self.users = users # a list/array of users
        self.S = S
        self.b = b
        self.N = N
        self.checks = checks
        
        self.Sinv = np.linalg.inv(self.S)
        self.theta = np.matmul(self.Sinv, self.b)
        self.checked = len(self.users) == sum(self.checks.values())

    def update_check(self, i):
        self.checks[i] = True
        self.checked = len(self.users) == sum(self.checks.values())
        
class SCLUB(LinUCB_IND):
    def __init__(self, nu, d, NoiseScale, lambda_, alpha=None):
        super(SCLUB, self).__init__(nu, d, NoiseScale, alpha, lambda_)

        self.clusters = {0:Cluster(users=[i for i in range(nu)], S=lambda_ * np.eye(d), b=np.zeros(d), N=0, checks={i:False for i in range(nu)})}
        self.cluster_inds = np.zeros(nu)

        self.global_time = 0
        self.stage_id = 0
        self.time_to_next_phase = 2**self.stage_id
        self._init_each_stage()

        self.userID2userIndex = bidict()
        self.cur_userIndex = 0

        # self.alpha = 4 * np.sqrt(d)
        # self.alpha_p = np.sqrt(4) # 2
        self.lambda_ = lambda_
        self.CanEstimateUserPreference = False
        self.CanEstimateUserCluster = True

    def _init_each_stage(self):
        for c in self.clusters:
            self.clusters[c].checks = {i:False for i in self.clusters[c].users}
            self.clusters[c].checked = False
            
    def recommend(self, i, items, t):
        cluster = self.clusters[self.cluster_inds[i]]
        return self._select_item_ucb(cluster.S, cluster.Sinv, cluster.theta, items, cluster.N, t)

    def store_info(self, i, x, y, t):
        super(SCLUB, self).store_info(i, x, y, t)

        c = self.cluster_inds[i]
        self.clusters[c].S += np.outer(x, x)
        self.clusters[c].b += y * x
        self.clusters[c].N += 1

        self.clusters[c].Sinv, self.clusters[c].theta = self._update_inverse(self.clusters[c].S, self.clusters[c].b, self.clusters[c].Sinv, x, self.clusters[c].N)

    def _factT(self, T):
            return np.sqrt((1 + np.log(1 + T)) / (1 + T))

    def _split_or_merge(self, theta, N1, N2, split=True):
        # alpha = 2 * np.sqrt(2 * self.d)
        alpha = 1
        if split:
            return np.linalg.norm(theta) >  alpha * (self._factT(N1) + self._factT(N2))
        else:
            return np.linalg.norm(theta) <  alpha * (self._factT(N1) + self._factT(N2)) / 2

    def _cluster_avg_freq(self, c, t):
        return self.clusters[c].N / (len(self.clusters[c].users) * t)

    def _split_or_merge_p(self, p1, p2, t, split=True):
        # def _pradius(t):
        #     return np.sqrt((np.log(4) + 4 * np.log(t)) / (2*t))
        alpha_p = np.sqrt(2)
        if split:
            return np.abs(p1-p2) > alpha_p * self._factT(t)
        else:
            return np.abs(p1-p2) < alpha_p * self._factT(t) / 2

    def split(self, i, t):
        c = self.cluster_inds[i]
        cluster = self.clusters[c]

        cluster.update_check(i)

        if self._split_or_merge_p(self.N[i]/(t+1), self._cluster_avg_freq(c, t+1), t+1, split=True) or self._split_or_merge(self.theta[i] - cluster.theta, self.N[i], cluster.N, split=True):

            def _find_available_index():
                cmax = max(self.clusters)
                for c1 in range(cmax + 1):
                    if c1 not in self.clusters:
                        return c1
                return cmax + 1

            cnew = _find_available_index()
            self.clusters[cnew] = Cluster(users=[i],S=self.S[i],b=self.b[i],N=self.N[i],checks={i:True})
            self.cluster_inds[i] = cnew

            cluster.users.remove(i)
            cluster.S = cluster.S - self.S[i] + self.lambda_*np.eye(self.d)
            cluster.b = cluster.b - self.b[i]
            cluster.N = cluster.N - self.N[i]
            del cluster.checks[i]

    def merge(self, t):
        cmax = max(self.clusters)

        for c1 in range(cmax + 1):
            if c1 not in self.clusters or self.clusters[c1].checked == False:
                continue

            for c2 in range(c1 + 1, cmax + 1):
                if c2 not in self.clusters or self.clusters[c2].checked == False:
                    continue

                if self._split_or_merge(self.clusters[c1].theta - self.clusters[c2].theta, self.clusters[c1].N, self.clusters[c2].N, split=False) and self._split_or_merge_p(self._cluster_avg_freq(c1, t+1), self._cluster_avg_freq(c2, t+1), t+1, split=False):

                    for i in self.clusters[c2].users:
                        self.cluster_inds[i] = c1

                    self.clusters[c1].users = self.clusters[c1].users + self.clusters[c2].users
                    self.clusters[c1].S = self.clusters[c1].S + self.clusters[c2].S - self.lambda_*np.eye(self.d)
                    self.clusters[c1].b = self.clusters[c1].b + self.clusters[c2].b
                    self.clusters[c1].N = self.clusters[c1].N + self.clusters[c2].N
                    self.clusters[c1].checks = {**self.clusters[c1].checks, **self.clusters[c2].checks}

                    del self.clusters[c2]

    def decide(self, pool_articles, userID):
        if userID not in self.userID2userIndex:
            self.userID2userIndex[userID] = self.cur_userIndex
            self.cur_userIndex += 1
        self._init_each_stage()
        # gather x of all articles into array num_items * d
        items = np.zeros((0, self.d))
        for article in pool_articles:
            items = np.concatenate((items, article.contextFeatureVector[:self.d].reshape(1, self.d)), axis=0)
        
        article_id = self.recommend(self.userID2userIndex[userID], items, self.global_time)

        self.cluster = []
        cid = self.cluster_inds[self.userID2userIndex[userID]]
        uind_cluster = np.where(self.cluster_inds==cid)[0].tolist()
        for uind in uind_cluster:
            if uind in self.userID2userIndex.inverse:
                self.cluster.append(self.userID2userIndex.inverse[uind])

        return pool_articles[article_id]

    def updateParameters(self, articlePicked, click, userID):

        self.store_info(self.userID2userIndex[userID], articlePicked.contextFeatureVector[:self.d], click, self.global_time)
        self.split(self.userID2userIndex[userID], self.global_time)
        self.merge(self.global_time)


        self.global_time += 1
        self.time_to_next_phase -= 1
        if self.time_to_next_phase == 0:
            self.stage_id += 1
            self.time_to_next_phase = 2**self.stage_id

    # def run(self, envir):
    #     for s in range(self.num_stages):
    #         print(s, end = ' ')
    #         for t in range(2 ** s):
    #             if t % 5000 == 0:
    #                 print(t // 5000, end = ' ')

    #             self._init_each_stage()
    #             tau = 2 ** s + t - 1

    #             i = envir.generate_users()[0]  # [oneuser]

    #             items = envir.get_items()  # array num_items * d
    #             kk = self.recommend(i, items, tau)
    #             x = items[kk]
    #             y = envir.feedback(i, kk)
    #             self.store_info(i, x, y, tau)

    #             # c = self.cluster_inds[i]
    #             self.split(i, tau)

    #             self.merge(tau)

    #     print()