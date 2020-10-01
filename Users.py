import numpy as np
from util_functions import featureUniform, gaussianFeature, fileOverWriteWarning
import json
from random import choice, randint

class User():
    def __init__(self, id, theta = None, CoTheta = None):
        self.id = id
        self.theta = theta

class UserManager():
    def __init__(self, dimension, userNum, gamma, UserGroups, thetaFunc, argv = None):
        self.dimension = dimension
        self.thetaFunc = thetaFunc
        self.userNum = userNum
        self.gamma = gamma
        self.UserGroups = UserGroups
        self.argv = argv
        self.signature = "A-"+"+PA"+"+TF-"+self.thetaFunc.__name__

    def generateMasks(self):
        mask = {}
        for i in range(self.UserGroups):
            mask[i] = np.random.randint(2, size = self.dimension)
        return mask

    def simulateThetafromUsers_original(self):
        usersids = {}
        users = []
        mask = self.generateMasks()
        global_parameter_set = []
        if (self.UserGroups == 0):
            for key in range(self.userNum):
                thetaVector = self.thetaFunc(self.dimension, argv = self.argv)
                l2_norm = np.linalg.norm(thetaVector, ord =2)
                new_theta = thetaVector / l2_norm
                if global_parameter_set == []:
                    global_parameter_set.append(new_theta)
                else:
                    dist_to_all_existing_big = all([np.linalg.norm(new_theta - existing_theta) >= 0.7 for existing_theta in global_parameter_set])
                    while (not dist_to_all_existing_big):
                        thetaVector = self.thetaFunc(self.dimension, argv=self.argv)
                        l2_norm = np.linalg.norm(thetaVector, ord=2)
                        new_theta = thetaVector / l2_norm
                        dist_to_all_existing_big = all(
                            [np.linalg.norm(new_theta - existing_theta) >= 0.7 for existing_theta in
                             global_parameter_set])
                    global_parameter_set.append(new_theta)
                users.append(User(key, thetaVector / l2_norm))
        else:
            for i in range(self.UserGroups):
                usersids[i] = range(self.userNum*i/self.UserGroups, (self.userNum*(i+1))/self.UserGroups)

                for key in usersids[i]:
                    thetaVector = np.multiply(self.thetaFunc(self.dimension, argv = self.argv), mask[i])
                    l2_norm = np.linalg.norm(thetaVector, ord =2)
                    users.append(User(key, thetaVector/l2_norm))
        return users

    def simulateThetafromUsers(self):
        users = []
        # Generate a global unique parameter set
        global_parameter_set = []
        for i in range(self.UserGroups):
            thetaVector = self.thetaFunc(self.dimension, argv=self.argv)
            l2_norm = np.linalg.norm(thetaVector, ord=2)
            new_theta = thetaVector / l2_norm

            if global_parameter_set == []:
                global_parameter_set.append(new_theta)
            else:
                dist_to_all_existing_big = all([np.linalg.norm(new_theta - existing_theta) >= self.gamma for existing_theta in global_parameter_set])
                while (not dist_to_all_existing_big):
                    thetaVector = self.thetaFunc(self.dimension, argv=self.argv)
                    l2_norm = np.linalg.norm(thetaVector, ord=2)
                    new_theta = thetaVector / l2_norm
                    dist_to_all_existing_big = all(
                        [np.linalg.norm(new_theta - existing_theta) >= self.gamma for existing_theta in
                         global_parameter_set])
                global_parameter_set.append(new_theta)
        global_parameter_set = np.array(global_parameter_set)
        assert global_parameter_set.shape == (self.UserGroups, self.dimension)
        # Uniformly sample a parameter for each user as initial parameter
        parameter_index_for_users = np.random.randint(self.UserGroups, size=self.userNum)
        print(parameter_index_for_users)

        for key in range(self.userNum):
            parameter_index = parameter_index_for_users[key]
            users.append(User(key, global_parameter_set[parameter_index]))
            assert users[key].id == key
            assert np.linalg.norm(global_parameter_set[parameter_index] - users[key].theta) <= 0.001

        return users, global_parameter_set, parameter_index_for_users

