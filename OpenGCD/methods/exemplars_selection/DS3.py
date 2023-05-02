"""
This file contains the implementation of 'Dissimilarity-based Sparse Subset Selection' algorithm using different
types of optimization techniques such as, message passing, greedy algorithm, and ADMM.
"""
from methods.exemplars_selection.ADMM import ADMM
from numpy import linalg as LA
import numpy as np


class DS3(object):
    """
    :param dis_matrix:  dis-similarity matrix for the dataset calculated based on euclideon distance.
    :param reg:         regularization parameter

    """
    def __init__(self, dis_matrix, reg):
        self.reg = reg
        self.dis_matrix = dis_matrix
        self.N = len(self.dis_matrix)

    def regCost(self, z, p):
        """
        This function calculates the total cost of choosing the as few representatives as possible.

        :param z: matrix whose non-zero rows corresponds to the representatives of the dataset.
        :param p: norm to be used to calculate regularization cost.

        :returns: regularization cost.
        """

        cost = 0
        for i in range(len(self.dis_matrix)):
            norm = LA.norm(z[i], ord=p)
            cost += norm

        return cost * self.reg

    def encodingCost(self, z):
        """
        This function calculates the total cost of encoding using all the representatives.

        :param z: matrix whose non-zero rows corresponds to the representatives of the dataset.

        :returns: encoding cost.
        """

        cost = 0
        for j in range(len(self.dis_matrix)):
            for i in range(len(self.dis_matrix)):
                try:
                    cost += self.dis_matrix[i, j] * z[i, j]
                except:
                    break

        return cost

    def transitionCost(self, z, M, m0):
        """
        This function calculates the total cost of transitions between the representatives.

        :param z:  matrix whose non-zero rows corresponds to the representatives of the dataset.
        :param M:  transition probability matrix for the states in the source set.
        :param m0: initial probability vector of the states in the source set.

        :returns: transition cost.
        """

        sum1 = 0
        for i in range(1, self.N):
            sum1 += np.matmul(np.matmul(np.transpose(z[:,(i-1)]), M), z[:, i])
        sum2 = np.matmul(z[:, 1], m0)

        return sum1 + sum2


    def ADMM(self, mu, epsilon, max_iter, p, k):
        """
        This function finds the subset of the data that can represent it as closely as possible given the
        regularization parameter. It uses 'alternating direction methods of multipliers' (ADMM) algorithm to
        solve the objective function for this problem, which is similar to the popular 'facility location problem'.

        To know more about this, please read :
        Dissimilarity-based Sparse Subset Selection
        by Ehsan Elhamifar, Guillermo Sapiro, and S. Shankar Sastry
        https://arxiv.org/pdf/1407.6810.pdf

        :param mu:        penalty parameter.
        :param epsilon:   small value to check for convergence.
        :param max_iter:  maximum number of iterations to run this algorithm.
        :param p:         norm to be used.

        :returns: representative of the data, total number of representatives, and the objective function value.
        """

        # initialize the ADMM class.
        G = ADMM(mu, epsilon, max_iter, self.reg)

        # run the ADMM algorithm.
        z_matrix = G.runADMM(self.dis_matrix, p)

        # A larger value of the row sum means that the sample corresponding to this row is more representative of the target set
        row_sum = np.sum(z_matrix, axis=1)
        score_sort = np.argsort(row_sum)
        data_rep = score_sort[-k:]
        return data_rep, len(data_rep)