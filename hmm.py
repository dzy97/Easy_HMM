# -*- coding:utf-8 -*-
# 隐马尔科夫链模型
# By tostq <tostq216@163.com>
# 博客: blog.csdn.net/tostq

import numpy as np
from abc import ABCMeta, abstractmethod


class _BaseHMM():
    """
    基本HMM虚类，需要重写关于发射概率的相关虚函数
    n_state : 隐藏状态的数目
    n_iter : 迭代次数
    x_size : 观测维度
    start_prob : 初始概率
    transmat_prob : 状态转换概率
    """
    __metaclass__ = ABCMeta  # 虚类声明

    def __init__(self, start_prob_, transmat_prob_, n_state=1, x_size=1):
        self.n_state = n_state
        self.x_size = x_size
        self.start_prob = start_prob_
        self.transmat_prob = transmat_prob_

    # 虚函数：返回发射概率
    @abstractmethod
    def emit_prob(self, x):  # 求x在状态k下的发射概率 P(X|Z)
        return np.array([0])

    # 虚函数
    @abstractmethod
    def generate_x(self, z):  # 根据隐状态生成观测值x p(x|z)
        return np.array([0])

    # 通过HMM生成序列
    def generate_seq(self, seq_length):
        X = np.zeros((seq_length, self.x_size))
        Z = np.zeros(seq_length)
        Z_pre = np.random.choice(self.n_state, 1, p=self.start_prob)  # 采样初始状态
        X[0] = self.generate_x(Z_pre)  # 采样得到序列第一个值
        Z[0] = Z_pre

        for i in range(seq_length):
            if i == 0: continue
            # 当前状态为Z_pre，根据transmat采样得到下一个状态
            Z_next = np.random.choice(self.n_state, 1, p=self.transmat_prob[Z_pre, :])
            Z_pre = Z_next
            # 根据当前状态Z_pre采样得到发射的观测值
            X[i] = self.generate_x(Z_pre)
            Z[i] = Z_pre

        return X, Z

    # 估计序列X出现的概率
    def X_prob(self, X, Z_seq=np.array([])):
        # 状态序列预处理
        # 判断是否已知隐藏状态
        X_length = len(X)
        if Z_seq.any():
            Z = np.zeros((X_length, self.n_state))
            for i in range(X_length):
                Z[i][int(Z_seq[i])] = 1
        else:
            Z = np.ones((X_length, self.n_state))
        # 向前向后传递因子
        _, c = self.forward(X, Z)  # P(x,z)
        # 序列的出现概率估计
        prob_X = np.sum(np.log(c))  # log P(X)
        return prob_X

    # 已知当前序列预测未来（下一个）观测值的概率
    def predict(self, X, x_next, Z_seq=np.array([]), istrain=True):
        X_length = len(X)
        if Z_seq.any():
            Z = np.zeros((X_length, self.n_state))
            for i in range(X_length):
                Z[i][int(Z_seq[i])] = 1
        else:
            Z = np.ones((X_length, self.n_state))
        # 向前向后传递因子
        alpha, _ = self.forward(X, Z)  # P(x,z)
        prob_x_next = self.emit_prob(np.array([x_next])) * np.dot(alpha[X_length - 1], self.transmat_prob)
        return prob_x_next

    def decode(self, X):
        """
        利用维特比算法，已知序列求其隐藏状态值
        :param X: 观测值序列
        :return: 隐藏状态序列
        """

        X_length = len(X)  # 序列长度
        state = np.zeros(X_length)  # 隐藏状态

        pre_state = np.zeros((X_length, self.n_state))  # 保存转换到当前隐藏状态的最可能的前一状态
        max_pro_state = np.zeros((X_length, self.n_state))  # 保存传递到序列某位置当前状态的最大概率

        _, c = self.forward(X, np.ones((X_length, self.n_state)))
        max_pro_state[0] = self.emit_prob(X[0]) * self.start_prob * (1 / c[0])  # 初始概率

        # 前向过程
        for i in range(X_length):
            if i == 0: continue
            for k in range(self.n_state):
                prob_state = self.emit_prob(X[i])[k] * self.transmat_prob[:, k] * max_pro_state[i - 1]
                max_pro_state[i][k] = np.max(prob_state) * (1 / c[i])
                pre_state[i][k] = np.argmax(prob_state)

        # 后向过程
        state[X_length - 1] = np.argmax(max_pro_state[X_length - 1, :])
        for i in reversed(range(X_length)):
            if i == X_length - 1: continue
            state[i] = pre_state[i + 1][int(state[i + 1])]

        return state

    # 求向前传递因子
    def forward(self, X, Z):
        X_length = len(X)
        alpha = np.zeros((X_length, self.n_state))  # P(x,z)
        alpha[0] = self.emit_prob(X[0]) * self.start_prob * Z[0]  # 初始值
        # 归一化因子
        c = np.zeros(X_length)
        c[0] = np.sum(alpha[0])
        alpha[0] = alpha[0] / c[0]
        # 递归传递
        for i in range(X_length):
            if i == 0: continue
            alpha[i] = self.emit_prob(X[i]) * np.dot(alpha[i - 1], self.transmat_prob) * Z[i]
            c[i] = np.sum(alpha[i])
            if c[i] == 0: continue
            alpha[i] = alpha[i] / c[i]

        return alpha, c

    # 求向后传递因子
    def backward(self, X, Z, c):
        X_length = len(X)
        beta = np.zeros((X_length, self.n_state))  # P(x|z)
        beta[X_length - 1] = np.ones((self.n_state))
        # 递归传递
        for i in reversed(range(X_length)):
            if i == X_length - 1: continue
            beta[i] = np.dot(beta[i + 1] * self.emit_prob(X[i + 1]), self.transmat_prob.T) * Z[i]
            if c[i + 1] == 0: continue
            beta[i] = beta[i] / c[i + 1]

        return beta


class DiscreteHMM(_BaseHMM):
    """
    发射概率为离散分布的HMM
    参数：
    emit_prob : 离散概率分布
    x_num：表示观测值的种类
    此时观测值大小x_size默认为1
    """

    def __init__(self, start_prob_, transmat_prob_, emission_prob_, n_state=1, x_num=1):
        _BaseHMM.__init__(self, start_prob_, transmat_prob_, n_state=n_state, x_size=1)
        # self.emission_prob = np.ones((n_state, x_num)) * (1.0 / x_num)  # 初始化发射概率均值
        self.emission_prob = emission_prob_
        self.x_num = x_num

    def emit_prob(self, x):  # 求观测到x的状态的概率分布
        prob = np.zeros(self.n_state)
        for i in range(self.n_state): prob[i] = self.emission_prob[i][int(x[0])]
        return prob

    def generate_x(self, z):  # 根据状态z生成x p(x|z)
        return np.random.choice(self.x_num, 1, p=self.emission_prob[z])
