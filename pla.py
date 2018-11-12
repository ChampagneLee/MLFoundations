"""
    Machine Learning Foundation >> Homework 1 >> PLA
    Written by Xiangbin on Nov 12, 2018
"""
import pdb
import copy
import numpy as np

def sign(x):
    if x <= 0:
        return -1
    else:
        return 1

def pla(weights, x, y, lr=1., log=False):
    """
        Do naive cycle of PLA.
        Inputs:
            weights: Weights to be updated. || ndarray, shape=(5,).
            x: training examples with five features, in which x[0] represents bias. || ndarray, shape=(N, 5).
            y: training labels. || ndarray, shape=(N,).
            lr: learning rate. || float.
            log: whether to print information during training. || bool.
        Outputs:
            weights: weights after training. || ndarray, shape=(5,).
            nrof_updates: the number of updates to do with weights. || integer.
    """
    assert x.shape[0] == y.shape[0]
    nrof_training = x.shape[0]

    nrof_runs = 0       #Record the number of examples classified right since last update.
    nrof_updates = 0    #Record the number of updates has done.
    i = 0               #Total iterations.

    while True:
        if nrof_runs == nrof_training:
            break

        if log:
            print(i)
        x_single = x[i % nrof_training]
        y_single_hat = np.dot(x_single, weights)
        if sign(y_single_hat) != y[i % nrof_training]:
            weights += y[i % nrof_training] * lr * x_single
            nrof_updates += 1
            nrof_runs = 0
        else:
            nrof_runs += 1

        i += 1

    return weights, nrof_updates

def random_pla(weights, x, y, max_num=2000):
    """
        Do PLA using shuffled x and y with max_num iterations.
        Inputs:
            weights: weights to be updated. || ndarray, shape=(5,).
            x: training examples. || ndarray, shape=(N, 5).
            y: training labels. || ndarray, shape=(N,).
            max_num: max number of iterations. || integer.
        output:
            The average of number of updates. || integer.
    """
    nrof_training = x.shape[0]
    nrof_updates = np.zeros((max_num,), dtype=int)
    for i in range(max_num):
        weights = np.zeros((5,))
        inds = np.random.permutation(nrof_training)
        x_random = x[inds, :]
        y_random = y[inds]
        _, nrof_updates[i] = pla(weights, x_random, y_random, 0.5)
        print('The number of updates in {:d}th iteration is {:d}'.format(i+1, nrof_updates[i]))
    return np.sum(nrof_updates) / max_num

def sign_array(y):
    res = np.ones((y.shape[0]))
    neg_mask = np.where(y <= 0)[0]
    res[neg_mask] = -1
    return res

def check_weights(dataset, weights):
    """
        Calculate error rate made by weights on dataset.
        Inputs:
            dataset: Metrics to do on. || tuple(x, y).
            weights: weights using to valuate. || ndarray, shape=(5,).
        Outputs:
            The error rate. || float.
    """
    x, y = dataset
    n = x.shape[0]
    y_hat = sign_array(np.dot(x, weights).reshape(n,))

    return 1 - np.sum(np.equal(y_hat, y)) / n

def pocket_pla_single(x, y, lr=1., max_updates=50):
    """
        Single pocket PLA algorithm.
        Inputs:
            x: training examples. || ndarray, shape=(N, 5).
            y: training labels. || ndarray, shape=(N,).
            lr: learning rate. || float.
            max_updates: the number of updates to do in one PLA process. || integer.
        Outputs:
            weights_pkt: the weights after updates. || ndarray, shape=(5,).
    """
    nrof_training = x.shape[0]
    weights = np.zeros((5,))
    weights_pkt = None
    rate_wrong = 1  #Normalized to one.
    nrof_updates = 0

    inds = np.random.permutation(nrof_training)
    x_random = x[inds, :]
    y_random = y[inds]
    for j in range(nrof_training):
        x_single = x_random[j]
        y_single = y_random[j]
        y_single_hat = np.dot(x_single, weights)
        if sign(y_single_hat) != y_single:
            weights += lr * y_single * x_single
            nrof_updates += 1

            cur_rate_wrong = check_weights((x, y), weights)
            if rate_wrong > cur_rate_wrong:
                weights_pkt = copy.deepcopy(weights)    #Notice that must use deepcopy because "+=" op will change weights_pkt
                rate_wrong = cur_rate_wrong
        if nrof_updates == max_updates:
            break

    return weights_pkt

def pocket_pla(train_set, test_set, max_iters=2000, random=True):
    """
        Do pocket PLA algorithm with max_iters iterations.
        Inputs:
            train_set: training set with (x_train, y_train), in which x_train is ndarray with shape(N, 5) and
                y_train is ndarray with shape(N). || tuple.
            test_set: test set with (x_test, y_test), in which x_test is ndarray with shape(N, 5) and
                y_test is ndarray with shape(N). || tuple.
            max_iters: max number of iterations. || integer.
            random: whether to shuffle dataset. || bool.

        Outputs:
            The average of error on test set. || float.
    """
    x_train, y_train = train_set
    assert x_train.shape[0] == y_train.shape[0]
    errors = np.zeros((max_iters,), dtype=float)

    for i in range(max_iters):
        weights_pkt = pocket_pla_single(x_train, y_train, 0.5, 50)
        errors[i] = check_weights(test_set, weights_pkt)

    return np.sum(errors) / max_iters

def read_set(dataset_path):
    with open(dataset_path, 'r') as f:
        lines = f.readlines()
    x_list = [[float(x) for x in line.split()[0:4]] for line in lines]
    y_list = [int(line.split()[-1]) for line in lines]

    nrof_training = len(x_list)
    x = np.array(x_list)
    y = np.array(y_list)
    bias = np.ones((nrof_training, 1))
    x = np.hstack([bias, x])

    return (x, y)

def main():
    """
        Main function.
    """
    #1. Read training examples from txt file.
    train_set_path = 'hw1_18_train.txt'
    test_set_path = 'hw1_18_test.txt'
    train_set = read_set(train_set_path)
    test_set = read_set(test_set_path)

    #2. Do PLA training.
    ave_error = pocket_pla(train_set, test_set)
    print(ave_error)

if __name__ == '__main__':
    main()
