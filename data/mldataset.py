# /usr/bin/python
# coding:utf-8

import pandas as pd
import numpy as np
import random
import os
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def iris(show=True):
    iris = load_iris()

    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']   # add a 'label' col
    df.label.value_counts()
    
    plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], label='0')
    plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], label='1')
    plt.title('Dataset')
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    if show:
        plt.legend()
    
    data = np.array(df.iloc[:100, [0, 1, -1]]) # 选择前100行，第0,1以及最后一列
    X, Y = data[:,:-1], data[:,-1]     # x have two features, y is label
    # Y = np.array([1 if i == 1 else -1 for i in Y])    # change label 0 to -1

	# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
    return X, Y

def digit():
    file_path = os.path.join(os.getcwd(), "data", "digit_train.csv")
    data = []
    with open(file_path, "r") as f:
        lines = f.readlines()
        title = lines[0]   # label, pixel1, pixel2, ...pixel783
        for line in lines[1:]:
            if line[0] in ("0", "1"):
                data.append(line.strip())
    
    data_num = len(data)
    # data = list(map(float, data))
    # print(data[:2])

    # data divided by 8:2, 即取1/5数据作为TestSet
    testdata = random.sample(range(data_num), (int)(data_num/5))

    X_train, Y_train, X_test, Y_test = [], [], [], []
    for i in range(data_num):
        temp = np.array(list(map(float, data[i].split(","))), dtype=np.float32)
        if i in testdata:
            X_test.append(temp[1:])
            Y_test.append(temp[0])

        else:
            X_train.append(temp[1:])
            Y_train.append(temp[0])
    
    X_train = np.array(X_train, dtype=np.float32)
    X_test = np.array(X_test, dtype=np.float32)
    Y_train = np.array(Y_train, dtype=np.float32)
    Y_test = np.array(Y_test, dtype=np.float32)

    return X_train, X_test, Y_train, Y_test

# X_train, X_test, Y_train, Y_test = digit()
# print(X_train.shape)

