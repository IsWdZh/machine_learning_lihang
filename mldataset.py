# /usr/bin/python
# coding:utf-8

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def iris():
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
    plt.legend()
    
    data = np.array(df.iloc[:100, [0, 1, -1]]) # 选择前100行，第0,1以及最后一列
    X, Y = data[:,:-1], data[:,-1]     # x have two features, y is label
    # Y = np.array([1 if i == 1 else -1 for i in Y])    # change label 0 to -1

	# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
    return X, Y

