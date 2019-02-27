import random

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

df = pd.read_csv("./iris.data", header=None)
y = df.iloc[:, 4].values

le = LabelEncoder()
y_i = le.fit_transform(y)
x = df.iloc[:, [2, 3]].values

# 留出法
score = 0.0
for i in range(10):
    x_train, x_test, y_train, y_test = train_test_split(
        x, y_i, test_size=0.3, stratify=y)
    sc = StandardScaler()
    sc.fit(x_train)
    x_train_std = sc.transform(x_train)
    x_test_std = sc.transform(x_test)

    lr = LogisticRegression(C=100.0, random_state=1)
    lr.fit(x_train_std, y_train)
    s = lr.score(x_test_std, y_test)
    print(s)

# 交叉验证法

y = y_i

kfold = StratifiedKFold(n_splits=10, random_state=1).split(x, y)

lr = LogisticRegression(C=100.0, random_state=1)

print('======================')

for k, (train, test) in enumerate(kfold):
    lr.fit(x[train], y[train])
    score = lr.score(x[test], y[test])
    print(score)

# 自主法

times = 10

y = y_i
num = len(y)
all_index = set(np.arange(num))

lr = LogisticRegression(C=100.0, random_state=1)


def BootStrap(num):
    s = []
    while len(s) < num:
        p = random.randrange(0, num)
        s.append(p)
    return s


def GetDataByIndex(x, y, index):
    new_x = []
    new_y = []

    for i in index:
        new_x.append(x[i])
        new_y.append(y[i])

    return new_x, new_y


lr = LogisticRegression(C=100.0, random_state=1)

print("------------------------")

for i in range(times):
    print()
    train_index = BootStrap(num)
    unique_index = list(set(train_index))
    test_index = list(all_index - set(unique_index))

    x_train, y_train = GetDataByIndex(x, y, train_index)
    x_test, y_test = GetDataByIndex(x, y, test_index)

    lr.fit(x_train, y_train)
    s = lr.score(x_test, y_test)
    print(s)
