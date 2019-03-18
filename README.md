# ML

## 学习

学习的内容都是前人思想的结晶,在理解的时候,每个人想法不同,会有理解上的偏差,但是总的来说是好的,站在巨人的肩膀上

就好比如果没有前人,就不会有计算机,但是计算机是必然会出现的产物,如果我们这一代才开始研究计算机,那么社会的发展不就滞后了吗?这就是传承的意义所在,而且后人不一定都需要知道计算机的原理,但是照样可以使用计算机,就好比用计算机打游戏,了解原理和使用需要的时间和精力都是不同的,不同的侧重点也就意味着不同的选择

## 流程

1. 读取数据,观察数据
2. 处理数据,处理数据空缺和数值化
3. 分割数据集,训练集/测试集
3. 标准化
4. 选择模型,训练
5. 模型评估
6. 预测

## 距离和相似性度量

### 距离

1. [欧几里得距离](https://zh.wikipedia.org/zh-hans/%E6%AC%A7%E5%87%A0%E9%87%8C%E5%BE%97%E8%B7%9D%E7%A6%BB)
2. [minkowski distance](https://zh.wikipedia.org/wiki/%E6%98%8E%E6%B0%8F%E8%B7%9D%E7%A6%BB)
3. [马氏距离](https://zh.wikipedia.org/wiki/%E9%A9%AC%E5%93%88%E6%8B%89%E8%AF%BA%E6%AF%94%E6%96%AF%E8%B7%9D%E7%A6%BB)
4. [汉明距离](https://zh.wikipedia.org/wiki/%E6%B1%89%E6%98%8E%E8%B7%9D%E7%A6%BB)
5. [余弦相似性](https://zh.wikipedia.org/wiki/%E4%BD%99%E5%BC%A6%E7%9B%B8%E4%BC%BC%E6%80%A7)
6. [编辑距离](https://zh.wikipedia.org/wiki/%E7%B7%A8%E8%BC%AF%E8%B7%9D%E9%9B%A2)
7. [雅卡尔距离](https://zh.wikipedia.org/wiki/%E9%9B%85%E5%8D%A1%E5%B0%94%E6%8C%87%E6%95%B0)

### 相似性

1. [KL散度](https://zh.wikipedia.org/wiki/%E7%9B%B8%E5%AF%B9%E7%86%B5)
2. [信息熵](https://zh.wikipedia.org/wiki/%E7%86%B5_(%E4%BF%A1%E6%81%AF%E8%AE%BA))


## 分类和回归

分类:
给定一组数据,训练模型,最后给定不知分类的数据,希望通过这个模型给出预测的分类

回归
给定一组数据,训练模型,最后给定不知结果的数据,希望通过这个模型可以预测结果

区别:
分类的结果是离散的
回归的结果是连续的

## 数据预处理

### 缺失数据
1. 删除空数据
2. 通过均值补缺

### 类别数据
1. 可以直接转化为数值数据,这类数据可以比较
2. 没有大小之分,可以直接转化为bool类型,如果有多个类别则直接拆分为多列

### 归一化
将特征取值放到`[0,1]`范围内

### 标准化
数据通过处理后,否和标准正太分布,这样可以让模型快速收敛


## 线性模型

### 线性回归

1. 最小二乘法, 给定数据可以曲线拟合
2. 梯度下降法, 寻找最小值
3. 房价预测, housing数据

### 逻辑回归

1. 极大似然法
2. 花分类, iris数据


### 评估模型

1. MSE均方误差
2. R2 Score


## 决策树

1. 熵, 信息论
2. 树, 训练的过程就是构建树,分类的过程就是搜索


## KNN

1. 度量数据距离
2. 邻居个数k选择
3. 维度增加,过拟合风险增加,可通过降维和特征选择解决过拟合












