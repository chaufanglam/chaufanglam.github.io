---
layout:     post
title:      "ML20笔记之Classification"
subtitle:   " \"Hello World, Hello ML\""
date:       2020-10-13
author:     "Chaufang"
header-img: "img/post-bg-2015.jpg"
tags:
    - 生活
    - ML20
---

# Classification: Probabilistic Generative Model

### 概述

分类问题就是找一个function，它的input是一个object，它的输出是这个object属于哪一个class。

还是以宝可梦为例，已知宝可梦有18种属性，现在要解决的分类问题就是做一个宝可梦种类的分类器，我们要找一个function，这个function的input是某一只宝可梦，它的output就是这只宝可梦属于这18类别中的哪一个type。

### Ideal Alternatives

#### Function(Model)

我们要找的function f(x)里面会有另外一个function g(x)，当我们的input x输入后，如果g(x)>0，那f(x)的输出就是class 1，如果g(x)<0，那f(x)的输出就是class 2，这个方法保证了function的output都是离散的表示class的数值

#### Loss Function

我们可以把loss function定义成$L(f) = \sum_{n}\delta(f(x^n)\neq \hat{y}^n)$，即这个model在所有的training data上predict预测错误的次数，也就是说分类错误的次数越少，这个function表现得就越好

但是这个loss function没有办法微分，是无法用gradient descent的方法去解的，当然有Perceptron、SVM这些方法可以用，但这里先用另外一个solution来解决这个问题

![alt 图片](https://raw.githubusercontent.com/chaufanglam/MarkdownPhotos/master/ideal-alternatives.png)

#### Solution: Generative Model

由全概率公式，可以得到下图中的概率$P(x \vert C_1)$以及$P(x\vert C_2)$，分别表示拿到一个input x属于class 1和class 2的概率。为了计算出概率，需要知道以下四个值：$P(C_1),P(C_2),P(x\vert C_1),P(x\vert C_2)$，我们希望从Training data中估测这四个值。

![alt 图片](https://raw.githubusercontent.com/chaufanglam/MarkdownPhotos/master/two-class.png)

#### Prior

$P(x\vert C_1)$和$P(x\vert C_2)$这两个概率，被称为Prior，可以直接从Training data中的概率得到。

问题是怎么得到$P(x\vert C_1),P(x\vert C_2)$的值。

#### Probability from Class

我们通过假设这些Training data是从一个**Gaussian Distribution**中sample出来的，而Gaussian Distribution的概率密度函数=

$
f_{\mu,\Sigma}(x)=\frac{1}{(2\pi)^{D/2}} \frac{1}{\vert\Sigma\vert^{1/2}}exp(-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu))
$

其中，$\mu$ 表示均值，$\Sigma$表示方差，两者都是矩阵。

这个方法之所以称为**Generative Model**，是因为他预先假设这些Training data都是来自Gaussian Distribution的。

所以现在的任务就是估计出这个Gaussian Distribution的均值$\mu$ 和方差$\Sigma$

估计这两个参数的方法是**极大似然估计法(Maximum Likelihood)**，极大似然估计的思想是，找出最特殊的那对均值$\mu$ 和方差$\Sigma$，从它们共同决定的高斯函数中再次采样出79个点，使”得到的分布情况与当前已知点的分布情况相同“这件事情发生的可能性最大

经过推导得，最合适的均值$\mu$ 和方差$\Sigma$分别是

$
\mu=E(X), \Sigma=cov(X,X)
$

通过上式我们计算出class 1和class 2的均值$\mu$ 和方差$\Sigma$，得到了class 1和class 2估计Gaussian Distribution的概率密度函数。

#### Do Classification！

![alt 图片](https://raw.githubusercontent.com/chaufanglam/MarkdownPhotos/master/do-classification.png)

只要带入某一个input x，就可以通过这个式子计算出它是否是class 1了！

#### Modifying Model

事实上，**不同的class可以share同一个cocovariance matrix**，这样反而可以达到更好的效果。

其实variance是跟input的feature size的平方成正比的，所以当feature的数量很大的时候，$\Sigma$大小的增长是可以非常快的，在这种情况下，给不同的Gaussian以不同的covariance matrix，会造成model的参数太多，而参数多会导致该model的variance过大，出现overfitting的现象，因此对不同的class使用同一个covariance matrix，可以有效减少参数

![alt 图片](https://raw.githubusercontent.com/chaufanglam/MarkdownPhotos/master/modify-model.png)

此时，把$\mu_1$、$\mu_2$和共同的$\Sigma$一起合成一个极大似然函数，然后可以发现，得到的$\mu_1$、$\mu_2$和原来的一样，而$\Sigma$则是由原先的$\Sigma_1$和$\Sigma_2$加权求和得到。

再来看一下结果，你会发现，class 1和class 2在没有共用covariance matrix之前，它们的分界线是一条曲线；如果共用covariance matrix的话，它们之间的分界线就会变成一条直线，这样的model，我们也称之为linear model(尽管Gaussian不是linear的，但是它分两个class的boundary是linear)

![alt 图片](https://raw.githubusercontent.com/chaufanglam/MarkdownPhotos/master/modify-compare.png)

#### Three Steps of Classification

现在让我们来回顾一下做classification的三个步骤，实际上也就是做machine learning的三个步骤

- Find a function set (Model)

  这些required probability $P(C)$和probability distribution $P(x\vert C)$就是model的参数，选择不同的Probability distribution(比如不同的分布函数，或者是不同参数的Gaussian distribution)，就会得到不同的function，把这些不同参数的Gaussian distribution集合起来，就是一个model，如果不适用高斯函数而选择其他分布函数，就是一个新的model了

  当这个posterior Probability $P(C_1\vert x)>0.5$的话，就output class 1，反之就output class 2，因为$P(C_1\vert x)+P(C_2\vert x)=1$，所以没必要对class 2再去计算一遍

- Goodness of function

  对于Gaussian distribution这个model来说，我们要评价的是决定这个高斯函数形状的均值$\mu$和协方差$\Sigma$这两个参数的好坏，而极大似然函数的输出值$L(\mu,\Sigma)$，就评价了这组参数的好坏

- Find the best function

  找到的那个最好的function，就是使$L(\mu,\Sigma)$值最大的那组参数，实际上就是所有样本点的均值$\mu^*=\frac{1}{n}\sum_{i=0}^nx^i$和协方差$\Sigma^*=\frac{1}{n}\sum_{i=0}^n(x^i-\mu^*)(x^i-\mu^*)^T$
  
  这里上标i表示第i个点，这里x是一个features的vector，用下标来表示这个vector中的某个feature
  
  ![alt 图片](https://raw.githubusercontent.com/chaufanglam/MarkdownPhotos/master/three-steps.png)