# Udacity Data Scientist Nano Degree

[TOC]

## Part 2: Deep Learning

### Introduction to NN

#### Error Function

> Error function describes the distance from current to the target -- optimized point
>
> And the direction and step towards the target is determined by gradient descent

![image-20190101102731144](../img/image-20190101102731144-6309651.png)

#### Activation Function

> Transform discrete output to continuous output, so that easier for optimization

**Sigmoid**

> Use sigmoid instead of step activation function, transforming yes/no to probability

![image-20190101103301949](../img/image-20190101103301949-6309981.png)

![image-20190101103338337](../img/image-20190101103338337-6310018.png)

**Softmax**

![image-20190101104744334](../img/image-20190101104744334-6310864.png)

#### Maximum Likelihood of Model

> The higher likelihood, the better model

![image-20190101114555113](../img/image-20190101114555113-6314355.png)

#### Cross Entropy of Model

> Connection between minimizing error function and maximizing likelihood

**1. We prefer sum to product --> use log**

**2. We prefer positive to negative --> minus**

**3. Therefore**

![image-20190101115041100](../img/image-20190101115041100-6314641.png)

**Multi-Class Cross-Entropy**

![image-20190101115440161](../img/image-20190101115440161-6314880.png)

#### Logistic Regression

![image-20190101115605266](../img/image-20190101115605266-6314965.png)

####Gradient Descent

![image-20190102082206882](../img/image-20190102082206882-6388526.png)

##### Perceptron vs Gradient Descent of Logistic Regression

![image-20190102084649344](../img/image-20190102084649344-6390009.png)

#### Neural Network Architecture

> Combining multiple linear models **(probability graphs)** into one non-linear model

![image-20190102085424872](../img/image-20190102085424872-6390464.png)