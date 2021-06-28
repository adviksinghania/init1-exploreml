# INIT1-ExploreML

Exploring simple linear regression using Python3.

## What it is

It’s an algorithm used by many in introductory machine learning, but it doesn’t require any “learning”. It’s as simple as plugging few values into a formula. In general, linear regression is used to predict continuous variables - something such as stock price, weight, and similar. Linear regression is a linear algorithm, meaning the linear relationship between input variables (what goes in) and the output variable (the prediction) is assumed.

The algorithm is also rather strict on the requirements.

- Linear Assumption - model assumes the relationship between variables is linear
- No Noise - model assumes that the input and output variables are not noisy - so remove outliers if possible
- No Collinearity - model will overfit when you have highly correlated input variables
- Normal Distribution - the model will make more reliable predictions if your input and output variables are normally distributed.
- Rescaled Inputs - use scalers or normalizer to make more reliable predictions

## Explanation

We need to solve the linear equation of the form _y = B<sub>0</sub> + B<sub>1</sub>x_. Where _B<sub>0</sub>_ is the constant and _B<sub>1</sub>_ is the slope.
The slope can be found using the formula:

![Formula](https://miro.medium.com/max/705/1*UZ2HPCd8hT54QE_yYuLuww.png)

The _X<sub>i</sub>_ represents the current value of the input feature, and _X_ with a bar on top represents the mean of the entire variable. The same goes with _Y_, but we’re looking at the target variable instead.

And then the constant can be found using:

![Formula](https://miro.medium.com/max/570/1*SQQSb1D0mz0oQ1hNbWjcVQ.png)

<br>

**NOTE:** This repository/project was made by following the article on [Simple Linear Regression](https://towardsdatascience.com/master-machine-learning-simple-linear-regression-from-scratch-with-python-1526487c5964) by **Dario Radečić**
