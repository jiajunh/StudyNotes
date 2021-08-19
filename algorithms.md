

[toc]

## Logistic Regressiom

* 重要思路：用$sigmoid()$ 来代替线性回归，$y\in \left\{ {1,0} \right\}$

  普通线性回归：$y=w^T x$ 

  LR: $y=sigmoid(w^Tx)=\frac{1}{1-e^{-w^Tx}}$

* Logistic Regression 使用Sigmoid，$\hat{y} \in (0,1)$ ，是具有物理意义的，即预测值为label=1 的概率

* 对于数据$\left\{X, y\right\}$ ，使用 likelihood 来进行优化

  $L(y|X) = \prod_ip(y_i | x_i) = \prod_{i}\sigma(x_i)^{y_i} (1-\sigma(x_i))^{1-y_i}$

* 优化的核心在于对-log-likelihood求导，并可以借由$\sigma(x)^{'}=\sigma(x)(1-\sigma(x))$ 简化：

  

  $L(y|X) => -log(L(y|X)) = - \sum_{i}(y_ilog(\sigma(y_i))+(1-y_i)log(1-\sigma(x_i)))$

  

  $\frac{\part L}{\part w} = -\sum_i(y_i\frac{1}{\sigma(x_i)}\sigma(x_i)(1-\sigma(x_i))x_i-(1-y_i)\frac{1}{1-\sigma(x_i)}(\sigma(x_i)(1-\sigma(x_i))x_i \\=-\sum_i y_i-\sigma(x_i)$

  





