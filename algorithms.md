

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

  





## SVM

### 普通SVM

首先，SVM的本质上是一个hinge loss。即 $L = max(0, 1-y_i)$



假设一个线性变换能把数据进行分割，线性变换可以写成 $w^T x+b$ ，当 $w^T x+b=0$ 的时候就是直线本身，所以这条直线把平面分为了两部分，一部分$>0$ ，另一部分$<0$ 。在直线方程中 $w$ 是有意义的，就是直线的法向量。

在这个基础上，空间里面任意点 $x$ 到这条直线的距离可以写成：

$r = \displaystyle\frac{|w^Tx+b|}{||w||}$

普通来讲，只要把点分为 $w^T x+b>0$ 或者 $w^T x+b<0$ 就完成了分类。SVM 要求更高一点，首先它会分为hard margin和soft margin。简单来说hard margin对应的是所有点在margin上或者margin之外，也就是有一条线严格可以分割平面上的点。soft margin 不那么严格要求，点可以在margin内部，甚至另一边。

<img src="/Users/jiajunh/Library/Application Support/typora-user-images/image-20210827103134257.png" alt="image-20210827103134257" style="zoom:20%;" /><img src="/Users/jiajunh/Library/Application Support/typora-user-images/image-20210827103157939.png" alt="image-20210827103157939" style="zoom:20%;" />

对于SVM来说，所谓的margin 也就是两条与原来直线平行的直线 $w^Tx+b=\pm1$ ，这两条之间之间的距离为$\displaystyle\frac{2}{||w||}$，然后为什么叫support vector呢，对于hard margin来说在这两条线上的点就是support vector，对于soft margin 来说，在这条边上/另一侧的都是support vector。



####Hard Margin

那么接下来，只要把这个margin最大化也就找到了最优的分割线。当然，可以先把 $b$ 写到 $w$ 里面去，简化一点。

$w = \max\limits_{w} \displaystyle\frac{2}{||w||} \ s.t. \ y_i(w^Tx)>=1$ 

 这里有一个写法，就是对于可分的两类来说，先定义  $w^Tx >=1 $ 的部分 $y_i=1$ ，这样可以简化成上面的形式，就很牛，然后就联系到了hinge loss，只需要稍微变化一下，能得到

$w = \min\limits_{w} ||w|| \ s.t. \ 1-y_i(w^Tx)<=0 \\ = \min\limits_{w} \displaystyle\frac{||w||^2}{2} \ s.t. \ 1-y_i(w^Tx)<=0$

这样后面的部分就是一个hinge loss，然后整体的意义也清晰了，就是在满足hinge loss的情况下，最小化 $w$ 的膜。



然后就是SVM最关键的变换了，涉及到两个关键的优化问题，拉格朗日乘子，和KKT

简单来说就是 我们现在要优化 $\min\limits_{w}\displaystyle\frac{||w||^2}{2}$ ，这是一个需要优化的凸函数，以及我们有一系列的限制条件，也就是每个数据点的hinge loss。KKT的约束条件是$g(x) <= 0$ ，然后对于这个优化问题可以把限制加入需要优化的函数，

$L(w) = \displaystyle\frac{||w||^2}{2} + \lambda g(x)$

$\left\{ \begin{aligned} g(x)\leqslant0 \\\lambda\geqslant0 \\\lambda_i g(x_i)=0 \end{aligned} \right.$ 



这个变换也就是减少计算量的精髓，从上面可以得出，只有hinge loss 不等于0的时候，$\lambda$ 才会取 $\geqslant 0$ 的值，所以在优化的时候只有在margin上的点才会进入优化。

$L(w,\alpha) = \displaystyle\frac{||w||^2}{2} + \sum \alpha_i(1-y_i(w^Tx_i)) , \ \ \alpha_i\geqslant0$

还有一个关键的点，就是在这种约束下 $\max\limits_{\alpha} L(x, \alpha) = f(x)$ ，$f(x)$ 就是原始的优化函数。原来的优化问题就变成了看上去比较让人难受的

$\min\limits_{x}\ \max\limits_{\alpha} L(x, \alpha)$ 

这里有极小极大值这么一个东西，然后拉格朗日有一个名词叫primal/dual，对偶问题，把这个极小极大值变成极大极小值就叫做对偶问题。拉格朗日对偶性就是说的是这个$min \ max$ 是否等于$max \ min$。若是相等的话，就说是有强对偶性。

在SVM这里，KKT条件就是说明了这个问题的强对偶性，然后我们就可以直接免换成对偶问题。这里的条件指的是，优化函数是凸函数，约束函数是凸函数，并且不等式严格可行，SVM都是满足的。



这么一来，整个优化问题变成了可以先求最小值的对偶问题，对于可以先求最小值的情况，直接求导找=0的情况

$\max\limits_{\alpha} \ \min\limits_{x} \displaystyle\frac{||w||^2}{2} + \sum \alpha_i(1-y_i(w^Tx_i))$

$\displaystyle\frac{\partial L}{\partial w} = w - \sum \alpha_iy_ix_i = 0 \ => \ w= \sum \alpha_iy_ix_i$

$\max\limits_{\alpha} L(w,\alpha)=\max\limits_{\alpha} \displaystyle\frac{1}{2} \sum\limits_i \sum\limits_{j} (a_i a_j y_i y_j x_i^T x_j) + \sum\alpha_i - \sum\limits_{i} \sum\limits_{j}\alpha_i \alpha_j y_i y_j x_i^T x_j \\ = \max\limits_{\alpha} \sum \alpha - \displaystyle\frac{1}{2} \sum\limits_i \sum\limits_{j} (a_i a_j y_i y_j x_i^T x_j)$

当然这边可以看到出现了$x_i^T x_j$ 本质上也就是一个kernel，所以SVM在这里是可以使用各种kernel的。



有了这个形式，之后再使用SMO（Sequential minimal optimization）算法，SMO本质上是一个类似于GD的迭代算法，其实就和Coordinate Gradient Descent差不多。(个人觉得SMO 纯粹是数据很多的时候用的，不然的话有点烦，普通来说这种形式直接可以使用矩阵的运算。。。这里先给出SMO，下面soft margin里面按照EECS 545 里面的做法来做)



SMO每一次迭代只更新两个值$\alpha_1, \alpha_2$ ，其他都先看成常数，和CGD一样。

$W(\alpha_a, \alpha_2) = \max\limits_{\alpha} L(w,\alpha) = \alpha_1 + \alpha_2 - \displaystyle\frac{1}{2} (K_{11} \alpha_1^2+K_{22} \alpha_2^2)-y_1 y_2 K_{12}\alpha_1 \alpha_2 + C$

SMO把$w$ 拆开来，拆出来$w, b$….

能再出来一个限制条件：$\sum y_i\alpha_i = 0$，所以只有两个变量的情况下 $\alpha_1 y_1 + \alpha_2 y_2 = C$ ，然后消元 $\alpha_1 = \gamma-s\alpha_2, s=y_1y_2$

$W(\alpha_2) = \gamma + (1-s)\alpha_2 - \displaystyle\frac{1}{2} (K_{11}(\gamma-s\alpha_2)^2+K_{22}\alpha_2^2)-sK_{12}\alpha_2(\gamma-s\alpha_2)-y_2v_2\alpha_2-y_1v_1(\gamma-s\alpha_2)+C$

$\displaystyle\frac{\partial W(\alpha_2)}{\partial \alpha_2} = 1-s+K_{22}\alpha_2-K_{11}s\gamma-K_{11}\alpha_2+2K_{12}\alpha_2-y_2v_1-y_2v_2 = 0 \\ \alpha_2 = \displaystyle\frac{y_2(y_2-y_1+y_1\gamma(K_{11}-K_{12})+v_1-v_2)}{K_{11}+K_{22}-2K_{12}}$

这里可以看做为$\alpha_2 $ 做了迭代更新 $\alpha_2 = \alpha_2 + \displaystyle\frac{y_2 (E_1-E_2)}{K}, \\ \alpha_1 = \alpha_1+y_1y_2(\alpha_2^{old}-\alpha_2)$





#### Soft Margin

对于soft margin来说，唯一的区别在于限制项，从简单的$y_i(w^T x)\geqslant1$ 变成了 $y_i(w^T x) \geq 1-\xi_i\ ,\  \xi_i \geqslant0$ 

那么整个问题的约束变多了，拉格朗日乘子的项也变了，整个优化变成了：

$\min\limits_{w}L(w) = \min\limits_{w} \displaystyle\frac{||w||^2}{2} + C\sum\xi_i \ subject\ to\\ y_i(w^T x) \geq 1-\xi_i\ \\  \xi_i \geqslant0$

$\min\limits_{w} L(w,\alpha, \beta) = \min\limits_{w} \displaystyle\frac{||w||^2}{2} + C\sum\xi_i + \sum \alpha_i(1-y_i(w^T x_i)-\xi_i)-\sum \beta_i\xi_i \\ \alpha_i, \beta_i \geqslant0$

同样开始求导。。。

$\displaystyle\frac{\partial L}{\partial w} = w - \sum \alpha_iy_ix_i = 0 \ \ => w=\sum\alpha_iy_ix_i$

$\displaystyle\frac{\partial L}{\partial \xi_i} = C-\alpha_i -\beta_i=0 \ \ \ => \alpha_i+\beta_i=C$

$\alpha_i(1-y_i(w^T x_i)-\xi_i) = 0 \ \ \ => \alpha_i=0 \ \ or \ \ y_i(w^Tx_i)=1-\xi_i$

$\beta_i \xi_i=0 \ \ \ => \ \beta_i=0 \ \ or \ \ \xi_i=0$



全部带进去。。。。。

$L(\alpha, \beta) = \displaystyle\frac{||w||^2}{2} + \sum (C-\alpha_i-\beta_i)\xi_i + \sum\alpha_i - \sum\limits_{i} \sum\limits_{j}\alpha_i \alpha_j y_i y_j x_i^Tx_j \\ = \sum\alpha_i - \displaystyle\frac{1}{2}(\sum\limits_i \sum\limits_j \alpha_i \alpha_j y_i y_j x_i^T x_j)$

然后因为dual外面还有一层max，这里就先不用SMO了，用矩阵的计算来处理。

$\max\limits_{\alpha, \beta} L(\alpha, \beta) = \max\limits_{\alpha} \ \boldsymbol{\alpha^T1} - \displaystyle\frac{1}{2} \boldsymbol{\alpha^T K \alpha} = \min\limits_{\alpha} \ -\boldsymbol{\alpha^T1} + \displaystyle\frac{1}{2} \boldsymbol{\alpha^T K \alpha} \\ => \ \boldsymbol{K\alpha}_{LS} = \boldsymbol{1} \ \ \ \ \ subject \ to \ \ \ 0\leqslant\alpha_i\leqslant C$



这里当写成这个形式之后，首先$\boldsymbol{K}$ 是PSD，所以先不管限制条件解出$\alpha_{LS}$ ，然后对于整个$\alpha_{LS}$ 就有了一个坐标，然后把限制条件的区域和这个坐标进行比较，对每个维度比较要么取0，要么取C（即最大值）。



然后直接 $w = \sum\limits_{i \in S} \alpha_iy_ix_i$





