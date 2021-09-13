

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



当然，在写成这个形式之后，用一般来说用来求解的方法使用Qudratic Programming。这种加上kernel的表达式在各种convex optimization 库里面也都是可以搞定的。最近看了下python的cvxopt库。其实能不能这么做有一个比较好的判断，就是说能不能接受kernel的计算。



还是这么处理来的清楚，SMO绕死了。



然后直接 $w = \sum\limits_{i \in S} \alpha_iy_ix_i$







#### 类SGD 解法

不管是线性代数法还是SMO，都是不能处理超大规模数据的，就算SMO，他也是$O(n^2)$ 这很明显在数据量大的时候没法用。所以本质上我觉得SMO和CD很重复，但又不够精简，因为少量数据可能用不到他，大量数据还是得靠SGD。。。在coordinate descent面前SMO的两个参数就显得很愚蠢。



当然，因为SVM的Loss function是代有限制条件的，所以首先需要对loss function进行变形。

$w = \max\limits_{w} \displaystyle\frac{2}{||w||} \ s.t. \ y_i(w^Tx)>=1$ 

在这个情况下，可以直接用hinge loss来重写一遍，所以本质上largrange，kkt都是为了简化数据使用的方法，在数据量很大的时候，SVM还是一个带有正则项的hinge loss。（这么理解还是比较舒服的）

$Loss = \displaystyle\frac{1}{n} \sum (max(0, 1-y_iw^Tx_i))+ \lambda||w||^2$

因为这个loss并不是光滑的，所以在使用SGD的同时，需要定义subgradient，也就是hinge loss中那个转折点的导数。

使用SGD来做可以处理原始的SVM，但有一个缺点，就是kernel的技巧处理不了。kernel是让SVM拥有非线性的一个很好用的技巧，但计算量很大，普通的SGD也没法变形。



当然除了这个方法，也有方法能够处理kernel，也就是之前一直提到的Cordinate Descent。Cordinate Descent能做到这一点，主要是他可以以变换以后的dual问题来做Iteration Descent。

kernel是在变换完以后出现的，

$L = \max\limits_{\alpha} \sum \alpha - \displaystyle\frac{1}{2} \sum\limits_i \sum\limits_{j} (a_i a_j y_i y_j x_i^T x_j)$

$0 \leqslant \alpha_i \leqslant \frac{C}{n}$

Coordinate Descent是每一次只更新一个参数，在这里即$\alpha_i$  





### SVR

之前都是SVC，也就是SVM的分类形式，其实SVR和SVC的区别只有一个，就是SVR是在最小化点到线的距离，然后他可以把loss function改编成正则项和到两遍的线的距离的和。

$L(w) = \lambda||w||^2 + C\sum (\xi_i^+ + \xi_i^-)$

然后都一样的，不想写了。。。





### SVDD(Support Vector Domain Description)

SVDD是一种用SVM的原理来做one class classification的。所谓的one class classification，是指你只有一部分train data数据，但是你想判断test data是否算是一个outlier。最主要的一个应用场景，就是异常值分析。普通的SVM是用来分类的，但one class classification的情况是你可能甚至没有label。因为异常值一般来说频率会很少，因此其实不需要label也能利用soft margin的思路达到outlier的效果。



SVDD 的思路其实很简单，就是你有输入数据，然后你需要找到一个体积最小的高维球体，来判断test data是否在这个球体中。当然我们需要使用soft margin的思路，因为很多情况下是训练数据中只有一小部分异常值，其他都是正常值，soft margin能在一定程度上直接对这些异常值有一个很大的cost。那么首先写出loss function。



$L(R,a,\xi) = R^2 + C\sum \xi_i \ \ \ subject \ to \ (x_i-a)^T(x_i-a) \leqslant R^2 + \xi_i \ , \ \ \xi_i \geqslant 0$

这里R就是半径，a是球体中心点，$\xi$ 是soft margin

然后和SVM一样，通过Largrange multiplier 进行变化，然后疯狂求导。。。带入。。。。

$L(R, a, \xi, \alpha, \beta) = R^2 + C\sum \xi_i - \sum \beta_i \xi_i + \sum (\alpha_i (x_i^Tx_i - 2x_i^Ta + a^Ta)-R^2-\xi_i)$

$\displaystyle\frac{\partial L}{\partial R} = 2R-2R\sum\alpha_i = 0 \ \ \ => \sum \alpha_i = 1$

$\displaystyle\frac{\partial L}{\partial a} = \sum \alpha_i(2a-2x_i)=2a-\sum 2\alpha_i x_i = 0 \ \ \ => a = \sum \alpha_i x_i$

$\displaystyle\frac{\partial L}{\partial \xi_i} = C-\beta_i-\alpha_i = 0$



$L(\alpha) = \sum \alpha_i <x_i,x_i> - \sum\limits_{i} \sum\limits_{j} \alpha_i \alpha_j<x_i, x_j>  \ \ \ \ subject \ to \\ \sum \alpha_i = 1 \\ 0 \leqslant \alpha_i \leqslant C$

 

因为sklearn中的oneclassSVM 不好用，它并不能用来做test，sklearn底层是用libsvm，他的做法就是转化啊成dual，然后使用SMO。

自己使用convex optimization来写的话，这里可以展开写一写。



这里在写的时候意识到的主要的一些问题，就是$\alpha _i$ 只有在对应的数据是support vector的时候才会大于零，而需要用到所有的support vector来计算中心点和半径。这在我看来有点影响精度，因为极少数的异常值有可能会离标准数据很远，所以这也就要求了训练数据基本需要有一定程度的挑选，又或者说有个大致的标签。这样训练的时候选取一小部分数据作为support vector，才不会算出很离谱的半径。



我们在使用qudratic programming的时候有这么一个标准形式：

$min \ \displaystyle\frac{1}{2}x^TQx + P^Tx \\ subject\ to \ \ Ax=b \\ Gx \leqslant h$

在这里把转化成dual之后的L加个负号变成找最小值，这样可以对应每一个变量：

$Q = 2K =K+K^T$

$P=-diag(K)$

$A = \boldsymbol{1}^T$

$b = 1$

$G = \left[ \begin{matrix} -I \\ I \end{matrix} \right]$

$h = \left[ \begin{matrix} 0 \\ C \end{matrix} \right]$

用一些qudratic programming 就可以解出所有的$\alpha _i$ ，然后也可以得到只有一少部分才会是非零值。

在利用上面得到的特性，$a=\sum \alpha _i x_i$ 计算得到中心点坐标。

然后对于任何点来说 $(z-a)^T(z-a) \leqslant R^2$ 都是满足正常值的范围的，之后可以代入距离公式或者自定义的距离kernel 就能分类是否异常值。





### Deep SVDD

其实很容易就想到，把传统SVDD加点deep NN。只不过要完成deep，需要重新定义几个东西，第一个就是Loss Function

从SVDD过来的话，Loss function还是很好想的，就是和半径有关，和各个点距离球体的距离有关，还有正则项，在这篇paper里面，soft margin的loss function就是

$L = R^2 + \displaystyle \frac{1}{nC} \sum max(0, ||\phi(x)-c||^2-R^2) + \displaystyle \frac{\lambda}{2} \sum ||W||_F^2$

论文里面还有一项简化，就是在one class的时候假定异常值数量是比较少的，那么可以把优化半径最小的项去掉。

$L = \displaystyle \frac{1}{n} \sum ||\phi (x_i, W)-c||^2 + \displaystyle \frac{\lambda}{2} \sum ||W||_F^2$ 

这个简化了的loss function直接解读的话就是正则项加上到中心点距离和。所以说本质上就是把kernel，dual这个过程直接用网络替代掉了。这也是合理的，kernel存储读取是比较耗时间的，用一个小一点的网络直接计算有一定的优势。



这里还有几个点

1. 当W全为0的时候，那么中心点直接就出来了。。。。所有的结果都是一样的，因此为了避免W被优化到全是0的情况，需要避免中心点初始化到这个点附近。
2. 不能存在bias，如果存在偏置项b，当某一层参数W为0时，那么所有x的输出均为一个与b相关的常数(假设为B)。那么后续的层所接受到的特征均为常数，那么网络后续的层的更新策略只需要将B映射到C(球心)即可。总之也是避免W全部优化为0的措施。
3. 优先选择ReLU，一个单调activate function如果有界，那么如果一个特征全为正/负，到最后可能就一样了。



这里中心点就成了简单取平均值，在one class中甚至不需要关心R，基本流程就是先pretrain一个autoencoder，中间的hidden state就是中心点向量，然后再固定中心点以后，继续训练encoder网络。。。本身是一个很简单的结构。



对于需要优化半径的情况，首先把半径初始化为0，在优化的时候，会设定一个outlier的比例，每一个迭代计算把距离排序之后取(1-nu)th quantile，也就是说找到这个百分位上的距离，设定为新的R。













## Naive Bayes

naive bayes本质上来说应该就是个计数，加上个贝叶斯。。。

当然，它是一个条件概率模型，那些出来可定就是这个形式，然后再假设feature独立，然后把它变成连乘的形式再取log变成连加。

$\displaystyle p(Class \ | \ Features) = \frac{p(Features \ | \ Class)*p(Class)}{p(Features)}$

这个式子中经典的只要关注分子，因为分母在训练数据定了以后是个定值。

然后因为独立性就变成了 $p(Class)*\prod p(F_i \ \ |\ C) $

后面只要求最大的class

$C = \max\limits_{c} p(C)*\prod p(F_i|C)$

$p(C)$ 直接统计算出来，$p(F_i|C)$ 也直接全部统计一遍。。。然后就结束了。



最简单的NB很容易，纯粹的统计一遍就结束了。









## EM（Expectation-Maximum）

EM都知道有两部分构成，期望步和极大步，对于什么都忘光了的我来说EM的核心就是活用Jensen不等式。。。

### Jensen Inequality

首先对于函数f为凸函数，即二阶导>0，那么$E(f(x)) \geqslant f(E(x))$，并且仅当随机变量X是常量的时候等号成立。



EM算法的思想就是会认为给定了一个training dataset，$x_i\in X$之后，这些数据之间有一定的联系，这些联系由latent variable，$z$来确定。我们假设训练数据是符合某个分布的，即有一些未知的模型参数$\theta$ 决定了概率密度函数，然后每一个数据都可以看作是在这个分布上的一次采样，那么可以写出他的采样概率 $p(x_i|\theta)$

在推导EM的时候，和其他概率模型一样，要用到likelihood

$\hat{\theta}=argmax \ \prod p(x_i|\theta)$

$L(\theta) = -\sum log(p(x_i|\theta))$



那么问题来了，怎么把latent variable $z_i$ 用上去呢。所谓的隐藏变量，可以认为就是一部分没展示在train data里面的feature，所以我们的概率分布因该是变成在$\theta$ 下 $x_i, \ z_i$的联合分布，即 $p(x_i,z_i |theta)$。当然可以结合GMM来理解$z_i$，这里的$z_i$可以理解为有很多个分布构成了总体的分布，当前的数据是输入其中一个分布的，或者说当前的数据是其中几个分布的加权和。

$L(\theta) = -\sum log\sum\limits_{z_i}(p(x_i,z_i|\theta))$ 

然后为了想办法再变形，直接引入一个$Q_i(z_i)$，可以认为是隐藏变量的一些分布之类的。

$L(\theta) = -\sum log\sum\limits_{z_i}(p(x_i,z_i|\theta))= -\sum log\sum\limits_{z_i} Q(z_i)\displaystyle\frac{(p(x_i,z_i|\theta))}{Q(z_i)}$

那么这里$log(x)$是一个凹函数，直接取负就是凸函数，以及里面的东西可以看作是一个期望，就可以用上Jensen Inequality

$L(\theta)= -\sum log\sum\limits_{z_i} Q(z_i)\displaystyle\frac{(p(x_i,z_i|\theta))}{Q(z_i)} \geqslant -\sum \sum\limits_{z_i} Q(z_i) log(\displaystyle\frac{p(x_i,x_i|\theta)}{Q(z_i)})$

并且由Jensen Inequality的等号成立条件，我们可以知道当且仅当$\displaystyle\frac{p(x_i,z_i|\theta)}{Q(z_i)}=c$ 是常数的时候，等号成立。

并且，因为满足$Q(z_i)$ 是一个分布，所以和为1，$\sum\limits_{z} p(x_i, z_i|\theta) = c$

$\displaystyle\frac{p(x_i,z_i|\theta)}{Q(z_i)} = \sum\limits_{z} p(x_i, z_i|\theta) $

$\displaystyle\frac{p(x_i,z_i|\theta)}{\sum\limits_{z} p(x_i, z_i|\theta)} =  Q(z_i) = \displaystyle\frac{p(x_i,z_i,\theta)}{p(x_i|\theta)} = p(z_i | x_i,\theta)$ 

变换到这个形式，就能直接看出来$Q(z_i)$ 就是后验概率了。。。另外这里我们就是找到了不等式的等号条件，也就是M步，而如何选择Q就是E步。



### GMM 的 EM推导

在GMM这里，$\theta$ 就是 $\mu,\ \Sigma$

$p(x|\mu,\Sigma)=\displaystyle\frac{1}{(2\pi)^\frac{d}{2} |\Sigma|^\frac{1}{2} } exp (\displaystyle -\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu))$

$L(\mu, \Sigma) = -\log(\prod p(x_i | \mu ,\Sigma)) \\ =-\sum -\displaystyle\frac{d}{2}(2\pi)-\displaystyle\frac{1}{2}|\Sigma|-\displaystyle\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)$



在GMM里面隐藏的变量就是对于每个数据是否属于各个高斯分布，就是0或者1的一个类似于one-hot的向量。其实现在上面的Q就是当前数据处于各个高斯分布的概率分布。



E-step

在E-step中，训练数据和$\theta$ 是已经知道的，需要更新的是在当前条件下每个点属于各个分布的概率值。

也就是说在给定了当前的训练数据和参数，先计算对于每个数据来说是否来自于每一个高斯分布，这里的结果就是他来自于各个高斯分布的概率。按照正常的写法，$E(f(x))$ 现在可以写成

$Q(\theta|\theta_i) = E[log(p(X,Z|\theta))|X,\theta_i]$

需要更新的是：

$\pi_{ij} = p(z_{ij}=1|x_i,\theta_t)=\displaystyle \frac{p(x_i|z_{ij}=1,\theta_t)*p(z_{ij}|\theta_t)}{p(x_i|\theta_t)} \\ =\displaystyle \frac{p(x_i|x_{ij}=1,\theta_t)*\pi_{ij}^{t}}{p(x_i|\theta_t)} = \displaystyle \frac{p(x_i|z_{ij}=1,\theta_t)*\pi_{ij}^{t}}{\sum\limits_{j} \pi_{ij}^t p(x_i|z_{ij}=1,\theta_t)}$

$p(x_i|z_{ij}=1,\theta_t)$ 用高斯分布的共识直接就能求，这样就更新了新的期望值。



M-step

$\theta^{t+1} = \max \ \ Q(\theta | \theta^t)$

$p(X,Z|\theta) = \prod \prod p(x_i | z=z_j, \theta) p(z=z_j|\theta) = \prod\prod (\pi_j \phi(x_i|...))^{z_{ij}}$

$Q(\theta|\theta_i) = \sum\sum \pi_{ij} log(\pi_j \phi(x | z, \theta))$



这里有个比较骚的操作，就是在写联合分布的时候，还是likelihood的指数的操作，然后现在有了一个限制条件，就是$\sum\pi_i=1$，就可以用拉格朗日了。。。

$L = -Q(\theta |\theta^t) + \lambda(\sum \pi_i-1)$

正常来说还有一项约束条件就是$\pi>=0$，所以本来应该是要在加一项的，但由于在这里$\pi$ 不可能是一个为0的数，所以所有的系数必定都要=0，所以加了和没加没区别。

然后就要求导了：

$-\sum \pi_{ij} \frac{1}{\pi_j} + \lambda = 0 => \pi_j = \frac{1}{\lambda}\sum_i\pi_{ij}$

然后需要满足限制条件，就能得出$\lambda=\sum\limits_{j}\sum\limits_{i}\pi_{ij}$ 

对GMM来说 $\pi_j$ 就是每个维度的高斯分布的分量，那么总体的分布也就是各个分布的加权和。

然后有了每一个点的分布概率，重新计算$\mu_k, \Sigma_k$



具体来算一下吧

$\mu_{t+1} = max \ \ \sum\sum\pi_{ij}log(x_i|\mu_t, \Sigma_t) \\ =\sum\sum \log(\prod p(x_i | \mu ,\Sigma)) \\ L = \sum\sum \pi_{ij} (log(-\displaystyle\frac{d}{2}(2\pi)-\displaystyle\frac{1}{2}|\Sigma|-\displaystyle\frac{1}{2}(x_i-\mu)^T\Sigma^{-1}(x_i-\mu)))$



$\displaystyle \frac{\partial L}{\partial \mu} = \sum\ - \pi_{ij} \Sigma_j^{-1}(x_i-\mu)=0 \\ \mu_{t+1}= \displaystyle \frac{\sum \pi_{ij} x_i}{\sum \pi_{ij}}$



$\displaystyle \frac{\partial L}{\partial \Sigma} = -\sum \pi_{ij} (\frac{1}{2}\Sigma^{-1}-\frac{1}{2}(x_i-\mu)^T (x_i-\mu)\Sigma^{-2})=0$

$\Sigma_{t+1} = \displaystyle \frac{\sum \pi_{ij} (x_i-\mu)^T(x_i-\mu)}{\sum \pi_{ij}}$





## LDA（Linear Discriminant Analysis 和 Latent Dirichlet Allocation）

这两个LDA不是一个东西。。。

这里顺便都整理一遍算了



### Linear Discriminant Analysis

在这个情况下，一般会一起考虑QDA（Quadratic Discriminant Analysis），这里的LDA和QDA一般适用于分类的情况。

LDA和QDA都是建立在概率模型的基础上，并且基本上是用的multivariant  Gaussian来计算距离。

$p(y=k|x) = \displaystyle \frac{p(x|y=k)*p(y=k)}{p(x)} = \displaystyle \frac{p(x|y=k)*p(y=k)}{\sum p(x|y=y_i)*p(y=y_i)}$

然后一个假设就是每个类别下的数据都是一个多维高斯分布。

$p(x|y=k) = \displaystyle \frac{1}{(2\pi)^{\frac{d}{2}}|\Sigma_k|^{\frac{1}{2}}} exp(-\displaystyle \frac{1}{2} (x-\mu_k)^T \Sigma^{-1} (x-\mu_k))$

$log(p(y=k|x))=log(x|y=k)+log(p(y=k))-log(p(x))$

对于training data确定了以后，$log(p(x))$ 是一个确定值，可以事先计算出来，对于每一类都一样，因此没有比较的意义。

$log(p(y=k|x))=log(x|y=k)+log(p(y=k))+C \\ = \displaystyle -\frac{1}{2}(x-\mu_k)^T\Sigma_k^{-1}(x-\mu_k)-\displaystyle \frac{d}{2}log(2\pi) - \displaystyle \frac{1}{2} log(|\Sigma_k|) + log(p(y=k))+C$

对于每一类来说其他都可以直接计算出来，所以本质上也就是在比较到各个高斯分布的distance。

对于两类来说，直接比较两个的大小就行了

$log(p(y=1|x))-log(p(y=0|x)) = \displaystyle (x-\mu_0)^T\Sigma_k^{-1}(x-\mu_0) \displaystyle -(x-\mu_1)^T\Sigma_k^{-1}(x-\mu_1)+log(p(y=1))-log(p(y=0)) + log(|\Sigma_0|) - log(|\Sigma_1|)$

先把其他直接能算出来的归到一起变成一个常数，然后直接化简。

$log(p(y=1|x))-log(p(y=0|x)) = \displaystyle (x-\mu_0)^T\Sigma_0^{-1}(x-\mu_0) \displaystyle -(x-\mu_1)^T\Sigma_1^{-1}(x-\mu_1) - T$

$=> x^T(\Sigma_0^{-1}-\Sigma_1^{-1})x - 2(\mu_0^T\Sigma_0^{-1}-\mu_1^T\Sigma_1^{-1})x+(\mu_0^T\Sigma_0^{-1}\mu_0-\mu_1\Sigma_1^{-1\mu}\mu_1)-T$

$=> x^T(\Sigma_0^{-1}-\Sigma_1^{-1})x - 2(\mu_0^T\Sigma_0^{-1}-\mu_1^T\Sigma_1^{-1})x + C$

只要判断是不是大于零，就知道属于哪一类



到这里为止其实都是QDA，毕竟形式也都是二次项，而LDA相较于QDA变动就是简化了方差，LDA假设两类方差是一样的$\Sigma_0=\Sigma_1$

$2(\mu_0^T\Sigma^{-1}-\mu_1^T\Sigma^{-1})x > C$







### Latent Dirichlet Allocation

这个LDA粗略看了一遍就知道很麻烦了

LDA数学上的准备要求很多，没办法硬着来吧



#### Gamma 函数

其实gamma函数接触过几次了，但都是随便让它过去了。。。现在还是得补。。。不得不说每次都觉得自己要滞后别人一两年真的很难受。

$\Gamma(x) = \displaystyle \int_0^{+\infty} e^{-t}t^{x-1} dt \ \ (x>0)$

Gamma 函数的意义是计算阶乘的一般函数形式。Gamma 函数有一个很重要的性质：

$\Gamma(x+1) = \displaystyle \int_0^{+\infty} e^{-t} t^x dt = -e^{-t}t^x\displaystyle|_0^{+\infty} + x\displaystyle \int_0^{+\infty} e^{-t}t^{x-1} dt = x\Gamma(x)$



#### 二项分布

随机变量$X - Burnulli(n, p)$

$p(K=k) = \left (\begin{matrix}n \\ k  \end{matrix}\right) p^{k} (1-p)^{1-p}$



#### 多项式分布

多项式的话，就对应有多个类，然后每个类都有自己的选中几率，从里面选到第$i$类$x_i$ 个

$p(x_1,x_2,...x_k,n,p_1,p_2...p_k) = \displaystyle \frac{n!}{x_1! x_2!...x_k!}p_1^{x_1}....p_k^{x_k}$

当然这里一大堆阶乘，可以用Gamma函数来表示



#### Beta 分布

beta分布是定义在（0，1）中的一个连续的分布，先给出pdf

$f(x| \alpha,\beta) = \displaystyle \frac{x^{\alpha-1}(1-x)^{\beta-1}}{\displaystyle \int_0^1u^{\alpha-1}(1-u)^{\beta-1}du}$

这里其实最关键的点就是化简分母那一部分，因为分布函数是0-1的所以相当于0-+inf，>=1 的部分都为0。

$\displaystyle \int_0^1u^{\alpha-1}(1-u)^{\beta-1} du = u^{\alpha-1} \displaystyle \frac{(1-u)^\beta}{\beta} \bigg|_0^1+ \displaystyle \int_0^1 \displaystyle \frac{\alpha-1}{\beta} u^{\alpha-2}(1-u)^{\beta} du = \displaystyle \frac{\alpha-1}{\beta} \displaystyle \int_0^1 u^{\alpha-2}(1-u)^{\beta} du$

然后发现这又是一个相同形式的东西，所以可以疯狂写下去。。。先用$f(\alpha, \beta)$ 来表示。

$f(\alpha,\beta) =  \displaystyle \frac{\alpha-1}{\beta} f(\alpha-1, \beta+1) \\ =\displaystyle \frac{(\alpha-1)(\alpha-2)...1}{\beta(\beta+1)...(\beta+\alpha-2)} f(0, \beta+\alpha)$

$f(0, \alpha+\beta) = \displaystyle \int_0^1 (1-u)^{\alpha+\beta-1} du = -\displaystyle \frac{(1-u)^{\alpha+\beta-1}}{\alpha+\beta-1} \bigg|_0^1 = \displaystyle \frac{1}{\alpha+\beta-1}$

$=>f(\alpha,\beta) = \displaystyle \frac{1...(\alpha-1)}{\beta...(\beta+\alpha-1)} = \displaystyle \frac{1...(\alpha-1) 1...(\beta-1)}{1...(\alpha+\beta-1)} = \displaystyle \frac{\Gamma(\alpha)\Gamma(\beta)}{\Gamma(\alpha+\beta)}$



$f(x| \alpha, \beta) = \displaystyle \frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)+\Gamma(\beta)}x^{\alpha-1}(1-x)^{\beta-1}$



这里还是要定义一下一个函数$B(\alpha,\beta) =\displaystyle \frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)+\Gamma(\beta)}$



到这里用阶乘的定义联系上了Gamma函数，然后把这样一个分布叫做beta分布。。。Beta分布中最重要的就是$\alpha, \beta$这两个参数，算一下期望就会发现$E(x,\alpha,\beta) = \displaystyle\frac{\alpha}{\alpha+\beta}$。





#### Dirichlet 分布

Dirichlet分布可以看作是多项式分布和beta分布的一个结合，pdf为

$f(x_1,x_2,..x_k|\alpha_1,\alpha_2,...\alpha_k) = \displaystyle \frac{1}{B(\alpha)}  \prod x_i^{\alpha_i-1}$

$\sum x_i = 1$

$B(\alpha) = \displaystyle \frac{\prod\Gamma(\alpha_i)}{\Gamma (\sum\alpha)_i}$

同样这种形式可以直接参考多项式的期望，写出$E(x) = [\displaystyle\frac{\alpha_1}{\sum \alpha_i}, ....]$





#### Gibbs Sampling

一种MCMC采样方法，适用于条件概率比边缘概率更好计算的场景。





#### LDA..(part 1)

前面那些东西单拿出来证明一下，倒也都还好说。。

LDA是对单个单词来说的，认为词之间，文档之间都是相互独立的，并且不考虑word之间的顺序关系。所以生成一个语料库的概率就是所有词概率相乘，也就是likilihood的这种写法。这种写法就是认为每一个词出现的频率就是一个定值，只不过我们可能需要统计什么的来计算

$p(w) = \prod p_{w_i}^{n_i}$

当然也有另外一种看法，就是认为词的概率满足一个概率分布，在LDA中用的就是dirichilet 分布，当然超参数就是每个词的$\alpha$，他计算的就是每个词的联合概率分布。这具体来说就像是从所有文档中重复抽取M个document，每个document选取N的词，这样生成了整个语料库。这样来看的话每个词的频率就会满足一定的概率分布，是一个随机变量，而不是一个固定值。

$Dir(p|\alpha) = \displaystyle \frac{1}{\Delta(\alpha)} \prod p_k^{\alpha_k-1}$

因为需要满足积分为1，$\Delta(\alpha_i) = \displaystyle \int\prod p_k^{\alpha_k-1} dp$

这里因为Dirichlet 和多项式分布形式上是一样的，所以可以把两者结合，结合之后新的概率分布就又有词频率项又有超参的调节项。同样也可以认为$\alpha$ 是选到一个词袋的概率，$n$ 就是词的频率。

$Dir(p|\alpha, \hat{n}) = \displaystyle \frac{1}{\Delta(\alpha+\hat{n})} \prod p_k^{\hat{n}+\alpha_k-1}$

$E(p)=[\displaystyle \frac{\alpha_1+n_1}{\sum \alpha_i+n_i}, ...]$

那么对于整个语料库来说，给定$\alpha$ 生成一个语料库W的概率为：

$p(W|\alpha) = \displaystyle \int p(W|p)*p(\bar{p}|\alpha) d\bar{p} = \displaystyle \int \prod p_i^{n_i} \displaystyle \frac{1}{\Delta(\alpha)}p_i^{\alpha_i-1} dp \\ = \displaystyle \frac{1}{\Delta(\alpha)} \displaystyle \int \prod p_i^{n_i+\alpha_i-1}dp = \displaystyle \frac{\Delta(n+\alpha)}{\Delta(\alpha)}$





####pLSA (Probabilistic Latent Semantic Analysis)

LSA基本使用了SVD的思路来踢去核心的概念，从SVD的角度来说同义词这种多词对一概念是可以被处理的，因为SVD是降维提取核心思想的方法，但一词多义就很难体现。所以在pLSA中引入了一个topic的latent variable。

pLSA中认为doc-topic满足一种分布，也就是说每个document对应着各种topic的概率，另外topic-word也满足一种分布，在每个topic中词向量都有对应的概率。所以pLSA认为一篇document的构建过程是先随机选择出一个topic，然后从这个topic中选取一个单词。对于pLSA中依然是用的是词袋模型，doc之间，word之间是相互独立的。

assume给定一个document，总共有K个topic，然后要生成一个n个词的document

$p(\hat{w}|d_m) = \prod_{i=1}^n \sum_{j=1}^k p(w_i|t_j)*p(t_j|d_m)$





#### LDA (part 2)

那么现在来看LDA和pLSA，两者其实还是很相似的，首先都是以某个先验概率选择一篇document，并且都设有latent variable，只不过在pLSA中是一个topic，并且是直接由doc-topic，topic-word来生成的。而在LDA中可以看作topic是和dirichlet distribution相关的东西。LDA是把这两个分布都用Dirichlet 和多项式分布来处理。

<img src="/Users/jiajunh/Library/Application Support/typora-user-images/image-20210913164057602.png" alt="image-20210913164057602" style="zoom:20%;" />



从这里就可以看出doc-topic，topic-word用了两个Dirichlet，多项式分布来处理。

$\alpha$  这边表示的是在生成第m篇document的时候，从一个Dirichlet+多项式分布中生成了这个词对应的topic的概率分布

$\beta$  这边表示的是在生成topic-word的时候，从一个Dirichlet+多项式分布中生成了这个词对应的概率分布。

因为每个词都是独立的，所以对于生成一篇document来说，一共需要采样2N次，每个词需要采样一个topic，一个word，然后他们可以交换顺序，可以先采样完所有的topic，再采样所有的word。

$p(topic|d_m) = \displaystyle \int \prod p(topic|\bar{p_1})*p(\bar{p_1}|\alpha) d\bar p = \displaystyle \frac{\Delta(\alpha+k)}{\Delta(\alpha)}$

$p(word|topic) = \displaystyle \int \prod p(word|\bar{p_2})*p(\bar{p_2}|\beta) d\bar p = \displaystyle \frac{\Delta(\beta+n)}{\Delta(\beta)}$

$p(\bar{w},\bar{t} | \alpha,\beta)=\displaystyle\prod \displaystyle\frac{\Delta(\alpha+k)}{\Delta(\alpha)} \prod \displaystyle \frac{\Delta(\beta+n)}{\Delta(\beta)}$  

这里的$\alpha, \beta, k, n$全是向量，是超参向量，和给定的document的中集合中topic / word的分布情况。





现在我们来采样

首先在所有document中选定M个作为语料库，topic一共有K个，每篇document有N个word，还用了$i=(m,n)$ 来表示第m篇document中第n个词。$\theta_m$ 是一个document m 的topic分布，$\phi_k$ 是topic k 的word 分布。



