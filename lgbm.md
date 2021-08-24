[toc]

## 决策树（Decision Tree）

###ID3 (Iterative Dichotomiser 3)

​	信息熵：$H(S)=\Sigma_{x \in X } -p(x) * log_2(p(x))$​， 其中 $X$​ 是数据集 $S$​ 中的不同类。在分类树中，$X$​ 就是不同的类别

​	在某一个节点选择分割特征的时候需要看条件熵，即选取的特征能使信息熵降得最低。

​	$H(X|Y) = H(X, Y) - H(X)$

​	信息增益 = 信息熵 - 条件熵.  $InfoGain(S, c) = H(S) - \Sigma_c \frac{|S_c|}{|S|} H(S_c)$​​ 

​	** ID3 的缺点

	1. 信息增益会趋向于选择类别比较多的属性，特征类别过多不一定有利于分类
	2. ID3 只适用于离散数据
	3. 不能处理缺失值，不能剪枝



### C4.5

​	由于信息增益的局限性，ID4.5 主要改进的就是选取特征的判断依据。

​	信息增益率：$GainRatio(S, f) = \frac{InfoGain(S, f)}{SplitInfo(S, f)}$​​​​，其中f为特征，$SplitInfo(S, f)$ 代表在选取特征$f$ 之后数据分布情况。

​	$SplitInfo(S, f) = -\Sigma_i \frac{|S_i|}{|S|} log_2 \frac{|S_i|}{|S|}$​

​	在选取特征之后，按照特征会分为不同的类别，当特征类别特别多的时候，$SplitInfo$​​ 值也会变大，一定程度上限制特征的选择



​	C4.5依然不能处理连续值的情况。



###CART （Classification and Regression Tree）

​	虽然C4.5 改进了特征的选取方法，但基于熵的计算涉及很多对数计算，在CART中使用了新的Gini指数作为选取标准。

​	$Gini(S) = 1-\Sigma_i p_i^2 = 1-\Sigma_i \frac{|D_i|^2}{|D|^2}$​​​  如果数据能被特征$f$ 分为两部分 $Gini(S, f) = \frac{|D1|}{|D|}Gini(D1)+ \frac{|D2|}{|D|}Gini(D2)$​

​	在特征分布接近平均时，基尼系数会比较大，因此使用基尼系数，在减小计算的同时，也会优先选取分割均匀的特征。



​	CART 在引入基尼系数的同时，还将之前按照特征值来分割为不同个数的单元，变成只分为两部分的二叉树。

​	

​	CART 在使用回归树的时候，由于规定只分裂为二叉树，一次可以定义连续值的处理。

 1. 选取一个特征，并排序

 2. 任意两个值之间作为分割点，可以有n-1种分法

 3. 计算每种情况的loss，并选取最小值，$Loss = \sum_i (f(X_i)-g_i)^2)$，其中$g_i$ 为叶结点上的均值，所以实际就是选取分裂以后两边方差和的最小的分裂值

 4. 选取最大增益的特征

    ​	

    使用回归树的时候，最终的结果是对应的Leaf Node的所有结果的均值，因此可以定义Loss Function

    

    

###缺失值处理

计算信息增益时，

- 计算信息熵, 忽略缺失值
- 计算信息增益, 乘以未缺失实例的比例

分裂的时候，缺失数据会继续走所有节点下面的分支上，每个分支的缺失数据有一个weight，没有缺失值的数据weight=1

对于有缺失特征的数据点，定义了几个weights

$\rho=\frac{|无缺失值数据|}{|所有数据|}$​​​

$p_k=\frac{｜无缺失数据中class(k)｜}{|所有无缺失数据|}$

$r_v=\frac{｜无缺失数据在特征a上取值a_v的数量｜}{｜无缺失数据在特征a上的数量｜}$​

$weight = \rho r_v$

N：叶结点样本权值总和，E：该叶结点与该数据label不同的权值总和

使用 $N/E$​ ​来表示，

* 其实就是所有叶结点上各个分类的概率取最大值



###剪枝

剪枝是树模型中有效的防止过拟合的方法，最简单的做法就是直接限制树的深度和叶结点的数量。另外，CART模型其实对于偏离散的数据更容易过拟合。因为在切分离散特征的时候，相当于一次性添加了很多非线性的效果。

后剪枝：

​	后剪枝的目的是在测试集上计算loss function，通过剪枝使loss function降低

​	主要是从叶结点开始从下往上，计算如果将跟结点下所有数据合并在一起loss function是否会降低。

​	后剪枝的Loss function添加了regularization项：$\alpha |T|$， 叶结点个数*$\alpha$ 。

​	$L_\alpha(T) = L(T) + \alpha|T|$ ， 剪枝以后的Loss：$L_\alpha(t) = L(t)+\alpha$  => $\alpha_{lim}=\frac{L(t)-L(T)}{|T|-1}$	





##GBDT

### 基本原理

GBDT就是由一系列 弱分类器（经常为 Decision Tree） 组成的加法模型，后续的树都是在减小前面结果的残差，用$F_m(x)$ 来表示累积至m层的输出结果。

$F_m(x) = F_{m-1}(x) + \mathop{argmin}\limits_{h \in H}\ Loss(y,\ F_{m-1}(x)+h(x))$

所以实际上每一层tree训练的是上一层的Loss的负梯度，所以整个流程是：

```
1. Initialize F0(x), Loss(x, y) -> L=Loss(x, y)
2. Find F0(x)=argmin(Loss(y, F0(x)))
3. for i in range(m):
			compute gradients: gi
			find h(x) = argmin(Loss(Fi(x)-gi))
			Fi = F0 + a*h(x)
...
```



对于回归来说，经常食用MSE，即$Loss(x, y) = (F_{m}(x)-y)^2$ ，$\frac{\part{L}}{F_m(x)}=2(F_m(x)-y)$，也就是最小化残差。



### Bias & Variance

CART 本身作为树模型会比较不稳定，数据波动带来影响比较大，也就是方差会比较大。在enemble学习中boosting减小bias，bagging减小variance，作为具体实现的XGBoost，LightGBM都有都具有两种技巧，使用方差大的分类器更容易通过最后的ensemble得倒更稳定的输出。

High Bias是指模型过于简单，loss表现不好，High Variance是指模型过于复杂，以至于容易受到异常数据的干扰，波动大。

对于Bagging来说，假设每个model都是I.I.D.：

​	$E(\frac{\sum_i X_i}{n}) = \sum_i \frac{E(X_i)}{n} = E(X)$

​	$Var(\frac{\sum_i X_i}{n}) = \frac{Var(X)}{n}$ 

对于Boosting来说，模型之间相互依赖，Boosting的目标就是最小化Loss function -> 所以从定义上来说就是在减小bias，而由于模型之间相互关联，对于variance来说没什么帮助



### 回归和分类

对于回归问题比较简单，只要找到合适的Loss function按照流程走就行了

对于分类问题，可以参考Logistic Regression，每一层输出的结果都经过一层$Sigmoid()$， 即 $p_m = \frac{1}{1-e^{-F_m(x)}}$，这样就可以按照LR的流程定义 $L_m=-\sum ylog(p)+(1-y)log(1-p)$

求导之后下一层的object就是残差$y-p_m$ 



### 优缺点

优点：

* 预测很快，可以在预测阶段并行运算
* 在连续稠密的数据特征上表现很好
* 树模型不需要对数据归一化等处理

缺点：

* 训练过程只能串行，因为每层有依赖
* 在稀疏数据上由于一次分裂就相当引入很高阶的非线性，效果不佳，不如SVM/NN



##XGBoost





##LightGBM

首先LightGBM是一个boosting框架，基于数模型的学习算法（官网的描述）。。

基本上lightGBM就是GBDT的一个实现，分类器一般都是CART，但他有很多优化点。



### 一些优化算法

1. 目标函数二阶泰勒展开

   LightGBM和XGBoost一样，不仅仅是一个简单的Loss function，为了防止过拟合，还有一个正则项。

   目标函数为：$OBJ=\sum_i L(F_m(x_i), y_i) + \sum_{j=1}^{m}\Omega(f_j)$ ，前项为Loss function， 后一项是正则项。

   GBDT 本身两棵树之间是计算Loss function的负梯度来作为新一轮的target，这本质上是泰勒展开一阶形式，也可以说是梯度下降：

   $L(y_i, \hat y_{i}^{t-1}+f_t(x_i)) = L(y_i, \hat y_i^{t-1})+g_i f_t(x_i)$

   而XGBoost 和LightGBM 用了二阶导数，

   $Obj_i = \sum_i (L(y_i, \hat y_i^{t-1})+g_if_t(x_i)+\frac{1}{2}h_if_t^2(x_i)+\Omega(f_t)$

   对于优化，每一层树只保留相关项  $=> Obj_i = \sum_i (g_if_t(x_i)+\frac{1}{2}h_if_t^2(x_i)+\Omega(f_t)$

   

   先把正则项展开，LightGBM / XGBoost 定义了正则项：$\Omega(f_t) = \gamma T+\frac{1}{2}\lambda ||\omega||^2$

   其中$T$ 为叶结点的个数，$\omega_i$为每个叶结点的回归值，然后构成一个向量。

   

   把正则项带入之后可以得到：

   $Obj_i = \sum_i (g_i w_j+\frac{1}{2}h_i w_j^2) + \gamma T + \frac{1}{2}\lambda||w||^2 \\=\sum_i (g_i w_j + \frac{1}{2}(\lambda+h_i) w_j^2)+\gamma T$

   $min\ Obj_i = min\ \sum_i (g_i w_j + \frac{1}{2}(\lambda+h_i) w_j^2)+\gamma T \\=>w_j^*=-\frac{2\sum_{i,i\in j} g_i}{\lambda+\sum_{i, i\in j}h_i}$

   即对于每个叶结点来说存在这么一个理论上的最优值，$Obj_i^* = \frac{1}{2}\lambda\sum_j||w_j^*||^2 + \gamma T \\=\sum \frac{2G_j}{\lambda+H_j}+\gamma T$

   所以在计算分裂的时候能直接一点，分裂的时候本质上来说就是找loss function最小值。

   

   * 讲道理这不就是牛顿法和梯度下降的区别吗，本质上海赛矩阵就是二阶导，但一般来说求二阶导会很麻烦，但因为在这里，loss function 一般是mse，求起来就很容易。
   * 另外，有这么一种形式存在，也就说对于任何存在一阶导和二阶导的loss function都能直接写，应该说留了一定的自由度。

   

2. 把feature按照直方图来分割

   * CART 在选择特征进行分裂的时候，需要对每个特征的每个中间值进行计算。当数据量特别大的时候，首先需要排序，然后进行循环计算。在LightGBM中首先会对特征分箱做直方图，直接减小存储空间和计算时间。相当于说是以一定的分割精度来换计算资源的提升。但本来CART 就是一个弱模型，不那么精确的分割点不是特别重要。

   * LightGBM和XGBoost一样，在排序的时候只对非零值进行分箱操作。

   * LightGBM 还有一个优化，在做直方图的时候，父结点的直方图理论上来说是两个子结点的和，LightGBM也用了这个关系。在构建直方图的时候还是需要遍历该结点上的所有数据，因此可以先遍历数据少的结点，然后另一个节点直接用父结点与该结点直方图做差来得到。

3. LightGBM把按层生长的限制，拓宽成按叶结点个数来限制生长

   XGBoost在构建CART的时候还是限制树生长的最大层数来防止过拟合，同时这么做也可以在同一层中并行计算。但由于数据分布不一定均匀，结点之间数据量，重要度都不完全相同，这会强行分裂一些本来不用分裂的结点。

   LightGBM的Lead-wise策略，回先从当前所有叶结点中选取增益最大的进行分裂。但也有可能会导致树左右不平衡，出现一些比较深的结构。

4. Gradient-based One-Side Sampling ：单边梯度采样（GOSS）

   GBDT不想Adaboost有数据重要度的计算，因此可以利用每个数据的梯度来作为权重，可以用来采样。GOSS的目的是减少梯度小的数据的采样频率，用梯度比较大的数据来采样。但也不能直接不用小梯度的数据，导致数据分布改变。

   ```
   GOSS 流程
   1. 输入 X: training_data, d: iterations, a: 大梯度数据采样百分比, b: 小梯度数据采样百分比
   	 Initialize Loss function, weak_learner
   	 
   select top N |gradient|: a*len(X), random choose data b*len(X)(1-a)
   for i in range(d):
   	preds = model.predit(X)
   	g = loss(preds, y), weight = uniform()
   	sort(abs(g))
   	topSet = g[:a/100*len(X)]
   	randSet = randomChoose(g[a/100*len(X)+1 : g[a/100*len(X)+b/100*len(X)])
   	
   	useSet = [topSet, randSet]
   	weight[randSet] *= (1-a/100)/b
   	
   	newModel.fit(L(useSet, -g[useSet], weight))
   ```

   所以LightGBM越往下训练数据会越来越少

5. Exclusive Feature Bundling, EFB（互斥特征捆绑算法）

   高维的特征可能具有稀疏性，eg：one-hot，可以通过两个特征相乘来捆绑构建直方图，降低特征数量。

   LightGBM把特征结合的问题转化为图的问题，先判断哪些需要绑定，绑定集合使用分箱的结果

   ```
   Initialize: F: features, K: max conflict counts
   					  => G: graph : {Vertexs: F, Edges: weight(conflicts)}
   					  
   order = sortByDegree(G)
   bundles = {} , conflictBundles = {}
   for i in range(order):
   	needNew = Fasle
   	for b in bundles:
   		cnt = countConflict(b, F[i])
   		if (cnt+len(conflictBundles) < K):
   			needNew = False
   			bundles.add(i)
   			
   	if needNew:
   		add b as a new bundle
   
   ```

   ``` 
   # Bundle features
   
   F : one bundle of exclusive data
   
   suppose A:[lb1, ub1], B:[lb2, ub2]
   				-> B=>[ub1, ub2-lb2+ub1]
   ```

   本质上这个算法是用来解决稀疏特征的，但稀疏特征本来就不适合CART，应该做一些处理。个人感觉应该也不会有很多在用他前不处理数据的情况。



### 工程优化

1. LightGBM直接支持Catagory

   一般来说catagory需要通过编码来作为输入（one-hot / {1,2,3,4….}），尤其在类别多的时候使用one-hot会影响效果。

   LightGBM支持many-vs-many来分割特征。按照类别特征分类的话，最多需要确定$2^k-1$钟情况。LightGBM 基于 Fisher的《On Grouping For Maximum Homogeneity》论文实现了$O(nlog(n))$的时间复杂度。

   对于分类特征的排序：$\frac{G}{H} = \frac{\sum Gradient}{\sum Hessian}$，主要是因为类别排序要先引入一个比较单元，拿一阶导除以二阶导算是梯度变化率的倒数

   （细节还没仔细研究）

   

2. 高效并行

   LightGBM支持不同的并行方法，feature/data/voting

   XGBoost的主要思路是并行地在不同的机器上计算不同特征的最优分割点，然后同步取最优。这个方法主要是把数据的不同特征划分到不同机器上，需要频繁通信得到全局的结果。

   LightGBM每台机器都存全部数据，得到方案后在本地划分。（这么牺牲存储空间真的会快很多吗，毕竟结果通信数据量很少）

   

   对于数据的并行来说，最简单的不同机器本地画直方图然后合并，开销为$O(machines*bins*features)$ 

   LightGBM 合并直方图的时候 对于不同feature 也分到不同机器上，再得倒一个global。

   

   LightGBM 会在本地选取TopK的特征，合并的时候只合并选出来的特征。（这种方法在特征很多的时候比较有用）

   

3. 缓存优化

   LightGBM对缓存有优化（还没理解）



### 参数调优

一些关键的参数：

* boosting：gbdt(default)，rf，dart，goss

  rf：random forest

  goss：就是前面提到的根据梯度绝对值来采样的轻量化的gbdt

  dart：Dropouts meet Multiple Additive Regression Trees，基本上来说就是用drop随机丢弃DT

* num_iterations / num_rounds：好几个名字，就是决定树的做多有多少颗，因为每一棵树都在优化loss梯度项，所以相当于NN里面的一个iteration

* num_leaves / max_depth：一般两个都要设置，防止生成深度过深的不均衡的树。

* tree_learner：是分布式中的一些设置，也就是分布式优化中那些并行的优化

  - `serial（default）`, single machine tree learner
  - `feature`, feature parallel tree learner, aliases: `feature_parallel`
  - `data`, data parallel tree learner, aliases: `data_parallel`
  - `voting`, voting parallel tree learner, aliases: `voting_parallel`

* num_threads / n_jobs:  可以设成-1

* device_type：cpu，gpu，cuda

* verbose_eval: 几个iteration做一次validate

* min_sum_hessian_in_leaf：是为了防止leaf的过多分裂

* bagging_fraction / subsample：只选部分数据训练

* feature_fraction: 只选部分特征训练

* early_stopping_round：如果最后**会和没有提升，提前结束

