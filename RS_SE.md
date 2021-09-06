[toc]

##Before Everything

ad-hoc-retrieval：it typically signifies a solution for a specific purpose, problem, or task rather than a generalized solution adaptable to collateral instances. 



Wildcard queries：通配符查询，一些常见的通配符：*, ? , $…



tf：term-frequency



RSV：Retrieval Status Value，算是一个重要的指标



结果指标：

Precision：$precision = \frac{TP}{TP+FP}$ = (选中结果中正确的) / (所有选中的结果)

Recall：$recall = \frac{TP}{TP+FN}$ = (选中的结果中正确的) / (所有相关的)，之前还一直不懂为啥叫recall，现在感觉就是对应上了recall这个过程。





##Information Retrieval

Information retrieval (IR) is finding material (usually documents) of an unstructured nature (usually text) that satisfies an information need from within large collections (usually stored on computers).



基本单元为`term`，一般来说是单词，词语。



### Boolean Retrieval



布尔检索可以首先建立一个数据库，存储所有出现过的term，然后把要搜索的信息与每个term建立布尔关系，根据向量的位操作能快速检索。

最简单的方法：建立term-document-matrix，eg：每一列代表一个document中出现过的单词。这么做的问题在于存储，一旦词库或者文件量特别大，整个矩阵会变得很大，而且会有稀疏性。因此可以想到，指储存值为1的部分。

<img src="/Users/jiajunh/Library/Application Support/typora-user-images/image-20210825111455640.png" alt="image-20210825111455640" style="zoom:25%;" />



Inverted Index：因为词库过大而造成的稀疏性可以通过反向编码来解决。即不是建立文件中出现过的词在整个词库中的向量，而是记录词库中每个item，map到所有出现过这个词的文件。这样一来内存中可以只存每个term和对应的链接指针，相当于一个hashmap，value是一个链表。在这中间把document编码以后，需要防止重复id，所以经常需要排序。排序还有一个好处，就是在取交集/并集的时候会更加有效，直接就是对两个有序链表的操作。当链表很长以后，可以加个skip pointer，加速查询/变成红黑树。

<img src="/Users/jiajunh/Library/Application Support/typora-user-images/image-20210825111422475.png" alt="image-20210825111422475" style="zoom:25%;" />



有时后可能需要一定得模糊查询，用一些符号和标准来获得一定程度的模糊，正则表达式什么的。。。

eg：

​	\space：分割

​	/k ：within k words of 



虽然boolean retrieval没太多应用场景，这里还有几个过程中的概念。

* Tokenize：简单的来说就是把句子分成单词/词组，也就是分词的步骤。
* normalize：Map text and query term to same form，把词都统一成标准的形式，eg：USA应该和U.S.A是一个term
* stem/LEMMATIZATION：语法处理，词有不同的形式，eg：现在时，过去时可能需要对应到同一个词
* stop word：一些助词，没有意义的词，可以直接去掉，一般都会有一个stop word维护列表。eg：the，a。。。。web search现在也经常不用stop word。个人感觉主要是因为这种都靠语义来映射query 和 term

每个语种语法什么的都不同，所以应该都要有一套不同的规则。



Phrase Query：

很多时候但个单词的词库并不完整，很多时候需要两个词或多个词组成的词组。

一些方法：

* 连续两个词组成一个term（FP，数量会很大）

* Positional indexes，形成一种格式。可以先通过单词搜索，然后通过相同的doc，merge。

  ```d
  <term, #docs contains term;
  doc1: idxes
  doc2: idxes
  ```

  

####存储过程：

为了上大量数据能够存储，有很多进化方法

1. BSBI（blocked sort-based indexing algorithm）分成blocks，每个block存<item_id, doc_id> 的pair，对所有block中的pair排序，把临时内容存在磁盘里，然后合并。
2. SPIMI（single-pass in-memory indexing） 对每个item建立记录表，然后直接append，碰到没遇到的term，新开一个。排序合并。
3. 分布式（基本都用这个）MapReduce / Spark。。。。



减少存储空间使用的压缩技术

1. 对于字典：需要 [term(string), freq(int), pointer(address)]，最浪费的地方在于固定词的长度，可以把term 变成一个指针，减少浪费空间。多级索引。。。（都是数据库里的那种东西，讲道理数据量真的大，对于单机来说，按照数据库的方式构建是效率最高的，只不过，即多级索引 + B+tree + sortmerge/gracehash ）
2. Postings compression：小trick，把排序后的索引，用差分来存，可以减少很多大数。一些编码上的优化，像是连续数字的压缩之类的。字典序压缩，gamma压缩。。。



####Dictionaries and Tolerant Retrieval:

首先，存储的term-doc的数据结构应该是最经典的 hashmap + tree 的结构，也就是dict。

处理通配符的时候，一些方法，

eg: *

首先可以使用字典序来构建 term，因为*还需要考虑两遍相符的字符串，所以可以正序，逆序构建两颗B-树，然后取交集，就得到了所有符合的结果。

eg：$ + *，可以变换顺序

K-gram：先对字符串取固定窗口，把所有的字串找出来，然后对这些字传取交集。



Spell Corrections：

* Edit distance：对于字符串s1，s2，就是最少字符操作（改变一个字符，加上一个字符，减去一个字符，交换相邻两个字符）(可以用dp做)
* k-grim+jaccard：jaccard coefficient：$\frac{|A\cap B|}{A\cup B}$ 只选取k-grim过程中jaccard coff大于一个阈值的结果。可以先用k-gram来选取合适的term，然后和query 比较jaccard coff。



Noisy Channal Model：

Input：w (word with errors)，通过bayes rule 选取概率最大的词 -> output：$\hat w$ 

$\hat w = \arg\max\limits_{x \in W}  \ p(x|w) = \arg\max\limits_{x \in W} \frac{p(w|x)p(x)}{p(w)}$



有了distance，就可以代入贝叶斯公式中，找到概率最大的term。



对于仅限于term和query的情况

其中：$p(x)$ 就是每一个候选term的出现概率，$p(w|x)$就是edit prob，它和键盘分布等因素有关，可以事先绘制一个matrix来表示各种编辑初现的频率，eg：$p(w|x)=\frac{matrix[edit_{ij}]}{count(x)}$ ，然后经典的bayes的稳定项，分子分母都加一个常数来稳定。



对于需要结合语境的情况，可以使用term 和context 来计算k-gram，但基本上语义还是得nlp。



#### Problems

1. boolean retrieval经常会导致没有结果或者特别多的结果。
2. 对不了解原理的用户体验不好，需要用户做的工作比较多，要求比较高。（毕竟可能多输入一点点query 就搜不到东西了）







### Ranked Retrieval

排序检索最大的不同就是会返回一个经过排序的documents序列，并且一般来说并不需要通配符或者其他的语法，只需要输入几个词/词组。排序检索带来的另一个好处就是，有效缓解了数据量带来的问题。因为boolean的时候你不知道重要度，所有结果都有相同的重要程度，需要全部返回。有排序条件了以后，只需要返回top-k。所以最大的问题就是如何去排序。



很容易想到，可以给每个document打分，对于每一个query，根据匹配程度给document打上[0-1]，根据之前的经验，最简单的可以想到Jaccard Coefficient：很简单，对于query/doc 长度也没什么限制。但问题也很明显，没有考虑频率的问题。



首先可以来改进打分的机制，对于打分的机制来说，首先有几个很容易想到的规则：

1. 如果query term没有出现在document中，那么应该是0分。
2. query term出现频率越高，分数应该越高。



####TF-IDF

由此最简单的改进就是直接统计每个term在document中出现的频率（也就是把boolean中的 term-doc matrix中的bool变成统计总量），然后就构成了一个term counts的向量。（当然这么做，一个很明显的问题就是顺序不影响结果）



在打分的时候，虽然说频率高，分数也高，但不意味着应该是线性关系。一种对应计算方法是log

TF：term-frequency，term在docment中出现的次数

$score = \sum\limits_{t \in(q\cap d)} 1+log_{10}(tf_{t,d})$



然后就到了经典的TF-IDF，TF-IDF是出于一条比较经典的规则，即 Rare terms are more informative than frequent terms，这其实也挺好理解，想想stop word就是最直接的例子，出现的又多又没用。

IDF：inverse document frequency，DF是指含有term 的document的数量，在idf使用log主要是为了把idf数值稳定住，减小一些影响。当然，idf本身对于一个词的term是没有影响的，只是相当于一个系数。

$idf= log_{10}(\frac{N}{df_t})$



$tf-idf: w_{t,d} = log(1+tf_{t,d}) * log(\frac{N}{df_t}) \\score=\sum\limits_{t \in (q \cap d)} tf-idf$



现在可以把t-d matrix 进行填充，对于每一个文件来说，就是一个稀疏的向量，那么和query来比较的话，可以把他们先映射到一个向量空间中，然后计算相似度 / 距离反比。既然是两个向量的相似度，那么最基础的就是归一化以后算点积，也就是cosine。

这里有不同的tf-idf计算方式（感觉log / log_avg 应该更加泛用）。

<img src="/Users/jiajunh/Library/Application Support/typora-user-images/image-20210826135701340.png" alt="image-20210826135701340" style="zoom:25%;" />





#### Probabilistic Information Retrieval

一般来说，需要根据一些特征来计算document和term之间的相关性。特征可以是tf，idx，boolean vector。。。。

然后假设相关的话：$R=1$，不相关的话：$R=0$

可以用$\frac{p(R=1)}{p(R=0)}$来表征结果

先把多条件贝叶斯写在这里：贝叶斯本质都是写成联合概率，然后一层一层嵌套条件概率的。

$P(X,Y,Z)=P(X|Y,Z)P(Y|Z)P(Z)=P(Y|X,Z)P(X|Z)P(Z)$



##### Binary Independence Model (BIM)

所谓的binary其实也就是boolean，所以本质上BIM就是boolean retrieval，所谓的independence指的是term在每一个document中是independent。

本质上BIM模型的近似就是在计算tf-idf



对于BIM来说，向量$x$就是boolean中的vector，所以先写成概率模型呗，

$O(R|q,x) = \displaystyle\frac{p(R=1|q,x)}{p(R=0|q,x)}= \displaystyle\frac{\frac{p(x|R=1,q)p(R=1|q)}{p(x|q)}}{\frac{p(x|R=0,q)p(R=0|q)}{p(x|q)}} = \displaystyle \frac{p(R=1|q)}{p(R=0|q)}* \frac{p(x|R=1,q)}{p(x|R=0,q)}$

对于一个推荐系统来说前面一项对于一个query是常数，而后面一项由于query和document独立，所以可以展开成连乘。这里的x代表每个词是否出现，那么联合起来就是对于每个$x_i=1/0$ 在相关文档query的概率的积。

$O(R|q,x)=O(R|q)*\displaystyle\prod\limits_{x_i=1}\frac{p_i}{r_i} \displaystyle\prod\limits_{x_i=0}\frac{1-p_i}{1-r_i}$ 

那么出现这个形式的话就很容易想到log-likelihood，直接取log，后面那部分就成了

$RSV=\displaystyle \sum\limits_{x_i=1} \frac{p_i(1-r_i)}{r_i(1-p_i)}$

其实也不用写那么复杂。。。。

就是和query相关的document数量为n，总共为N

对于$R=1，\ \ x_i=1: s,\ \ \ \ x_i=0: n-s$

对于$R=0，\ \ x_i=1: S,\ \ \ \ x_i=0: N-n-s-S$

$p_i=\frac{s}{S+s}, \ \ \  r_i=\frac{n-s}{N+n-2s-S}$

这里还有一个简化，就是不相关的如果当作全部数量的话，那么$r_i=idf_d$





#####Okapi BM25（best match 25）

BIM原来是为短文本设计的，对于现代文本来说，需要同时考虑到文本长度。BM25的目标就是在不引入过多参数的情况下能够修正这个问题。

其实BM25的主要思路还是很简单的，就是整个query与文档的相关性有三个主要的特征，term的权重，term和document的相关性，term和query的相关性。

* 单词权重，这个主要还是由idf来表达

* term-document：在之前是用tf来表示的，但是在BM25中有一点比较重要的变化，就是tf和相关性肯定不能说是一个线性关系。所以在tf中我们会使用log来表达这种相关性。在BM25中有一个阈值的概念，对于每一个文档，阈值都与文档有关。那么这里就可以引入一些参数，把文本长度的信息也表达进去。从公式上来讲，一般是用这种形式：

  $\displaystyle \frac{(k_1+1) tf_{td}}{k_1 ((1-b) + b\frac{L_b}{L_{avg}})+tf_{td}}$

  这里面$L_d, \ L_{ave}$ 分别是文档的长度和平均长度，$k_1$是一个正值的参数。$b$ 是0-1的一个参数。

  那么这里$k_1$ 主要就是控制词频的参数，$b$ 就是控制文本长度的参数

   

* query-term：这个也同样加一个参数来控制query长度带来的影响

  $\displaystyle \frac{(k_3+1)tf_{tq}}{k3+tf_{tq}}$



$RSV = \sum \displaystyle \frac{N}{df} \displaystyle \frac{(k_1+1) tf_{td}}{k_1 ((1-b) + b\frac{L_b}{L_{avg}})+tf_{td}} \displaystyle \frac{(k_3+1)tf_{tq}}{k3+tf_{tq}}$



BM25 变形很多，其中一种就是把idf给变了 $\displaystyle\frac{N-df_t+0.5}{df_t+0.5}$ ，但本质上最核心的思路就是加几个控制参数，和文本长度的特征。



BM25F：主要是针对document分割成几个zone，这其实也是有道理的，一个document可能有很多不同的部分，标题，文本内容，其他非文本内容等。那么有必要对每个部分分配不同的weight，然后求和。。。（都是很容易的思路）



处理non-text features：

除了header， anchor，title。。。这些特征还可能会有一些不是文本的特征，eg：pagerank

pagerank就是正整数，google对网页的排名算法，是对超链接的分析算法。

这样的话，需要把RSV分两部分考虑，加上一项非文本特征项：

$RSV = \sum c_i(tf_i) + \sum \lambda_iV_i(f_i)$

$V_i(f_i) = log \displaystyle \frac{p(F_i=f_i | R=1)}{p(F_i=f_i | R=0)}$

$\lambda$ 是可调节的系数。

$V_i$ 可以选多种形式，。。。 一般就取 $V_i = log(\lambda_i^{'}+f_i)$



#### Evaluation

判断搜索结果还是挺重要的。

最重要的：User Needs

User Needs 会被转化为 queries，relevance对应的是user needs 而不是queries



* Precision：P(Revalant&Chosen | Chosen)

* Recall：P(Received&Chosen | Relevant)
* accuracy = (tp + tn) / (tp + fp + fn + tn)，accuracy真的对于检索系统来说不是很好，毕竟绝大部分都是不相关的。

这两个标准对于有没有排序都可以用，算是普适的标准了。





对于有排序的推荐/搜索，还有一些其他的标准

Binary：

* Precision@K：很魔幻的名字，居然还有个特殊字符。。。它其实就是计算前一部分数据中的relevance
* Average Precision：average of precision@K
* Mean Average Precision：不同query的Average Precision 的均值。。。

Beyond Binary：

* Discounted Cumulative Gain (DCG)：有两点假设：越相关的document比边缘相关的要有用的多，不太相关的不太会被检查到。所以从高到底排序应该设立一个衰减系数，一般用$\frac{1}{log(x)}$

  $DCG = Rel_1 + \displaystyle \sum \frac{Rel_i}{log(i)}$

* *Normalized DCG (NDCG)：Normalize DCG at rank n by the DCG value at rank n of the ideal ranking，ideal ranking 就是从高到低relevance 排序。

  DCG最大的问题就是在搜索/排序过程中，结果的数量都不固定，不固定数量的累积和并不能表征不同查询之间的效果。

  $NDCG@N = \frac {DCG@N} {IDCG@N}$

  IDCG@N 是理论上排序以后前N结果的和。

* Mean Reciprocal Rank（MRR）：$MRR = \displaystyle \frac{1}{N} \sum \displaystyle \frac{1}{rank_i}$ 就是对于不同query下，最相关的document的rank的倒数和。很神奇的标准。



既然已经到这一步了，就顺便吧一些其他的评价标准都整理一下吧：

* F-score：最多的应该是F1 score，$F_1 = \displaystyle \frac{2}{recall^{-1}+precision^{-1}}=\displaystyle \frac{2precision * recall}{precision + recall}$

  $F_\beta = (1+\beta^2) \displaystyle \frac{precision *recall}{\beta^2 precision + recall}$

  F-score 是Precision 和 Recall的调和平均值，

* Precision-Recall curve（PRC）：一般来说都是偏锯齿状的，好的结果应该是都偏上

* Receiver Operating Characteristics（ROC）：$\displaystyle \frac{True Positive Rate}{False Positive Rate} = \displaystyle \frac{Recall}{\frac{FP}{FP+TN}}$ 

  ROC就很有用了，一般来说是从左下到右上，过（0，0）和（1，1）的线。ROC上每一个点都代表了一个的分类器在不同阈值下的结果。

  假设两个类别平均分布，随机选择的话就是接近y=x的直线，效果越好，越接近（0，1）

* Area Under Curve（AUC）：就是ROC下半部分与坐标轴的面积。









