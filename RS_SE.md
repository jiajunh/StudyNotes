[toc]

##Before Everything

ad-hoc-retrieval：it typically signifies a solution for a specific purpose, problem, or task rather than a generalized solution adaptable to collateral instances. 



Wildcard queries：通配符查询，一些常见的通配符：*, ? , $…



tf：term-frequency



RSV：Retrieval Status Value，算是一个重要的指标



CTR：Click Through Rate

CVR：Click Value Rate



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

除了header， anchor，title。。。这些特征还可能会有一些不是文本的特征，eg：pagerank, hit rank, doclength, trust rank 等

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

  IDCG@N 是理论上排序以后前N结果的和。这个东西基本应该就是人工标注了。讲道理人工对相关性排序也很难啊。

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







#### Efficient Ranking

Safe Ranking / Non-safe ranking：选出的top-K是否是严格分最高的K个

对于document数量很多的情况，显然sorting并不是个好方法。并且在选择top-K的时候，计算cosine是bottlenect。

那么在计算的时候只选取一部分document然后在计算cosine。这个选择过程自然可以想到doc中term的idf。

1. 只选取含有idf大的term，这样的话像the， a这种会直接被过滤掉，然后相当大一部分doc会直接被过滤掉。
2. 只选取doc有很多term，eg 含有3-4个term，这可以在posting index vector中直接取交集。
3. Champion List：对每个term 事先计算有最高权值的一些doc，把这些称为champion list。这样之后只用处理这一部分doc。



在ranking的时候，我们是需要both relative & authoritive

relavance 的评价标准就是cosine scores

Authority is typically a query-independent property of a document：eg：有wiki的，有很多citation的paper，pagerank这种

对于authoritive，每一个doc应该都有一个数值来表示，基本会scale到 [0,1]。

所以综合的分数用Net Score来表示：

$net\_score(q,d) = g(d) + cosine(q,d)$



然后就可以取top-K了，这里当然要fast！！

1. 那么结合上面的一些操作，可以在champion list中按照$g(d)$ 排序，这些都可以离线完成，并且这样可以使posting中出现term的平均时间变短，有效提高效率。当然改进一点就可以用$g(d)+tf-idf$ 来排序。
2. clustering：选取$\sqrt{N}$ 个cluster 然后聚类。这样的话就相当于每个query，可以直接先找到最相关的doc，在这个cluster中做排序。直接把数据减少到$\frac{1}{\sqrt{n}}$，另外甚至可以多层clustering。



Tierd Index

既然说到Champion List，champion list就直接把doc分成两类了，一类是high score，另一类是low score。=>分成两个tiers，那么一个拓展的点就是把posting list再按照重要度多分几分，就成了tiered index。整个流程就是只遍历最高分的，除非他没够K个，在依次计算。当然如果在计算的时候数量已经够了，可以直接early stop



##### Safe Ranking

前面的都是对于non-safe ranking的一些策略，毕竟很多时候只要结果不是很离谱都没太大区别，就不需要safe ranking。

WAND scoring

。。。。。。to be continued







### Text Classification

Naive Bayes， KNN， K-mean。。。LR，Decision Tree，。。。。

Naive Bayes本质就是计数。。。所以来说训练很快，在spamming filter有一席之地





### Low dimension vector

前面所有的doc_vector, query_vector其实都不太会应用，因为这些one-hot vector都太大了而且很稀疏，所以基本上nlp最核心的一点就是把这些映射到一个低维向量空间中，这样很多问题就直接解决了。

#### Tranditional Way（Latent Semantic Indexing/Analysis）LSA

先回到原始的问题，我们现在有一个超级大的query-doc矩阵，然后想要找到一个低维的空间。

SVD：先不考虑矩阵过大是否能处理的问题，首先这是一个降维的问题，也就意味着SVD/PCA这种肯定是能想到的。当然PCA只是SVD的特例，一般也都直接SVD



#### Neural Embedding

然后就到了embedding的时代，那么经典的方法肯定就来了呀，

##### Word2Vec

* CBOW
* skip-GRAM

Two (moderately efficient) training methods 

1.  Hierarchical softmax 
2.  Negative sampling 
3.  Naïve softmax



所谓的embedding基本上就是几层网络，然后把维度压缩下来，那么这里面首先Loss function要定义好。

对于word2vec来说，基本的方法就是填词，先不说CBOW，一般是skip-gram比较有效。在定了一个窗口大小之后，我们经过网络embedding，然后decode出来的词就是output，那么我们输入中心词，去预测边上的词，根据现有的词库就可以有一个概率分布。

$p(w_{t+m} | w_t)$

那么在整个窗口中的联合概率就又出现连乘的likelihood，然后转化一下就出来了loss function

$max \ \prod\limits_{t}\prod\limits_{i \in [-m,m]} p(w_{t+i}|w_t, \theta)$

$L = - \frac{1}{T} \sum\limits_{t} \sum \limits_{i} log \  p(w_{t+i}|w_t, \theta)$

这计算的时候会用到center word 和 context word，他们都是经过同一个网络embedding得到的。那么条件概率可以通过softmax来表示

$p(o | c) = \displaystyle \frac{exp(u_o^T v_c)}{\sum exp(u^Tv_c)}$

这里我也第一次知道为什么softmax叫softmax，max是因为他取最大值作为分类结果，soft是指小概率事件他依然有一定的值。。。



另外还有一些很有用常见的验证方法：

1. 拼写相关的验证：比如单复数/不同时态的动词，他们的vector相减应该都接近0

2. 语义相关：通过词义queen-woman=king-man

   

#####Glove（Global Vectors for Word Representation）

SVD利用的是全局的一些共有特征，word2vec用的是局部的特征，Glove同时用了两方的特征。

Glove引入的是一个Co-occurrence Probabilities Matrix

首先有一个word-word matrix，$X_{i,j}$ 指的是在语料库中出现在$word_i$ 上下文中 $word_j$ 的次数，然后就有了概率 $P_{i,j}=\displaystyle\frac{X_{i,j}}{X_i}$

有这两个矩阵引出Co-occurrence Probabilities Matrix

$Ratio=\displaystyle\frac{P_{i,k}}{P_{j,k}}$  有一定的意义和规律，

1. i,k相关，j,k相关 => 1
2. i,k不相关，j,k不相关 => 1
3. i,k相关，j,k不相关 => 很大
4. i,k不相关，j,k相关 => 很小

Glove认为词向量在经过一定的映射关系之后能够呈现这种规律，在向量空间里面相关性直接相减，所以其实也很好表示

$F(w_i, w_j, w_k) = F((w_i-w_j)^T w_k) = \displaystyle\frac{P_{i,k}}{P_{j,k}}$

然后通过exp()联系起来

$P_{i,k} = exp(w_i^T w_k) => w_i^T w_k=log(\displaystyle \frac{X_{i,k}}{X_i})=log(X_{i,k})-log(X_i) = log(X_{i,k}) + b_i + b_k$

这样就有了新的Loss function

$L=\sum(w_i^Tw_k+b_i+b_k - log(X_{ik}))^2$



所以本质上我觉得他是发现了一种词之间相关性的东西，然后把skip gram 的loss部分重新替换掉了。



skip-gram 还会有一些采样，根据词出现的频率来采样。



##### Hierarchical softmax 

主要就是把原来的hidden->output的网络用一种新的huffman tree的softmax来替代了。它是一个二叉树的结构，它每个节点都是一套参数加上一个sigmoid，用来分类，到leaf就对应到了单词。

其实本质上来说，每一个词都是一个节点，事先可以先把所有词通过0-1编码先对应到整棵树上。在这棵树上，每个节点都会进行一次soimoid分类，计算出来的就是走路线的概率。然后按照这条路线一直往下直到对应的词，就可以用likelihood把loss优化了。当然要注意的是，每个节点都是用输出的vector和各自的weight来计算的，并不是nn连乘下来的关系。



##### Negative Sampling

这也是一个提高效率的方法。主要更新的点在于onehot编码的结果只有一个是1，其他全为0。那么求导更新的时候整个vec都要参与求导，她们认为这样太耗计算资源了，所以只随机取所有0中的几个参与更新。。。很暴力。那么负采样的词会按照一个概率来计算，频率越高越容易被选中。

$\displaystyle \frac{f_i^{\frac{3}{4}}}{\sum f_i^{\frac{3}{4}}}$









##### Dual Embedding Space Model (DESM)

word2vec 中虽然说他有两个网络，但一般只取前面那个input=>hidden，后面那个hidden=>output一般只是用来train的。有时候IR中也会需要两个都用，主要是在计算相似度的时候，整场都是用in embedding，DESM说query-doc应该用in-out。

一般来说document vector就是所有term vector取均值，而在计算query-document similarity的时候DESM认为query是用in vector，document用out vector。

个人感觉主要是都用in embedding，更加强调的是词之间的相似（包括语法拼写和语意），而in-out的话就是更加符合上下文连在一起的情况，也就是word2vec的训练目的。











### Learning to Rank

Training data: <query, doc> pairs, $c_i$ relevance ranking

GBDT：略，写过了。。。



#### RankNet

Assumptions & settings

* $x_i=>f(x_i, w)=s_i$
* 对于两个document，和一个query，表示两个document哪个更相关，$P_{ij}=P(d_i>d_j)=\displaystyle \frac{1}{1+e^{-\sigma(s_i-s_j)}}$
* Loss function使用cross entropy：$L=-\bar{P_{ij}} log(P_{ij})-(1-\bar{P_{ij}}) log(1-P_{ij})$



$L = -2S_{ij} log(\displaystyle \frac{1}{1+e^{-\sigma(s_i-s_j)}}) - (1-S_{ij})log(\displaystyle \frac{e^{-\sigma(s_i-s_j)}}{1+e^{-\sigma(s_i-s_j)}}) \\ =(1-S_{ij})\sigma (s_i-s_j)+2log(1+e^{-\sigma(s_i-s_j)})$

$S_{ij} \in \{0,1,-1\}$



$\displaystyle \frac{\partial L}{\partial s_i}= \sigma ((1-S_{ij})-\displaystyle \frac{2}{1+e^{-\sigma(s_i-s_j)}})$

$\displaystyle \frac{\partial L}{\partial w}=\sigma ((1-S_{ij})-\displaystyle \frac{2}{1+e^{-\sigma(s_i-s_j)}})(\displaystyle \frac{\partial s_i}{\partial w}-\frac{\partial s_j}{\partial w})=\lambda_{ij}(\frac{\partial s_i}{\partial w}-\frac{\partial s_j}{\partial w})$



这里可以看出$\lambda_{ij}$ 描述了两个doc之间的desire change，这里还可以定义一个$\lambda_i=\sum\limits_{j\in A}\lambda_{ij}+\sum\limits_{k\in B} \lambda_{ki}$

这就可以直接写成这种形式，分别问和i相关的错误顺序的数量。所以RankNet的lambda就是各种排序错误的梯度的和。。。



#### From RankNet to LambdaRank

LambdaRank的核心就是在与之前这个$\lambda_{ij}$ 可以通过scale by NDCG相关的参数来优化。他用的就是交换i，j以后NDCG的变化量

$\lambda_{ij} = \displaystyle \frac{-\sigma}{1+e^{-\sigma(s_i-s_j)}}|\Delta NDCG|$





####From LambdaRank to LambdaMART

MART可以通过GDBT来训练，所以lambdaMART就是LambdaRank 和 MART的结合。其实也就是一个GBDT的实现了。

<img src="/Users/jiajunh/Library/Application Support/typora-user-images/image-20210907184551997.png" alt="image-20210907184551997" style="zoom:20%;" />





### Link Analysis（本质上就是graph）

Link analysis首先就是要构筑一个graph，在这里基本上就是超链接提供的edge



#### PageRank

PageRank就是Link Analysis的一个应用。

PageRank Score最简单的想法就是随机开始，随机跳转到相连的网页，长期以后的稳定值就是score。但很多情况就是会走到头。Teleporting，设定一个几率跳转到随机其他网页，其他时候依然random walk，这样走到头以后还能跳出去。

pagerank是一个0-1的数

这么一来就是一个markov chain了，当然如果是平均概率， 那么有更多link指向的page自然会有更多的几率被访问到。并且因为有teleport的机制，就不存在循环周期的markov chain，那么就一定会有一个steady state。

$aP = a $ => a就是left eigen vector

对应下来，pagerank就是每个page steady state的概率值。

但是用解eigen vector的方法来计算显然不现实，page数量肯定是一个很大的数，构建出来的transition matrix就是一个巨大的稀疏矩阵。



当然，这种最简单的teleport肯定很有局限性，简单的改进就是先把按照topic把page打上tag，这样一来可以按照topic来teleport，毕竟直接跳转到一个随机的不相关的page效果肯定不如相似领域内的page好。在改进一步就是personalize pagerank，但这对于每个用户来说都是不同的。为了能够做到这一点，假设已经有了用户的一些基本信息，那么我们可以先把各个topic的比例都计算出来，然后整体的ss distribution就是各个topic之下的线性和。



#### Hubs and Authorities / Hyperlink-Induced Topic Search (HITS)

这两个东西是两种score，就是原来把一个page只打一个score，先在分成两个hub score 和 authority score。这么做主要是由于在搜索的时候，可以把一个网页认为有两个主要的部分，尤其是*broad-topic searches*。那么经常就是我做这种搜索期望的是由专业人员，专业机构提供的信息，这样的网页就被称为是authoritive。当然也有会把这些信息汇总起来的那种网页，就叫hub page，hub page往往会指向authoritive pages。所以好的authoritive page 往往也是被很多hub pages指向。

初始setting：初始页面$v$，$v->y$

$h(v)=a(v)=1$

$h(v) <= \sum\limits_{v->y} a(y)$

$a(v) <= \sum\limits_{v->y} h(y)$

那么一般这种情况，先把它写成Graph的形式，把Adjanct Matrix写出来。然后就有了

$h <= Aa=AA^Th, \ \ \ a<=A^Th=A^TAa, \ \ \ \ $

这里显然看出来是不可能直接取等号的，不然不停带入就爆掉了。但这个形式就很像是eigen value的形式，所以

$h = \frac{1}{\lambda}AA^Th, \ \ \ a=\frac{1}{\lambda}A^TAa$



对于一个query，首先找到含有含有term 的所有的pages，这个被称为root set，但这还不够，在加上root set中指向的page和指向root set中page的page构成base set。这么扩大set的原因就是可能query 搜索的是一个比较抽象的词，然后很多非常相关的具体的doc往往并不包含在root set中，但base set把这些也考虑进去了。

从graph的构建角度来说，就相当于base set就是root set往外扩充了一层。把base set作为计算hub /authorities score的基础集合。之后要选出top hub/authorities 只要直接疯狂迭代就行。虽然说正常需要一个特征值来scale，但只选top的话scale无所谓，只是为了防止迭代爆掉。。。然后这就被叫做hits了。。。





### Crawling and near-duplicate pages

爬虫要爬数据问题还挺多的，首先肯定是必须要分布式，不然效率太低。然后要排查掉恶意网站，和去重，并且不要给一个服务器很大的压力。

<img src="/Users/jiajunh/Library/Application Support/typora-user-images/image-20210908140319041.png" alt="image-20210908140319041" style="zoom:15%;" />

####

因为爬虫需要经常各个网站询问ip地址，DNS肯定不允许非常频繁的询问，那么就需要把ip缓存，以及batch发送。

Parsing: URL normalization：在遍历link的时候需要把link的完整url扩充

Filters and robots.txt：robot.txt是一个爬虫协议，他会告诉搜索引擎哪些网页可以抓取，哪些不行。当然碰到了之后也需要缓存。

Duplicate documents：很多时候要重复去爬一些网站，看有没有更新。那么对于判断内容是否重复可以计算相似度。相似度可以用k-gram+jaccard similarity来计算，k-gram的集合叫做shringle。当然整个doc很大，那计算量也会很大。所以可以对每个doc生成一个sketch-vector









##Recommendation System (RS)

传统的推荐系统可以主要分为两种

1. Content-based systems：这种方法主要以item的属性，相关性为主
2. Collaborative filtering：这种方法会主要比较user-item的相关性。
3. Latent factor based：

RS中最重要的两个东西就是item和user，这两个属性建立出来的matrix叫做Utility Matrix，每一个值都表示了user-item pair的degree。这个矩阵肯定是一个很大很稀疏的矩阵。在这个基础上建立的RS目标就是要填补上边的空缺，并且只需要填写一部分就行了。

Utility Matrix的核心问题：

1. 收集信息：打分，满意度之类的，
2. 从已知的pair去估计未知的pair，还有一个问题就是new item/user 会有cold start的问题，因为没有任何可以推断的数据。



### Content-based Recommendations

Item Profile：既然content-base 是依靠item之间的相关度来选择推荐内容的，那么就应该对每一个item构建一个profile，也就是每个item构建一个特征向量。

当然还需要构建user profile，最简单的就是对utility matrix中user-item pair中不为0的pair，可以按照feature 对每个item先进行加权平均，得到的就是user profile，然后利用cosine similarity来计算user-item的相似度。

content-base 他好就好在没有item冷启动的问题，只要来的item能够构建出feature，就能直接计算相关度，他对于用户数据的依赖仅仅限于用户自己，并不会要求有别人的数据。并且也正因此，这种方法推荐出来的东西都是针对每个用户的。当然认为的特征构建就是一个很难的事情，并且对于新用户来说没法推荐，因为没有相关历史数据（虽然个人还是认为用户冷启动就不是一个很关键的问题，如果真的要一开始就要你强推一些东西，那他用你干什么）。





### Collaborative Filtering

所谓的协同过滤也就是把utility matrix换个方向来看，content-base主要是关注item 的vector的相关性，而CF是关注相近的用户之间的行为，通过相似的用户来推荐。当然CF分user-user 和 item-item，主要核心思想是通过和其他人的比较。



如果最简单的把utility matrix的user行作为vector，那么Jaccard，cosine，Pearson correlation coefficient就很容易计算。那么先找到了K个相近的用户之后，给其他item的评价就是其他所有users 的平均值，或者说根据相关度的加权。

对于item-item CF，本质一样的，只不过把user vector变成item vector，找出相近的之后也是加权平均。

对于实际情况来说item-item相对来说特征是固定的，而user-user，user的喜好会随时间变化。



当然CF的优点就是可以没有feature，但肯定处理不了cold start的问题，要在相近的范围内进行推荐首先要求有一定的数据量存在。并且这回更容易推荐热门。



### Hybrid

单纯的CF，CB是在是太容易能看出问题了，一眼就能看出来怎么改进，最简单的就是把feature 引进到CF里面，那么计算相似度就更合理了。至少说new item的问题可以解决了。当然也可以先聚类。





### Latent Factor Models

SVD

对于每一个utility matrix中的pair可以认为是两个向量的点积，对于每一个item/user来说可以认为有latent vector，构成两个矩阵，其实本质上来说就是不太能解释的features，这其实和SVD的思路上就很契合。

现在utility matrix是item-user matrix A，那么item-matrix就是U，user-matrix是$\Sigma V$ 

写成loss function

$\min\limits_{P,Q} \ \sum(r_i-q_i^Tp_x)^2 + (\lambda_i\sum||p_i||^2 + \lambda_2 \sum ||q_x||^2)$

然后就可以SGD 迭代优化P，Q

当然实验表明不直接预测rating值，而是通过把它变成均值加上一个diff的形式形成学习残差的结构，效果可能会更好。当然，除了global的均值，还可以设定一个item偏置，user偏置，这样更加合理

$\hat{r_{ix}} = \mu + b_i + b_x + q_i^Tp_x$



在SVD的基础上，用户对于很多item有除了评分之外的交互行为。这里面还有相当大量的信息。

$\hat{r_{ix}} = \mu + b_i + b_x + q_i^Tp_x + q_i^T(p_x+|N(i)|^{-\frac{1}{2}}\sum y_j)$

$SSE = \sum(r_i-q_i^Tp_x)^2 + \lambda(\sum||p_i||^2 + \sum ||q_x||^2 + \sum||b_i||^2 + \sum||b_x||^2 + \sum ||y_j||^2)$

这个就是SVD++的形式了，SVD++的核心就在于把那些交互隐藏的信息同样变成一个vector



其实SVD的思想就是MF（Matrix Factorization），把整个大矩阵分解成User矩阵和Item矩阵，转化为一个个内积的形式。

但是注定MF系列的算法应用是很困难的，尤其是在确定latent vector的时候。







### 各种算法

#### FM系列

#####FM（Factorization Machine）

一般来说对于广告，推荐系统，会手工构建很多特征，之后最简单的就是用LR，GBDT这些来后续处理。但很多时候直接的特征并不够相关，而一些特征之间的组合能够造出更加强相关性的特征。FM就是为了解决在数据稀疏的时候，特征组合的问题。一般来说，最常见的情况就是CTR预测，CTR的特征都比较稀疏。。。



很多时候很多特征都是onehot编码的，这么一来数据就会有稀疏性。而构建融合特征的时候，最容易想到的就是二阶特征，因为多阶特征构造的时候数量是呈指数型上升的，往往二阶特征就能够表达足够的信息。并且如果存在稀疏性的话，二阶特征同样也具有稀疏性，因为只有当两个特征上的值都为1的时候，二阶特征才为1。

原始特征为$x_i$

$y(X)=w_0+\sum w_i x_i + \sum\sum w_{i,j}x_i x_j$

既然二阶特征数量又多，又特别稀疏，那么如果直接拿来训练，很可能有些特征上的权值会因为数据不充分，而使结果不够好。同样也可以注意到，二阶特征一共有$\displaystyle \frac{n(n-1)}{2}$ ，如果先考虑了顺序的情况下，就可以直接写成一个对称矩阵。而且我们是需要尽可能把参数的数量降下来，那么很自然就可以想到kernel。我们同样可以对这个W矩阵按照矩阵分解的方法每个元素$w_{i,j}=v_i^T v_j = <v_i,v_j>$ 

这样完全就是kernel了，并且参数直接从$\displaystyle \frac{n(n-1)}{2}$减少到O(n)个参数，只不过这里v像kernel一样，也是一个latent vector。

$\sum\sum w_{i,j}x_ix_j => \displaystyle\frac{1}{2} \sum\sum<v_i, v_j> x_i x_j - \displaystyle \frac{1}{2} \sum <v_i,v_i>x_ix_i \\ = \displaystyle\frac{1}{2} ((\sum v_i x_i)^2-\sum v_i^2 x_i^2)$



这一部分直接求导：

$\displaystyle \frac{\partial }{\partial v_i} = (\sum\limits_{j} v_i x_i x_j) -  v_ix_ix_i$



##### FFM （Field-aware Factorization Machine）

FFM就是一个进阶版的FM。FM中每一个feature只对应了一个latent vector，在FFM中分的更加细，FFM给所有的特征加了一个field属性，并且不同field中的特征融合带来的影响应该不一样。所以FFM对于每一个feature 要有一些相对应的fields。然后假设feature $j_1, j_2$ 对用的fields分别为$f_1, f_2$

$\phi_{ffm} = \sum\sum (w_{i,f_j}w_{j,f_i})x_1 x_2$

FFM采用的pair是对应的另一个的field向量。整体来说我觉得FFM是一种FM和原始的latent vector的一种权衡，本质上加入field就是对于特征数量的调节，现在加入fields属性后特征数量就*f，也就是原来的一个feature 对应一个latent vector，现在对应k个不同的vector。





##### Wide & Deep

这个东西基本上就是一个小融合，我们首先知道数据基本上是one-hot编码过的，并且可以通过两两融合特征来进一步构建更大规模的特征库。然后对于这些特征LR，GBDT。。。之类的操作，这样基本就是一个wide model了

然后Deep说白洁就是那种带embedding的NN结构。。。

所以本质上wide & deep就是纯粹把这两个东西做了一个ensemble。。。是在时没什么新意啊。所以这个东西就是融合了一下之后能够一定程度上避免一些负面效果。





##### DeepFM

deepFM其实可以看错是一种wide & deep的变种实现。他的wide部分就是FM出来的东西，当然他的latent vector可以是直接用embedding的结果来表示。当然可以顺便用FFM给特征增加一点fields。他的deep部分就是直接拿embedding的结果在加上一些NN层。最后把两边的结果融合起来。

本质上来说也没什么好讲的，和wide&deep一样是很容易想到的思路。

本质上引入deepNN就是为了减少计算量，对于稀疏的特征，最容易的方法就是embedding。





####CRF(Conditional Random Field)

CRF一般适用于序列标注的算法，conditional指的是条件概率，random指的是随机变量。CRF和HMM是很相近的，。
