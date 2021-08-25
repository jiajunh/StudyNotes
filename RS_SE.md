[toc]

##Before Everything

ad-hoc-retrieval：it typically signifies a solution for a specific purpose, problem, or task rather than a generalized solution adaptable to collateral instances. 



Wildcard queries：通配符查询，一些常见的通配符：*, ? , $…







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

K-gram：先对字符串取固定窗口，把所有的字串找出来，然后对这些字传去交集。



Spell Corrections：

* Edit distance：对于字符串s1，s2，就是最少字符操作（改变一个字符，加上一个字符，减去一个字符）(用dp做)
* k-grim+jaccard：jaccard coefficient：$\frac{|A\cap B|}{A\cup B}$ 只选取k-grim过程中jaccard coff大于一个阈值的结果。







