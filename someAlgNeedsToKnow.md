[toc]



### MST

#### Kruskal

对于一系列Nodes，需要找到一个MST，按照edge的cost排序，如果当前的edge两边的node没有连通，添加edge 进graph，否则继续下一个。判断又没有连通，用union set。

```C++
algorithm Kruskal(G) is
    F:= ∅
    for each v ∈ G.V do
        MAKE-SET(v)
    for each (u, v) in G.E ordered by weight(u, v), increasing do
        if FIND-SET(u) ≠ FIND-SET(v) then
            F:= F ∪ {(u, v)} ∪ {(v, u)}
            UNION(FIND-SET(u), FIND-SET(v))
    return F
```



#### Prim





#### Tarjan





### 字符串匹配算法

#### KMP

exkmp,ac自动机,后缀数组,manacher,hash





###最短路径

#### Floyd

#### Dijstra

#### BellmanFord





\1. 二分图匹配（匈牙利），最小路径覆盖
\2. 
网络流，最小费用流。
\3. 线段树.
\4. 并查集。
\5. 
熟悉动态规划的各个典型：LCS、最长递增子串、三角剖分、记忆化dp
6.博弈类算法。博弈树，二进制法等。
7.最大团，最大独立集。
8.判断点在多边形内。
\9. 
差分约束系统.
\10. 双向广度搜索、A*算法，最小耗散优先.





树状数组，线段树，treap，splay







### 一些随机的算法

#### 用rand(7)生成rand(10)

这个算法其实就是先找到一个比10大的随机数，然后不停拒绝比10大的随机数。

比较常见的就是先生成两个rand(7)，把他们相乘，就得到等概率的1-49，然后把大于49的拒绝掉进行while循环。。。



#### 蓄水池抽样（Random Reservior Sampling）

主要适用于大数据流。从数据流中抽取k个数据点，数据流中共有N个数据点。

对于k=1的时候，第i个数数保留的概率为$\frac{1}{i}$，保留就把之前的书丢掉。

对于k>1，先保留前k个数，对于第k+1个数，以概率$\frac{k}{k+1}$保留。如果被保留，那么对于前k个数中$n_r,r \in 1:k$ 在这一轮被保留的概率为

$p = 1 * (\frac{1}{k+1} + \frac{k}{k+1}*\frac{k-1}{k})=\frac{k}{k+1}$

对于第k+2个数以概率$\frac{k}{k+2}$ 保留，那么$n_r$ 被保留的概率为

$p=\frac{k}{k+1}*(\frac{2}{k+2} + \frac{k}{k+2}*\frac{k-1}{k}) = \frac{k}{k+2}$



```C++
vector<int> ReservoirSampling(vector<int>& results, vector<int>& nums, int k)
{
    // results.size(): k
    // nums.size(): N
    int N = nums.size();

    for (int i=0; i<k; ++i) {
        results[i] = nums[i];
    }

    for (int i=k; i<N; ++i) {
        int random = rand()%i;
        if (random<k) {
            results[random] = nums[i];
        }
    }

    return results;
}
```





### 查找算法

#### 字典树Trie

一般来说Trie要有insert和search的功能，delete什么的可能并不是必要的。这里写一个小模板

```C++
class TrieNode {
public:
  bool isLeaf;
  TrieNode* children[26];
  
  TrieNode() {
    this->isLeaf = false;
    for (int i=0; i<26; i++) {
      this->children[i] = nullptr;
    }
  }
  
  void insert(const string& key) {
    TrieNode* curr = this;
    for (int i = 0; i<key.size(); i++) {
        if (curr->children[key[i]] == nullptr) {
            curr->children[key[i]] = new Trie();
        }
        curr = curr->children[key[i]];
    } 
    curr->isLeaf = true;
  }
}
```

