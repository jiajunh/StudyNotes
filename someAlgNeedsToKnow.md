[toc]



### MST

#### Kruskal

对于一系列Nodes，需要找到一个MST，按照edge的cost排序，如果当前的edge两边的node没有连通，添加edge 进graph，否则继续下一个。判断又没有连通，用union set。

```
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