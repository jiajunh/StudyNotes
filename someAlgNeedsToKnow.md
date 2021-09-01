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

