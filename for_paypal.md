

### 问题：

1. 为什么需要对数值类型的特征做归一化？

   一般是均值/高斯 normalize，正常来说normalize一定程度上影响特征数值的分布，把原先较大/小的跨度直接拉到合理的范围内，在SGD学习速度一定的情况下，能更快/稳定收敛。

2. 如何处理catagorial，

   可以直接按照原来的大小关系映射到数字上，/转化为onehot，/二进制编码，二进制编码更紧密

3. A/B test

   离线评估无法完全消除模型过拟合的影响，线上会有数据的异常，丢失等问题

   A/B test 是对用户进行分桶，对好分布相似，

4. 在模型评估过程中，有哪些主要的验证方法，它们的优缺点是什么

   Holdout， 直接切分 train-test，CV-慢但是会稳定，variance会降低，自助采样：进行n次放回的随机采样，在n趋向无穷大的时候，对n个样品抽样n次，根据极限，约有1/e的数据没抽到过，把这些作为val

5. 超参的调法

   网格，随机，在local minima附近继续搜索

6. K-means 的优缺点：

   需要先确定K值，受到初始值的影响比较大，噪音影响大

   一般k-mean k的选择需要先试好几组比较log变化率。

   K-mean++：初始点随机，后续根据和之前点的距离来改变采样频率 -> 本质上就是优化初始点的选择

   ISODATA：K可以变化，会设置一个方差和数量阈值，方差过大且数量足够，可以分裂成两个类。两个类之间距离小的话合并

7. 典型的loss function

   一般来说会选择0-1函数，光滑的凸函数

   Hinge loss, cross entropy，MSE。。。。。



###优化算法的迭代：

1. 直接求解

   对于数据量不多的情况下，可以直接求导=0的点

2. 迭代法，展开成泰勒公式

   $L(x+\Delta \ x) = L(x) + \Delta x \  G(x) => \Delta x = \alpha G(x)$

   牛顿法：对于接近零点的$x_0$

   对于凸函数，$f(x) = f(x_0) + (x-x_0) G(x_0) => dx =- \frac{f(x_0)}{g(x_0)}$

   直接让f(x)取一阶导数 $d(x) = -\frac{G(x)}{H(x)}$

   收敛更快，要求hessian





### Linux 

fork: fork（）函数通过系统调用创建一个与原来进程几乎完全相同的进程

一个进程调用fork（）函数后，系统先给新的进程分配资源，例如存储数据和代码的空间。然后把原来的进程的所有值都复制到新的新进程中，只有少数值与原来的进程的值不同。	

在子进程中，fork函数返回0，在父进程中，fork返回新创建子进程的进程ID

```
//用fork实现pipe


```





### SQL related

别忘了：

COUNT(), 

DISTINCT()

LIMIT 1 OFFSET 1 找第几高

需要有null的时候可以外面再套一层select

```
select (
	select *
	from .....
) as XXX
```



字符转换

LOWER， UPPER， TRIM

date_format(date, %Y-%m-$s %H:%M%S)

设置变量

DECLARE A INT;

SET A=m-1;



条件判断：

1. if (judgement, true_statement, false_statement)

2. ```
   case
   	when cond1 then statement1
   	when cond2 then statement2
   	else
   		result
   end;
   ```



UNION



开窗函数：

​	函数名(列名) OVER(partition by 列名 order by 列名) 。	

```
rank() over (partition by XXX count(1) order by decs)  
```

Lead/lag, 神奇的方法可以读取当前row 之前/之后的其它行中的值







按照日期聚类

除了 直接group by

还可以 group by MONTH(date), YEAR(date),…..

规整化 日期格式 date_format(date, ‘%Y-%m-%d ’)







### System Design

- 設計一個停車場，API要能夠停車/移車，每個車佔的空間不同
- 設計一個工廠排班系統，指派不同的worker來處理不同的task
- 設計一個排行榜，實作API更新路況和回傳前k名領先人員



### brain teaser









