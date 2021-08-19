### Optiver Realized Volatility Prediction

训练数据：不同stock_id下每10分钟内的交易信息(bid_price, ask_price)，需要预测之后10分钟内的volatility

一些常用features：

> wap: (['bid_price1'] * ['ask_size1'] + ['ask_price1'] * ['bid_size1']) / (df['bid_size1'] + df['ask_size1'])
>
> log_return: log(wap)
>
> price_spread: (['ask_price1'] - ['bid_price1']) / ((['ask_price1'] + ['bid_price1']) / 2)
>
> bid_spread:  df['bid_price1'] - df['bid_price2']
>
> ask_spread:  df['ask_price1'] - df['ask_price2']
>
> bid_ask_spread: abs(df['bid_spread'] - df['ask_spread'])
>
> volume_imbalance: abs((df['ask_size1'] + df['ask_size2']) - (df['bid_size1'] + df['bid_size2']))
>
> 
>
> 对时间开不同的窗口，选取每10分钟内time>150, 450, …. aggregation (sum, mean, std)
>
> 
>

1. 一开始直接使用 kfold 进行分组，设定num_fold=5，后来按照time_id进行groupkfold，并把stock_id和time_id都添加为feature，并从结果发现得倒（最后的结果，重要性最高的特征是stock_id和time_id）。如果是这样的话，说明模型特别依赖stock_id，从理解上来说，这不利于模型的泛化。



### CommonLit Readability Prize

当前可用model

0.462 roberta-large-cv-multilayer-addnoise

0.462 roberta-large-alldata-multilayer-addnoise

0.464 roberta-large-multilayer

0.465 Albert-large-multilayer

0.464 roberta-large-neural-process

0.468 roberta-base



1. Bert-base，使用exponential lr 0.551

2. 尝试bert-large，单一模型 没有CV ， overfit

3. bert-base / large， 添加CV 取平均

4. 尝试使用SVR / ridge regression 作为finetune最后结果， 效果不佳

5. 原先使用CLS起始符fine tune，改为使用CLS 和所有词向量一起fine tune

6. 尝试使用Roberta-base/large，roberta比bert训练数据更多，训练时间更长，增大lr后进入0.488

7. 在基础上对效果不佳的fold多次训练取val-loss最小值，最好到0.477

8. 对roberta不同层使用不同的lr，分成3-5个block，最后的全连接层lr增加到0.001，效果显著

9. 尝试使用pretrain 对数据集先跑3-4个epoch，做MLM训练，应该有点效果，但不太好验证

10. 配合cosine with warmup，把每一个epoch做一次val 变更为20-30step做一次val，效果显著，大大提升训练效率

11. 把最后全连接层换成不同的head，单个attention-head比多个更好，可能是数据过小

12. 对attention-head添加multisample dropout 使模型获得更多的泛化能力

13. 尝试使用swa 最后epoch 直接weight做running average，没什么用

14. 尝试使用roberta中间输出层进行加权平均，对极限分数提升有效

15. 尝试使用数据中的std error来构建加噪音的训练集，std error过大，手动减小，貌似有些效果

16. 对每一层的输出softmax进行观察，发现后面几层feature权重偏低，选取各个部分进行试验

17. 把目前为止的base和large，ensemble取平均，提升至0.46，进入前2%

18. 没想到人为数据增强真的有用, 人为给数据添加variance能一定程度上防止过拟合

19. 对于roberta-large 目前使用8-25层效果最好，只使用0-16层效果最差，全部使用效果略差于8-25，可以认为前面几层虽然softmax权重较大，但没有深层计算的细节信息。

20. 在base上把variance加上去 效果特别差，可以认为base几乎到极限了

21. 突然想到weight_decay可能会很有用，调了两组实验

22. 重新按照neural process的思路写了一个新的head，具有一定的随机性，比多层的roberta-head要好一点。目前可以确定数据整体方差过大所以很难提高分数了。

23. Neural process 的随机性其实还不错，只是需要把cross attention去掉，因为在训练的时候GPU显存不够batch数量很少。Attention对于短序列的效果可能不如CNN好用。本质上CNN也是加权求和，在局部的效果可能比attention好得多。

24. 最后一小段时间尝试了几个其他模型 但没有什么时间去慢慢调了

25. 没太多资源和时间没有尝试用一个model去拉取wiki的数据然后打分作为新的数据来源。这个竞赛的数据量很少，能扩充训练数据的话，鲁棒性会更好一些。

    