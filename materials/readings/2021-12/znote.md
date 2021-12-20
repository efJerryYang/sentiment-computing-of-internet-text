## 常用网址

文献搜索

* [谷歌学术](https://scholar.google.com.hk/?hl=zh-CN)

文献阅读/引用

* [arXiv.org](https://arxiv.org/)
* [scihub1](https://sci-hub.se/), [scihub2](https://sci-hubtw.hkvisa.net/)

文献翻译

* [谷歌翻译](https://translate.google.com/?sl=en&tl=zh-CN&op=translate&hl=zh-CN)
* [一帆文档翻译](https://fanyipdf.com/)（需禁用代理）

## 阅读顺序

- [x] [Thumbs up? Sentiment Classification using Machine Learning Techniques](https://arxiv.org/pdf/cs/0205070)
- [ ] [Learning extraction patterns for subjective expressions](http://www.aclweb.org/anthology/W03-1014)
- [ ] [Convolutional Neural Networks for Sentence Classification.](https://arxiv.org/pdf/1408.5882)
- [ ] [Effective LSTMs for Target-Dependent Sentiment Classification.](https://arxiv.org/pdf/1512.01100)
- [ ] [TextCNN with Attention for Text Classification](https://arxiv.org/ftp/arxiv/papers/2108/2108.01921.pdf)



## Thumbs up? ...

### 主要内容

* 介绍了此前的工作主要是基于topic的文本分类，而基于sentiment的文本工作较少

* 作者解释了，使用电影评论来作为语料库的好处在于不需要手动的标注，根据评分或者星级可以直接得出三类评级(positive, neutral, negative)，但这里作者只关注了正面和负面的评价。corpus,label

* 相比于对主题的分析，人更加擅长于对文本情感进行分析，而机器做的好的方面和人是相反的

* 直观的策略是基于选定了两类特征(positive and negative)的，对于文本的词频进行统计，作者基于此做了preliminary experiment来制造了3个baseline。

  在这个实验上，人类直观想出来的特征（introspection）往往不如机器基于统计再加人的提取。

  这里作者只是做一个baseline所以没有特别的选择出最优的分类器。

* 疑问：人工语料分类器选择的人太少，只有两个人

* 疑问：基于主题的文本分类任务，机器能达到的准确率作者没有提及。其实是提及了的，只是在后文 6.2 Results。基于主题的多分类任务的结果(容易达到90%+)都是远超情感二分类的。

* 然后，作者表示，这篇论文主要处理的问题是，探索基于主题的文本分类方法能否直接应用于情感分类任务上。即把情感分类任务视为positive and negative两个主题。

* 作者应用了3中传统机器学习方法，朴素贝叶斯，最大熵，支持向量机。

* 疑问：没有尝试使用条件随机场进行检查(CRF)，尝试检查的模型太少。

* 简单的介绍了三种机器学习模型在本实验中的应用方式（因暂时未学，先略过了）

* 最终使用的是没有偏置的语料，简化实验的考虑。

* 接下来作者详细的介绍了他们做的工作，从数据集划分，数据来源和处理手段介绍，一些他们处理时的非传统手段(unconventional step)，来源于过去的一项研究。

* 有一段没有太看懂，"For this study, we focused on features based on unigrams (with negation tagging) and bigrams. Be
  cause training MaxEnt is expensive in ..."

* 介绍了他们最初的设想的探索结果，说明了文本主题分类任务的方法，不可以直接迁移到情感分类任务上。对于机器而言，主题分类任务是相对简单的，但情感分类任务是困难的，在前文作者提出的一点解释是，情感分类更需要understanding，即使机器能比人类对于主题分类细小差别掌握的更好，并不能有助于机器理解文本。

* 然后作者介绍了他们更多更深入的研究结果，作者发现，只考虑feature出现与否，而不是出现的频率能够得到更好的结果，尤其是在SVM上。作者发现，这一特点和过去曾经报道过的关于topic categorization的结论相反。原来的结果是关于naive bayes的。这样尝试的主要原因是最大熵模型的限定，具体需要在补学之后再思考。

  > We speculate that this indicates a difference between sentiment and topic categorization — perhaps due to topic being conveyed mostly by particular content words that tend to be repeated — but this remains to be verified.

  在这个考虑下，作者不再使用频率作为NB和SVM的特征。

* 接下来作者使用了bigrams，尽管显然不再满足朴素贝叶斯的假设，但研究表明，即使不满足假设也不意味着朴素贝叶斯表现会更差。这一点我比较感兴趣。

  > [Domingos and Pazzani1997] Pedro Domingos and Michael J. Pazzani. 1997. On the optimality of the simple Bayesian classifier under zero-one loss. Machine Learning, 29(2-3):103–130

* 从直觉上讲，bigrams能更好的体现上下文环境，但是结果没有体现这一点，反而对比之后发现，只使用bigrams相比于纯粹使用unigrams在三种模型上都下降了5%左右的正确率



* Turney的文章，被作者提了很多遍（毕竟是和他做了最相似的工作）

### 相关不熟悉术语

* n-fold cross validation

​	![n-fold cross validation_00](znote.assets/n-fold cross validation_00.png)

固定模型结构和当前模型的训练过程（学习率，训练轮数，优化器），更改模型训练的数据集划分，获取最终情况下的训练结果

> 也就是说，最好把训练模型的过程封装成config文件，对应于更新的训练过程（即训练策略作为一种固定的参数，这种参数在训练模型的过程中被记录，并作为此后模型训练的可选择参数，那么一开始可以有个default参数，然后在default基础上进行修改）
>
> 这样确保了模型训练过程的统一性和稳定性，不同的训练过程可以认为是不同的模型。对，就是这一点，应当认为一个模型保存框架，包含了不同的**训练策略**和**模型结构**及其**参数**。

> Todo: 封装通用的模型训练代码，yml还是json还是什么，之后要考虑。

> 其实还要考虑的问题就是，这模型本身的参数都是可以调整的，要如何决定也是个问题。

> 知识图谱，盲区，但是是个重要的东西，相当于模拟了我们人类知识学习的方式。

