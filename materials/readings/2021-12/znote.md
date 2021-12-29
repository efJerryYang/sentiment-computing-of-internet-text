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

> Tips:[How to Read a Paper Efficiently (By Prof. Pete Carr)](https://www.youtube.com/watch?v=IeaD0ZaUJ3Y)

## Thumbs up? ...

> 这篇基本很可能已经算是情感计算的开山之作了，从过去基于主题的(topic)分类，到作者他们尝试的基于情感(sentiment)的分类。当然，由于算是介绍性质的论文，作者他们应用的算法都很基本，他们这篇论文的关注点也不在于对准确率的提高，而是对情感计算这个问题的初步引入，并且和过去基于主题的文本分类进行对比。

作者首先验证了情感分类和主题分类的不同，并且验证了机器学习算法应用于情感分类是可行的，得到的分类结果会好于人直觉想出的分类标准。在测试的结果中，Naive Bayes表现最差，SVM表现最好，但差别并不显著。

除此以外，在实验过程中得到了两个和主题分类相比显著的不同：

1. 已经验证了的相同的传统机器学习算法在topic categorization和sentiment classification上面的表现差异是明显的，情感分类的准确度均大大低于主题分类的结果
2. 在所有尝试的特征中，竟然只有unigram是有着最好的分类结果的特征，加入特征词汇的频率也不能改善分类结果，这和主题分类中引入特征词汇的频率得到的结果截然相反。

当然，作者通过重新查看语料库对这些问题进行了解释，并用"thwarted  expectations"来描述这一现象，先抑后扬。主要体现在评论性文字中大量的某一情感方面的词汇并不一定是作者真正想表达的内容，换句话讲，评论者可能会先对电影的各方方面面进行批评，但最后的总结却明显表达对这一电影本身的喜爱之情。这一点在评论中是很常见的。

作者最后提出一点可能的未来改进的方向，就是分辨出选用的features，是否是真的在对评价的关注对象在进行描述。

>  Hence, we believe that an important next step is the identification of features indicating whether sentences are on-topic (which is a kind of co-reference problem)

当然，作者反复的在提及Turney他们的工作，等下补充阅读。

## Thumbs up or thumbs down? (Turney, 2002)

作者定义了一个术语（或者是使用了之前工作中被提出的术语）*sematic orientation*，用来对phrases的情感倾向进行计算，计算原则如下：

> ..., the semantic orientation of a phrase is calculated as the mutual information between the given phrase and the word “excellent” minus the mutual information between the given phrase and the word “poor”. A review is classified as recommended if the average semantic orientation of its phrases is positive.

目前读到的这两位，基本算是情感计算的开创者，此前虽然也有相关的应用(Tong, 2001)，但Tong本人也没能足够重视情感计算这个领域本身，只是把情感计算作为一种工具。Tong这个有意思的工作应该流传的是相当广泛的，课上秦老师还提到过她们也做过类似的东西。

>As far as I know, the only prior published work on the task of classifying reviews as thumbs up or down is Tong’s (2001) **system for generating sentiment timelines**. This system tracks online discussions about movies and displays a plot of the number of positive sentiment and negative sentiment messages over time. Messages are classified
>by looking for specific phrases that indicate the sentiment of the author towards the movie (e.g., “great acting”, “wonderful visuals”, “terrible score”, “uneven editing”). Each phrase must be manually added to a special lexicon and manually tagged as indicating positive or negative sentiment.

至于使用sematic orientation这一术语来对表达的情感进行评估的，还有作者提及的Hatzivassiloglou and McKeown's work (1997)，他们处理的是预测形容词的sematic orientation，被作者评价为与自己的工作最为接近的（在我看来，更多是核心思想上的接近）。

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

