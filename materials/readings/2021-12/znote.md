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
- [x] [Thumbs Up or Thumbs Down? Semantic Orientation Applied to Unsupervised Classification of Reviews](https://arxiv.org/ftp/cs/papers/0212/0212032.pdf)
- [x] [Effective LSTMs for Target-Dependent Sentiment Classification.](https://arxiv.org/pdf/1512.01100)
- [ ] [Seeing stars: Exploiting class relationships for sentiment categorization with respect to rating scales](https://arxiv.org/pdf/cs/0506075.pdf)
- [ ] [Learning extraction patterns for subjective expressions](http://www.aclweb.org/anthology/W03-1014)
- [ ] [Convolutional Neural Networks for Sentence Classification.](https://arxiv.org/pdf/1408.5882)
- [ ] [TextCNN with Attention for Text Classification](https://arxiv.org/ftp/arxiv/papers/2108/2108.01921.pdf)
- [ ] [Seeing stars when there aren’t many stars: Graph-based semi-supervised learning for sentiment categorization](https://aclanthology.org/W06-3808.pdf)

> Tips:[How to Read a Paper Efficiently (By Prof. Pete Carr)](https://www.youtube.com/watch?v=IeaD0ZaUJ3Y)

## 1-Thumbs up? ...

> 这篇基本很可能已经算是情感计算的开山之作了，从过去基于主题的(topic)分类，到作者他们尝试的基于情感(sentiment)的分类。当然，由于算是介绍性质的论文，作者他们应用的算法都很基本，他们这篇论文的关注点也不在于对准确率的提高，而是对情感计算这个问题的初步引入，并且和过去基于主题的文本分类进行对比。

作者首先验证了情感分类和主题分类的不同，并且验证了机器学习算法应用于情感分类是可行的，得到的分类结果会好于人直觉想出的分类标准。在测试的结果中，Naive Bayes表现最差，SVM表现最好，但差别并不显著。

除此以外，在实验过程中得到了两个和主题分类相比显著的不同：

1. 已经验证了的相同的传统机器学习算法在topic categorization和sentiment classification上面的表现差异是明显的，情感分类的准确度均大大低于主题分类的结果
2. 在所有尝试的特征中，竟然只有unigram是有着最好的分类结果的特征，加入特征词汇的频率也不能改善分类结果，这和主题分类中引入特征词汇的频率得到的结果截然相反。

当然，作者通过重新查看语料库对这些问题进行了解释，并用"thwarted  expectations"来描述这一现象，先抑后扬。主要体现在评论性文字中大量的某一情感方面的词汇并不一定是作者真正想表达的内容，换句话讲，评论者可能会先对电影的各方方面面进行批评，但最后的总结却明显表达对这一电影本身的喜爱之情。这一点在评论中是很常见的。

作者最后提出一点可能的未来改进的方向，就是分辨出选用的features，是否是真的在对评价的关注对象在进行描述。

>  Hence, we believe that an important next step is the identification of features indicating whether sentences are on-topic (which is a kind of co-reference problem)

当然，作者在反复的提及Turney他们的工作，等下补充阅读。

## 2-Thumbs up or thumbs down? (Turney, 2002)

作者使用术语*sematic orientation*（我直译为*语义指向*）用来对phrases的情感倾向进行计算，计算原则如下：

> ..., the semantic orientation of a phrase is calculated as the mutual information between the given phrase and the word “excellent” minus the mutual information between the given phrase and the word “poor”. A review is classified as recommended if the average semantic orientation of its phrases is positive.

目前读到的这两位，基本算是情感计算的开创者，此前虽然也有相关的应用(Tong, 2001)，但Tong本人也没能足够重视情感计算这个领域本身，只是把情感计算作为一种工具。Tong这个有意思的工作应该流传的是相当广泛的，课上秦老师还提到过她们也做过类似的东西。

>As far as I know, the only prior published work on the task of classifying reviews as thumbs up or down is Tong’s (2001) **system for generating sentiment timelines**. This system tracks online discussions about movies and displays a plot of the number of positive sentiment and negative sentiment messages over time. Messages are classified
>by looking for specific phrases that indicate the sentiment of the author towards the movie (e.g., “great acting”, “wonderful visuals”, “terrible score”, “uneven editing”). Each phrase must be manually added to a special lexicon and manually tagged as indicating positive or negative sentiment.

至于使用sematic orientation这一术语来对表达的情感进行评估的，还有作者提及的Hatzivassiloglou and McKeown's work (1997)，他们处理的是预测形容词的sematic orientation，被作者评价为与自己的工作最为接近的（在我看来，更多是核心思想上的接近）。Hatzivassiloglou和McKeown的工作，主要是利用给定的4步骤监督学习策略，使用连词来对形容词的语义指向进行推断：

1. 所有的conjunctions of adjectives是从给定语料库中抽取的；
2. 使用监督学习算法绘制出这些成对的形容词的语义指向图，结点为形容词，边标明语义指向是相同的还是不同的(sameness or difference)；
3. 使用聚类算法把相同语义指向的词汇分在同组中，在同一聚类子集中图的边会主要是sameness，贯穿聚类子集的图的边会主要是difference；
4. 由于positive的形容词使用频率会比negative的高，这会导致聚类结果中，使用频率较高的词汇会被认为是有positive的语义指向。

他们形容词分类结果的准确率在78%-92%之间，取决于语料库。而且，由于聚类算法不仅仅是二分类，这一算法还能够得出被分类到特定语义指向的符合度(goodness-of-fit)，所以是很好的结果。

> The algorithm can go beyond a binary positive-negative distinction, because the clustering algorithm (step 3 above) can produce a “goodness-of-fit” measure that indicates how well an adjective fits in its assigned cluster.

作者承认，他们的算法可以被直接应用到评论分类中，但作者小小的赞扬了一下自己2001年提出的PMI-IR算法，表示自己的算法更简单，更容易implement，并且可以处理phrases, adverbs，而不仅仅是isolated adjectives.

> **Todo:** Tong, R.M. 2001. An operational system for detecting and tracking opinions in on-line discussions. Working Notes of the ACM SIGIR 2001 Workshop on Operational Text Classification (pp. 1-6). New York, NY: ACM.
>
> Todo: PMI-IR 算法 Turney, P.D. 2001. Mining the Web for synonyms: PMI-IR versus LSA on TOEFL. Proceedings of the Twelfth European Conference on Machine Learning (pp. 491-502). Berlin: Springer-Verlag.
>
> Todo: Hatzivassiloglou, V., & McKeown, K.R. 1997. Predicting the semantic orientation of adjectives. Proceedings of the 35th Annual Meeting of the ACL and the 8th Conference of the European Chapter of the ACL (pp. 174-181). New Brunswick, NJ: ACL.

先到这里，回到情感计算正题，情感文本分类。

## 3-Effective LSTMs

> Duyu Tang, Bing Qin, Xiaocheng Feng, Ting Liu
>
> Harbin Institute of Technology, Harbin, China
>
> {dytang, qinb, xcfeng, tliu}@ir.hit.edu.cn

> 是秦老师她们的工作，谷歌学术搜索'sentiment classification'排到了第二，引用是六百多

这篇论文是在解决指定描述对象的情感分类问题，主要是使用target-dependent LSTM，以及后续提出的target-connection LSTM，这两种的分类准确度都远高于标准LSTM。老师们在使用LSTM的时候，是使用的两个LSTM作为整合上下文信息的方式。

> 我暂时没有理解为什么不直接使用bidirectional LSTM，或者对比二者结果。我推测可能的原因是，本文的主要核心是在处理target-dependent的问题，模型的选择时次要的（但我觉得这个说法不够充分，关键是使用双向RNN来处理上下文信息是很容易想到的），也可能只是相同的实现思想，只是老师她们没有专门说。这里就是我不熟悉模型底层实现导致的空想问题了。

有个问题是，文中地址和[Duyu Tang](http://ir.hit.edu.cn/~dytang)老师相关的链接已经失效了，重定向到[微软](https://www.microsoft.com/en-us/research/people/dutang/)的也失效了，至少给我提醒的一点是，最好有个自己的博客作为专门的展示渠道，或者使用非学校的第三方如共用的GitHub账号之类的。Word embedding这里，Stanford那边的链接还是有效的，他们的word embedding数据量更大些，结果自然会更好。但我有个想法是，没必要特别区分词嵌入预训练模型的数据来源，根本目的是让语义相关的词语具有较高的相似度，好吧，写着写着知道我自己在瞎扯，数据来源不同的话结果自然是存在着差异的，不同文本环境下，同一个词语的常用含义是会发生改变的，我们人不需要作区分大概只是阅读量够了，但模型训练量想要足够又变成数据量太大的问题了。

> Todo: (Jiang et al., 2011; Dong et al., 2014; Vo and Zhang, 2015)
>
> Long Jiang, Mo Yu, Ming Zhou, Xiaohua Liu, and Tiejun Zhao. 2011. Target-dependent twitter sentiment classification. ACL, 1:151–160.
>
> Duy-Tin Vo and Yue Zhang. 2015. Target-dependent twitter sentiment classification with rich automatic features. IJCAI
>
> Li Dong, Furu Wei, Chuanqi Tan, Duyu Tang, Ming Zhou, and Ke Xu. 2014. Adaptive recursive neural network for target-dependent twitter sentiment classification. In ACL, pages 49–54.

> 我可能需要看一下，文章中提到的用SVM和那些神经网络的做法，具体是解决的什么样的问题，和这篇论文的相关性如何，为什么这篇论文依旧认为处理target-dependent sentiment classification仍然是一个挑战。

> 其实真的感觉人的学习和神经网络的很相似，智力水平不足的容易欠拟合，训练量过大容易过拟合。

主要是从人自然的阅读方式是要关注上下文信息来判断针对对象的语义的进而引入使用上下文信息来做推断，这是很自然的想法，也是被很广泛利用的。这篇论文是处理RNN的应用，自然后面就会有使用CNN和transformer来搞东西的研究者。只是时间有限，不一定有精力读到那些使用和模仿attention机制的网络，但对现在而言应该都算得上是很成熟的技术了。

针对于对象的情感分类，可能难度确实要更大，最好的结果准确率也只有不到72%，相比于最初使用的机器学习方法对整个评论文本进行分类的结果（77%左右）而言，似乎有下降，尽管还没有充分了解，但考虑到技术进步的大趋势，大概率的可能是这一分类问题的困难程度要明显上升了。

细节方面，这篇文章提及了之前有的工作是使用监督学习的方法，但我没有理解为什么这篇文章要专门强调target-dependent sentiment classification仍然是一个困难。主要是我没有理解，这篇文章和所提及的那几篇论文的关系是什么，是改进了已有使用的方法，还是提高了分类的准确度，还是验证了什么猜想，只读到这里我还不能得到很明确的答案。或者说，这篇论文没有之前读的那两篇对自己实现了什么说的那么明确。虽然摘要的地方提及了是最好的，但没有数据的比较确实不是很具体。

> Majority of existing studies build sentiment classifiers with supervised machine learning approach, such as feature based Supported Vector Machine (Jiang et al., 2011) or neural network approaches (Dong et al., 2014; Vo and Zhang, 2015). Despite the effectiveness of these approaches, we argue that target-dependent sentiment classification remains a challenge: how to effectively model the semantic relatedness of a target word with its context words in a sentence.

提及直接将target-dependent features整合到特征里面来使用SVM比较麻烦，那这里应该是实现了一种免于人工标注的方法。很有可能，这篇论文也是相当的早期的结果。但我现在必须要搞清楚，这篇论文中使用的两个LSTM和Bi-LSTM的区别。以及，我并不清楚的是，我所学习到的东西是哪段时间的内容，所以我不能以后来者的角度来看问题。

>The model could be trained in an end-to-end way with standard backpropagation, where the loss function is cross-entropy error of supervised sentiment classification.

> 我对我们国家的学术界有偏见，所以会不自觉的往这个方向去想。如果我想要证明，他们是为了增加自引用或者模仿他人的工作，我需要假设他们没有模仿他人的已有成果，并且去证明这个结论是错误的可能性很大，这需要我去阅读论文搞清楚时间线问题。
>
> 现在先把这篇论文分析清楚。

当然，已经注意到的一个明显的问题是，重复的语句过多，或者就是复制粘贴前后文，虽然有主题的原因，但我觉得这样写也不太合适。可能是英语的原因，我们使用第二语言毕竟没有母语者那么熟悉。

> Todo: 清理出所有前后文混用的部分

```
abstract:
 Empirical results show that modeling sentence represen-
tation with standard LSTM does not perform well. Incorporating target information into LSTM
can significantly boost the classification accuracy.
intro:
 Empirical results show
that the proposed approach without using syntactic parser or external sentiment lexicon obtains state-of-
the-art classification accuracy. In addition, we find that modeling sentence with standard LSTM does not
perform well on this target-dependent task. Integrating target information into LSTM could significantly
improve the classification accuracy.
```

另一个明显的问题是，超链接失效，主要是url，说明对于维护的重视程度不够，前面已经提到过。

这篇文章主要是在对三个模型进行对比

>Afterwards, we extend LSTM by
>considering the target word, obtaining the Target-Dependent Long Short-Term Memory (TD-LSTM)
>model. Finally, we extend TD-LSTM with target connection, where the semantic relatedness of target
>with its context words are incorporated.

先简要介绍了RNN的功能，并解释原始RNN存在的问题，然后讲LSTM解决了梯度消失的问题，从而被更广泛的采用。然后老师们用原始LSTM来处理target-dependent的情感分类问题，但这样得到的结果是target-independent的，不符合目标。于是老师们改进出了TD-LSTM和TC-LSTM，都是很小的改动，但我其实很想知道这要怎么改代码，pytorch给的LSTM的结构应该是不能让两个LSTM共用同一个softmax的吧，怕不是要手写代码，当然，用bi-lstm就没有这个问题（但现在我还不清楚那个时候双向的lstm有没有被提出来）。

> 问题就是，现在看不到代码了，原来的链接已经失效了

将目标字符串作为最后预测的结果是合理的，这样从直觉上讲是更加充分的利用了信息的。

> We favor this strategy as we believe that regarding target string as the last unit could better utilize the semantics of target string when using the composed representation for sentiment classification

TD-LSTM基本可以看做就是稍稍改动的Bi-LSTM（或者就是，我现在尚不确定），能起到效果是我已知的内容。但这个TC-LSTM确实就是很精妙的东西了，训练时即使用了原来词向量本身预训练的结果，又考虑到在当前语境下，target和contexts的关系，或者说*mutual information*（前面两篇论文里面提到的一个术语），这样从直觉上是能够提高分类效果的，而结果验证了这一点，这一部分应该是这篇论文设定最精巧的地方。

> However, we think TD-LSTM is still not good enough because it does not capture the interactions between target word and its contexts.

>Based on the consideration mentioned above, we go one step further and develop a target-connection
>long short-term memory (TC-LSTM). This model extends TD-LSTM by incorporating an target con-
>nection component, which explicitly utilizes the connections between target word and each context word
>when composing the representation of a sentence.

> An overview of TC-LSTM is illustrated in Figure 2. The input of TC-LSTM is a sentence consist-
> ing of n words {w1, w2, ...wn} and a target string t occurs in the sentence. We represent target t as
> {wl+1, wl+2...wr−1} because a target could be a word sequence of variable length, such as “google” or
> “harry potter”. When processing a sentence, we split it into three components: target words, preceding
> context words and following context words. We obtain target vector vtarget by averaging the vectors
> of words it contains, which has been proven to be simple and effective in representing named entities
> (Socher et al., 2013a; Sun et al., 2015). When compute the hidden vectors of preceding and following
> context words, we use two separate long short-term memory models, which are similar with the strategy
> used in TD-LSTM. The difference is that in TC-LSTM the input at each position is the concatenation of
> word embedding and target vector vtarget, while in TD-LSTM the input at each position only includes
> the embedding of current word. We believe that TC-LSTM could make better use of the connection
> between target and each context word when building the representation of a sentence.

说实话，我不是很理解为什么老师们吧target-independent作为自己论文的核心，我感觉，TC-LSTM才是这篇论文最精巧的地方，也才是这篇论文的亮点。但这篇文章却重心是在讲TD-LSTM，也可能是在时间上，这是类似Bi-LSTM首先提出的论文之一，这一点还有待考证。

前面铺垫的太多了（），要不是读到TC-LSTM我都要读不下去了（）。

## 小结（1-3）

\# 写上面论文的小结，最好稍微理出来一个时间线（这需要从后往前读，现在先把下面的弄完再说）

- [ ] 要开始做PPT了（3h）
- [ ] 先读完这篇（1h），然后稍微读一下另外几篇的大致内容，略过实验部分（1h）
- [ ] 把这篇报告写完整（1h）

## 相关不熟悉术语

* n-fold cross validation

​	![n-fold cross validation_00](znote.assets/n-fold cross validation_00.png)

固定模型结构和当前模型的训练过程（学习率，训练轮数，优化器），更改模型训练的数据集划分，获取最终情况下的训练结果

> 也就是说，最好把训练模型的过程封装成config文件，对应于更新的训练过程（即训练策略作为一种固定的参数，这种参数在训练模型的过程中被记录，并作为此后模型训练的可选择参数，那么一开始可以有个default参数，然后在default基础上进行修改）
>
> 这样确保了模型训练过程的统一性和稳定性，不同的训练过程可以认为是不同的模型。对，就是这一点，应当认为一个模型保存框架，包含了不同的**训练策略**和**模型结构**及其**参数**。

> Todo: 封装通用的模型训练代码，yml还是json还是什么，之后要考虑。

> 其实还要考虑的问题就是，这模型本身的参数都是可以调整的，要如何决定也是个问题。

> 知识图谱，盲区，但是是个重要的东西，相当于模拟了我们人类知识学习的方式。

