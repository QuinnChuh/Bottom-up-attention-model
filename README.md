# Bottom-up-attention-model论文笔记

> 本学习笔记是针对论文：[Bottom-Up and Top-Down Attention for Image Captioningand Visual Question Answering]( https://arxiv.org/abs/1707.07998)，这是截止2018年2月2日，[COCO比赛排行榜](http://cocodataset.org/#captions-eval)中排第二的论文。
>
> 相关的代代码见[网址]( https://github.com/peteanderson80/bottom-up-attention/)
>
> 本笔记也有参考博主[Jamie](http://blog.csdn.net/sinat_26253653)的[博客](http://blog.csdn.net/sinat_26253653/article/details/78436112)

写在前面：由于我只是关注image captioning的相关内容，所以我会跳过本论文关于视频描述的部分。而且，本人是个**英语渣渣**，翻译不好不要见怪。

## 摘要

Top-down可视化注意力机制已经很普遍的运用在图像描述和视频问题回答（VQA）。通过fined-grained分析和multiple steps of reasoning可以更好地去对图像有更深的理解。在这篇论文中，作者提出了一个bottom-up和top-down机制的结合体。这个结合体能够关注去计算识别目标的level和其它的salient 图像区域。对于要考虑的attention而言，这是一个自然的basis。在作者提供的方法中，bottom-up是基于faster R-CNN的，它抓取出图像感兴趣的区域，并以一个相关的特征向量展示出来。另外，top-down是用来决定每一个特征的权重的。运用这个模型，在MSCOCO test server上取得了state-of-the-art的效果，其CIDEr的得分为117.9。

## 1.介绍

​	在图像描述的过程中，我们经常需要去展示一些fine-grained（颗粒度）可视化进程，或者是multiply steps of  reasoning（为了去产生高质量的输出）。因此，可视化的attention机制经常被运用在图像描述的领域。这些机制促进了对感兴趣区域的描述的性能。

​	在人类的视觉系统中，注意力是由两种因素被集中的。一种是意志力，如被当前的任务（top-down signal）所集中。另一种是自主的、无意识的，如一些没有期待出现的却又是好看的东西（bottom-up signal）所集中。在这篇论文中，采用的是类似的方法，对于那些不可见的、有精准分工的任务文本看成是“top-down”，把可视化的前向attention机制当做为“bottom-up”。

​	大多数运用在常规的visual attention机制是top-down中一种。我们经常使用CNN训练，但我们没有想过这样一个问题：图像中更需要的倾向于关注的物体是怎样取出来的。

![](https://github.com/QuinnChuh/Bottom-up-attention-model/blob/master/imgs/1.PNG)

​	由上图，典型而言，注意力机制运行在CNN特征上得出的是一个同等大小的网格区域，并没有将要关注的内容提取出来。而作者的方法，则能更多的关注图像中感兴趣的物体。

​	在这篇论文中提出的bottom-up和top-down结合的机制。bottom-up机制用来提取在图像中比较salient的区域，并将每个区域用一个pooled convolutional 特征向量来表示。而且，我们是利用faster R-CNN来实现bottom-up机制的，因为faster R-CNN能够展示bottom-up 注意力机制的自然expression。top-down机制使用task-specific context 去预测整副图像应该感兴趣的区域。feature glimpse（不是很懂是什么）是图像中所有区域的平均特征的权重。

​	在评估上，作者分成了两部分。首先，展示一个图像的描述模型，这个模型在图像描述生成的过程中会对感兴趣的区域进行更多的处理。根据实践，作者发现bottom-up attention在图像描述上有很好的效果。而且作者的模型，在MSCOCO test server上取得了state-of-the-art的效果，其CIDEr的得分为117.9，BLEU-4得分为36.9。

## 2.相关工作

​	在以前，每一个attention都在一个或多个CNN的输出层中被连接。为了去预测CNN输出上的每一个空间位置上的权重。然而，决定最佳的感兴趣的图像区域的数目需要在粗细的细节层次做一个无法取舍的选择。另外，在图片上的任意位置都可能产生我们想要的关注的内容，这个也让监测目标变得更加的困难。

​	语义的注意力机制是运用top-down attention到图像中的可视化attributes列表上。Attribute监测可以被看做是一个bottom-up attention进程的输出words，而不是被当做图像的特征向量。然而，这些以及其他的attribute监测都没有获得空间上的信，而仅仅的将监测的attributes当做是a bag of words。作为对比，在作者提供的基于特征的方法中，一个单一的特征向量可以被辨别为几个visual words。例如对于一个名词和一个形容词在监测上提供更强烈的信号。

​	对作者而言，和别人很相似的工作就是，作者也提出了利用attention去关注salient image regions。Jin et. al 曾经使用选择性的搜索去辨别 salient image regions，这被一个分类器所过滤，然后重置了CNN编码器的大小，使得他作为一个带有attention机制的图像描述模型的输入。同样的，Pedersoil et .al 提出的DenseCap模型是使用空间transformer networks去产生图像的各种特征，产生的特征可以使用一个基于bi-linear pairwise interaction的传统attention模型来进行处理。这个空间的transformer networks允许端对端的区域提取坐标的训练。作为对比，在作者的方法中，作者使用预训练来解决区域监督问题。

## 3.方法

### 3.1 Bottom-Up attention模型

​	图像中空间的特征V的定义是通用的。然而，在这个工作中，利用faster R-CNN来实现bottom-up attention模型。faster R-CNN是一个目标检测模型，它设计是为了去辨别目标的实例属于某个确定的classes，并用bounding boxes来定位他们。其它的区域区域提取网络也能够被训练为一个attention机制。

​	faster R-CNN检测目标分为两个阶段。在第一个阶段，使用RPN(Region Proposal Network)来预测目标proposals。在CNN的中级水平上一个小的网络下滑到特征。每个空间定位网络预测一个不可知的对象的得分和在多尺度和纵横比上细化bounding box。使用结合了IOU阈值的非最大suppression，top box proposals被选为第二阶段的输入。在第二阶段，对于每一个top box proposal，感兴趣的区域pooling被使用去提取一休哥小的特征向量（例如14 x 14）。这些特征maps然后会被batched在一起作为CNN最后一层的输入。模型的最后输出是由classes标签softmax分布和每一个box proposal的class-specific bounding box refinements组成。

​	在作者的模型中，作者使用了faster R-CNN和ResNet-101 CNN相连。为了产生图像特征V的输出集合并运用到图像描述上，作者使用IOU阈值把最后的模型输出和perform每一目标类的非最大suppression。然后，作者选择所有的区域，这些区域是他们的任意类的检测可能性都超出一个confidence阈值。对于每一个选择区域的i，vi被定义为从这个区域中做均值池化后的卷积特征。

​	为了预训练bottom-up模型，作者首先在ImageNet上用ResNet-101去预训练faster R-CNN模型。然后再在visual genome数据上训练。为了一个好的特征提取，作者对预测的attribute classes增加了一个额外的训练输出。为了预测区域i的attribute，作者将均值池化卷积特征vi和一个ground-truth object class的learned embedding拼接在一起。然后将他们送到一个额外的输出层。这个输出层是每个attribute class加“no attribute” class的softmax值。

​	原始的faster R-CNN多任务loss函数包含了4个成分，为RPN和最终的目标class proposals分别定义了分类以及bounding box regression输出。作者保留了这些成分，并增加了一个额外的multi-class loss成分去训练attribute predictor。

​	![](https://github.com/QuinnChuh/Bottom-up-attention-model/blob/master/imgs/2.PNG)

​	上图是这个模型的输出，每一个bounding box都被一个attribute class标签。

### 3.2 Captioning 模型

​	对于captioning部分，也是本论文的重点了，这里作者引入了两个LSTM，top-down attention LSTM和language LSTM，其captioning模型的整个结构图如下：

![](https://github.com/QuinnChuh/Bottom-up-attention-model/blob/master/imgs/3.PNG)

​	而且两个LSTM是交叉的，都有运用到彼此的输出。

#### 3.2.1 Top-Down attention LSTM

​	在上面提到了，bottom-up模型的输出被定位为特征集V。在Top-Down attention LSTM模型中包含有三个输入，每个输入的定义如下：

- 前一时刻的language LSTM模型的隐藏状态ht-1
- k个由bottom-up输出的特征vi经过mean-pooled后的特征v~.
- 还有之前生成的词的编码（We是embedding矩阵，TTt是t时刻输入单词的ont-hot编码）

​        对于*k*个image feature vi的attention权重αi,j，是由top-down attention LSTM在每一个时刻利用自身的hidden state h1t产生的：

![](https://github.com/QuinnChuh/Bottom-up-attention-model/blob/master/imgs/4.PNG)

#### 3.2.2 Language LSTM

​	对于这一模块，其输入由下面两部分构成：

- attention feature的加权和v^t
- 当前时刻的Top-Down attention LSTM的输出ht

​        而对于整个模型的输出，则是t时刻输出的任一单词的概率分布p(yt|y1:t−1)=softmax(Wph2t+bp)，其中(y1,...,yT)为单词的序列。

​	而整个句子的概率分布是每一个单词的连乘。

![](https://github.com/QuinnChuh/Bottom-up-attention-model/blob/master/imgs/5.PNG)

#### 3.2.3 训练目标

文章首先使用的是最小化cross entropy 

![](https://github.com/QuinnChuh/Bottom-up-attention-model/blob/master/imgs/6.PNG)

来进行训练的，其中y∗1:T是ground-truth caption。

另外文章还用到了[SCST](https://arxiv.org/abs/1612.00563)中的强化学习方法来对CIDEr分数进行优化：

![](https://github.com/QuinnChuh/Bottom-up-attention-model/blob/master/imgs/7.PNG)

## 4.结果分析

贴图：

![](https://github.com/QuinnChuh/Bottom-up-attention-model/blob/master/imgs/8.PNG)

参数说明：

​	Resnet是文章选取的一个baseline模型，用来代替bottom-up attention机制。也就是说这个baseline模型只有top-down的attention机制。

​	ATT是指论文Self-critical sequence training for image captioning，2017所用的模型。

​	输出也分两类进行说明，一个使用了SCST，一个没有。

下面是与其他论文算法的对比：

![](https://github.com/QuinnChuh/Bottom-up-attention-model/blob/master/imgs/9.PNG)

其它算法说明：

​	NIC：是google的 Show and tell: A neural image caption generator，2015

​	MSR Captivator：From captions to visual concepts and back，2015

​	M-RNN： Deep captioning with multimodal recurrent neural networks，2015

​	LRCN：Long-term recurrent convolutional networks for visual recognition and description，2015

​	Hard-Attention： Show, attend and tell:Neural image caption generation with visual attention，2015

​	ATT-FCN： Image captioning with semantic attention，2016

​	Review Net：Review networks for caption generation，2016

​	MSM：Boosting image captioning with attributes，2016

​	Adaptive： Knowing when to look: Adaptive attention via a visual sentinel for image captioning，2017

​	PG-SPIDEr-TAG： Optimization of image description metrics using policy gradient methods，2016

​	SCST:Att2all：Self-critical sequence training for image captioning，2017