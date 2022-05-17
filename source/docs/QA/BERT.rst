BERT
----

**目前需要一个模型来做搜索业务，用哪个好？BERT GPT 还是什么**

   bert或者gpt都可以，更推荐bert及其衍生模型。gpt属于自回归模型，更加适合生成任务。实际工作中，建议根据需求和硬件环境通过实测选择合适模型。

**谷歌翻译是不是用的就是bert模型？**

   后续会专门发布视频进行讲解。

**如果用BERT做Question Answering, 最上面的QA
层（softmax）和下面是全连接吗？请问有什么paper或者有关这方面的参考资料吗？**

   建议参考
   `原文链接 <https://qa.fastforwardlabs.com/pytorch/hugging%20face/wikipedia/bert/transformers/2020/05/19/Getting_Started_with_QA.html>`__
