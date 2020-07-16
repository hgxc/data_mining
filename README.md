# data_mining
NLP文本分类：词处理器+VSM+tf-idf+简单全连接神经网络

preprocessing 用于对dataset中的文本进行处理，进行去停用词、标点符号、数字等操作，得到纯净的文本并存储到sdata中

train则用来进行构建VSM，tf-idf,以及使用简单三层神经网络进行文本分类任务
