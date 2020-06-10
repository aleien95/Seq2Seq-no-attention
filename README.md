# 项目介绍
本项目是一个Seq2seq的英译汉翻译模型，整个模型很简单，适合刚刚入门pytorch和nlp的同学学习使用。必要的地方我都加了注释，模型图解参见[我的网站](http://www.big-what-ever.cn/index.php/2020/06/10/%e5%9b%be%e8%a7%a3seq2seq-attention%e6%a8%a1%e5%9e%8b/)，欢迎star！

# 运行环境
pytorch 1.0.1

# 运行代码：
python seq2seq-by-myself.py

# Belu Score
P1:0.8408
P2:0.7138
P3:0.6302
P4:0.5625
BP:1
BERT:0.6792

# 本文项目流程
一、获取cmn的中英文数据结构化处理
二、搭建Seq2seq
三、训练网络、验证、随机抽取查看结果
