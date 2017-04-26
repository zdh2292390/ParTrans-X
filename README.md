# ParTrans-X
ParTrans-X is an efficient parallel lib for translating embedding algorithms.

The main adavantage of our lib is that our algorithms are totally lock-free multithreading, which enables the algorithms to be highly efficient, and we give some theoretical guarantees in our paper.

Up to now, we implemented the parallel version of TransE, AdagradTransE, and TransH. Named ParTransE, ParAdagradTransE, ParTransH respectively.

Also, we implemented the parallel version of the link prediction program for each of them, named ParTest_TransE and ParTest_TransH. (TransE and AdagradTransE use the same test program).

In addition, we provide all the shell scripts for parameter tuning.

Evaluation Results
==========

We list the link prediction result and runtime of various methods implemented by ourselves in dateset FB15k and WN18.

FB15k,nepoch=1000(100 for AdagradTransE), threads=20

| Model      |    MeanRank(Raw) |   MeanRank(Filter)   |	Hit@10(Raw)	| Hit@10(Filter)| Time(s)| Speedup Ratio|
| :-------- | --------:| :------: | :------: |:------: |:------: |:------: |
| TransE |    184 | 73 |  44.5 | 60.7| 4658 | -|
| ParTransE |    185 | 69 |  45.3 | 62.3| 496 | 9|
| ParAdagradTransE |    186 | 70 |  44.9 | 61.9| 42 |111|
| TransH  |    183 |  60 |  46.6 | 65.5 | 6066 | -|
| ParTransH  |    183 | 60 |  46.8 |  65.7 | 474 | 13 |

WN18,nepoch=1000(100 for AdagradTransE), threads=20

| Model      |    MeanRank(Raw) |   MeanRank(Filter)   |	Hit@10(Raw)	| Hit@10(Filter)| Time(s)| Speedup Ratio|
| :-------- | --------:| :------: | :------: |:------: |:------: |:------: |
| TransE |    214 | 203 |  58.2 | 65.9| 473 | -|
| ParTransE |    217 | 206 |  55.7 | 63.1| 54 | 9 |
| ParAdagradTransE |    219 | 208 | 67.7 | 76.2| 17 | 28|
| TransH  |    227 |  216 |  66.5 | 75.9 | 637 | -|
| ParTransH  |    215 |  203 |  66.8 | 76.6 | 134 | 4.8 |

Compile and Use
==========

Since we use openmp for multithreading, your g++ should contain it.

compile eg:
```bash
g++ -fopenmp ParTransE.cpp -o ParTransE

g++ -fopenmp ParTest_TransE.cpp -o ParTest_TransE
```
Use the script for parameter tuning is quite easy:
```bash
nohup ./train_test_TransE > logs 2>&1 &
```
More specific details can be found in the shell scripts, you can also make customized scripts.

Notice:

You should set $coreNum in the shell script less than the max number of your cores in your machine.

Cite
==========

If you use the code, please kindly cite the following paper:

Denghui Zhang, Manling Li, Yantao Jia, Yuanzhuo Wang. Efficient Parallel Translating Embedding For Knowledge Graphs.arXiv preprint arXiv:1703.10316, 2017.[[pdf]](https://arxiv.org/pdf/1703.10316.pdf)