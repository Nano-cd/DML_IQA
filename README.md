# DML_IQA
This is official repository of article ：“Improving IQA Performance Based on Deep Mutual Learning” . ICIP2022.
DOI:10.1109/ICIP46576.2022.9897784
https://ieeexplore.ieee.org/abstract/document/9897784

In this paper, we propose a novel solution, termed DML-IQA, for the image quality assessment (IQA) tasks. DML-IQA holds a dual-branch network architecture and builds the IQA model through a deep mutual learning (DML) strategy. Specifically, the two branches extract stable feature representations by feeding different transformed images into the classical CNNs. The DML strategy first calculates the prediction loss of each branch and the consistency loss across two branches, followed by updating the network iteratively to converge. Overall, DML-IQA has the following advantages: 1) It is flexible to adapt to diverse backbones for tackling the IQA issues in both the laboratory and wild; 2) It improves the baseline’s performance by approximately 1%~2%, especially performs well in the case of small samples. Extensive experiments on four public datasets show that the proposed DML-IQA can handle the IQA tasks with considerable effectiveness and generalization.

![image](https://user-images.githubusercontent.com/72659127/232369654-b17bc63f-7309-4510-9f3e-2639b6033f98.png)
![image](https://user-images.githubusercontent.com/72659127/232369733-98beb19a-d549-4ca0-bbfa-c6ade9af3054.png)
![image](https://user-images.githubusercontent.com/72659127/232369795-73be7e63-97a7-43d8-9a71-6ff16bc6a686.png)


for train DML-IQA: 
You just need 1)define your own dataset and network.(dataset  and   model_hub)
\2)define your own parse_config
\3)train baseline.py to get the baseline model.
\4)train train.py to get the DML-IQA model.
\5)change the parse_config (train to test) and choose the model you saved.


In this paper, we diverted the attention from network improvement to learning strategy change and carried out a pioneering attempt to improve IQA performance based on deep mutual learning. Through extensive experiments, we draw the following conclusions. First, the proposed DML-IQA improves the baseline’s performance in terms of both effectiveness and generalization. Notably, the improvement is more apparent when the training set size or network is small. Second, the proposed DML-IQA is flexible to adapt to diverse CNNs and is conducive to tackling the practical IQA issues in the laboratory and wild. These findings give us new inspirations for the follow-up works: 1) The DML-IQA can be further incorporated with existing full-reference/reduced-reference CNN-based IQA methods, to obtain better 
performance by changing the current backbones and combining the DML strategy with previous methods. 2) The DML-IQA can be further extended to a semi-supervised IQA way to solve the problems that lack the training samples.
