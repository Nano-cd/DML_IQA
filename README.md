*DML_IQA*
===============
**This is official repository of article ：“Improving IQA Performance Based on Deep Mutual Learning” . ICIP2022.
DOI:10.1109/ICIP46576.2022.9897784
https://ieeexplore.ieee.org/abstract/document/9897784**
===============

## DML-IQA

![image](https://user-images.githubusercontent.com/72659127/232369654-b17bc63f-7309-4510-9f3e-2639b6033f98.png)

## Install

```js
install requeried tools: pytoch numpy, ect
```

## Usage
```
$ readme README.md
```

```js

// Checks readme in current working directory

You just need 
1)define your own dataset and network.(dataset  and   model_hub)
2)define your own parse_config
3)run baseline.py to get the baseline model.
4)run train.py to get the DML-IQA model.
5)change the parse_config (train to test) and choose the model you saved.


Among them, checkpoint, data, logs, results, and runs are respectively used to save the trained model,
the dataset and partitions required for training, training logs, and training results. It needs to be 
adjusted according to the path in your computer, etc. It is worth noting that MB_ BL uses the MBCNN-IQA 
method, which is not reported in this article. Interested readers can explore it on their own.
```

## Contributing

In this paper, we diverted the attention from network improvement to learning strategy change and carried out a pioneering attempt to improve IQA performance based on deep mutual learning. Through extensive experiments, we draw the following conclusions. First, the proposed DML-IQA improves the baseline’s performance in terms of both effectiveness and generalization. Notably, the improvement is more apparent when the training set size or network is small. Second, the proposed DML-IQA is flexible to adapt to diverse CNNs and is conducive to tackling the practical IQA issues in the laboratory and wild. These findings give us new inspirations for the follow-up works: 1) The DML-IQA can be further incorporated with existing full-reference/reduced-reference CNN-based IQA methods, to obtain better 
performance by changing the current backbones and combining the DML strategy with previous methods. 2) The DML-IQA can be further extended to a semi-supervised IQA way to solve the problems that lack the training samples.

## Citation
```
@inproceedings{yue2022improving,
  title={Improving IQA Performance Based on Deep Mutual Learning},
  author={Yue, Guanghui and Cheng, Di and Wu, Honglv and Jiang, Qiuping and Wang, Tianfu},
  booktitle={2022 IEEE International Conference on Image Processing (ICIP)},
  pages={2182--2186},
  year={2022},
  organization={IEEE}
}
```



