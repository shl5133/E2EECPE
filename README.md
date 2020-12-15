# E2EECPE

This repository contains the code and dataset of the following paper:

[Haolin Song](https://shl5133.github.io), [Chen Zhang](https://genezc.github.io), [Qiuchi Li](https://qiuchili.github.io), [Dawei Song](http://cs.bit.edu.cn/szdw/jsml/js/sdw/index.htm). **End-to-end Emotion-Cause Pair Extraction via Learning to Link**. arXiv preprint arxiv:2002.10710(2020). [[paper link](https://arxiv.org/abs/2002.10710)]

## Model

An overview of our model is given below:

![model](/images/model.PNG)
## Dataset
* The [Dataset](/datasets/sina) we are using is a publicly available dataset released by [(Xia and Ding 2019)](https://www.aclweb.org/anthology/P19-1096.pdf) [[dataset link](https://github.com/NUSTM/ECPE/tree/master/data_combine)]
* The pre-trained word vectors can be found [here](https://github.com/NUSTM/ECPE/blob/master/data_combine/w2v_200.txt).
## Requirements

* Python 3.6
* PyTorch 1.2.0

## Usage
#### 1.Clone or download this repository
```bash
git@github.com:shl5133/E2EECPE.git
```
#### 2.Download pre-trained word vectors [w2v_200.txt](https://github.com/NUSTM/ECPE/blob/master/data_combine/w2v_200.txt) and place it in the root path
```bash
python train.py
```
#### 3.Run our model
```bash
python train.py
```

## Citation

If you use the code in your paper, please kindly star this repo and cite our paper.

```bibtex
@article{song2020end,
  title={End-to-end Emotion-Cause Pair Extraction via Learning to Link},
  author={Song, Haolin and Zhang, Chen and Li, Qiuchi and Song, Dawei},
  journal={arXiv preprint arXiv:2002.10710},
  year={2020}
}
```
