# Subtractive Aggregation for Attributed Network Anomaly Detection (AAGNN)

## 1.Introduction
This repository contains code for paper "Subtractive Aggregation for Attributed Network Anomaly Detection" (CIKM'21).

## 2. Usage
### Requirements:
+ pytorch==1.10.0
+ scikit-learn
+ networkx
+ scipy
+ pandas

### Datasets:
Users can create datasets with injected anomalies by themselves. For details (e.g., code), users can refer to [this paper](https://arxiv.org/abs/2206.10071).

### Examples:
+ python main.py --dataset=BlogCatalog_anomaly --model=Atten_Aggregate --seed=1
+ python main.py --dataset=BlogCatalog_anomaly --seed=1

## 3. Citation
Please kindly cite the paper if you use the code or any resources in this repo:
```bib
@inproceedings{zhou2021subtractive,
  title={Subtractive aggregation for attributed network anomaly detection},
  author={Zhou, Shuang and Tan, Qiaoyu and Xu, Zhiming and Huang, Xiao and Chung, Fu-Lai},
  booktitle={Proceedings of the 30th ACM International Conference on Information \& Knowledge Management},
  pages={3672--3676},
  year={2021}
}
```


