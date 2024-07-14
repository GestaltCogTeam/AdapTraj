# AdapTraj: A Multi-Source Domain Generalization Framework for Multi-Agent Trajectory Prediction

A pytorch implementation for the paper: [AdapTraj: A Multi-Source Domain Generalization Framework for Multi-Agent Trajectory Prediction](https://arxiv.org/abs/2312.14394)

## Introduction 
Multi-agent trajectory prediction, as a critical task in modeling complex interactions of objects in dynamic systems, has attracted significant research attention in recent years. Despite the promising advances, existing studies all follow the assumption that data distribution observed during model learning matches that encountered in real-world deployments. However, this assumption often does not hold in practice, as inherent distribution shifts might exist in the mobility patterns for deployment environments, thus leading to poor domain generalization and performance degradation. Consequently, it is appealing to leverage trajectories from multiple source domains to mitigate such discrepancies for multi-agent trajectory prediction task. However, the development of multi-source domain generalization in this task presents two notable issues: (1) negative transfer; (2) inadequate modeling for external factors. To address these issues, we propose a new causal formulation to explicitly model four types of features: domain-invariant and domain-specific features for both the focal agent and neighboring agents. Building upon the new formulation, we propose AdapTraj, a multi-source domain generalization framework specifically tailored for multi-agent trajectory prediction. AdapTraj serves as a plug-and-play module that is adaptable to a variety of models. Extensive experiments on four datasets with different domains demonstrate that AdapTraj consistently outperforms other baselines by a substantial margin.

## Getting Started
 
```
python train.py
```

