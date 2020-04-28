# Dynamic AdveRsarial Training (Dynamic/DART) 
Code for ICML2019 Paper "On the Convergence and Robustness of Adversarial Training"

One **Important Message** in this paper: To ensure **better robustness**, it is essential to use **adversarial examples with better convergence quality at the later stages of training**. Yet at the **early stages**, high convergence quality adversarial examples are **not necessary** and may even lead to poor robustness. 

Convergence quality is measured by First-Order Stationary Condition (FOSC)
<img src="https://github.com/YisenWang/dynamic_adv_training/blob/master/fosc.png" width="50%" height="50%">

## Requirements
- Python 3.5.2, 
- Tensorflow 1.10.1 
- Keras 2.2.2

## Usage

Simply run the code by: python3 train_models.py

## Citing this work
If you use this code in your work, please cite the accompanying paper:

```
@inproceedings{wang2019dynamic,
  title={On the Convergence and Robustness of Adversarial Training},
  author={Wang, Yisen and Ma, Xingjun and Bailey, James and Yi, Jinfeng and Zhou, Bowen and Gu, Quanquan},
  booktitle={International Conference on Machine Learning},
  year={2019}
}
```
