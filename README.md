# Corruption Invariant Learning for Cross-Modal ReID
 The official repository for [Benchmarks for Corruption Invariant Person Re-identification](https://arxiv.org/abs/2111.00880) (NeurIPS 2021 Track on Datasets and Benchmarks) on cross-modal person ReID datasets, including SYSU-MM01, RegDB-C.

# Quick Start
### 1.Train
Train a CIL model on SYSU-MM01,

```
sh z_train.sh
```

### 2.Test
Evaluate the CIL model on SYSU-MM01,

```
sh z_test.sh
```
(Note: codebase from [AGW](https://github.com/mangye16/Cross-Modal-Re-ID-baseline))

## Citation

Kindly include a reference to this paper in your publications if it helps your research:
```
@misc{chen2021benchmarks,
    title={Benchmarks for Corruption Invariant Person Re-identification},
    author={Minghui Chen and Zhiqiang Wang and Feng Zheng},
    year={2021},
    eprint={2111.00880},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
