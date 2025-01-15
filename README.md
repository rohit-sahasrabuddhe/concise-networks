# Concise network models of path data
This repository contains the implementation of the framework described in paper:
Sahasrabuddhe, R., Lambiotte, R., and Rosvall, M., 2025. Concise network models of memory dynamics reveal explainable patterns in path data. arXiv preprint [arXiv:2501.08302](https://arxiv.org/abs/2501.08302).

## Requirements

The code was written in Python 3.9 and uses standard libraries such as `numpy`, `pandas`, and `scipy`. It uses the `sklearn` k-means clustering implementation and `joblib` for parallelising.

## General usage

The `airports_example.ipynb` notebook contains a working example.

## Credits

The Convex Non-negative Matrix Factorisation at the core of the framework is due to
Ding, C.H., Li, T. and Jordan, M.I., 2008. Convex and semi-nonnegative matrix factorizations. IEEE transactions on pattern analysis and machine intelligence, 32(1), pp.45-55.
