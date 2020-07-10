# URSABench
This repository contains the PyTorch implementation for the paper "URSABench: Comprehensive Benchmarking of Approximate Bayesian Inference Methods for Deep Neural Networks" by [Meet P. Vadera](https://meetvadera.github.io), [Adam D. Cobb](https://adamcobb.github.io/), [Brian Jalaian](https://brianjalaian.netlify.app/), and [Benjamin M. Marlin](https://people.cs.umass.edu/~marlin).

This paper will be presented at the ICML '20 Workshop on [Uncertainty and Robustness in Deep Learning](https://sites.google.com/view/udlworkshop2020/home). More updates to this repo will follow soon!

## Code references:

* Model implementations:
  - PreResNet: https://github.com/bearpaw/pytorch-classification
  - WideResNet: https://github.com/meliketoy/wide-resnet.pytorch

* The included inference schemes have been adapted from the following repos:
  - SWA https://github.com/timgaripov/swa/
  - SWAG https://github.com/wjmaddox/swa_gaussian

* For HMC, we use https://github.com/AdamCobb/hamiltorch.
* Some metrics incorporate code from https://github.com/bayesgroup/pytorch-ensembles

## Acknowledgements

Research reported in this paper was sponsored in part by the CCDC Army Research Laboratory under Cooperative Agreement W911NF-17-2-0196 (ARL IoBT CRA). The views and conclusions contained in this document are those of the authors and should not be interpreted as representing the official policies, either expressed or implied, of the Army Research Laboratory or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for Government purposes notwithstanding any copyright notation herein.
