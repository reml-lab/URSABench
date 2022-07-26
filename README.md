# URSABench
This repository contains the PyTorch implementation for the paper "URSABench: A System for Comprehensive Benchmarking of Bayesian Deep Neural Network Models and Inference methods" by [Meet P. Vadera](https://meetvadera.github.io), [Jinyang Li](https://scholar.google.com/citations?hl=en&user=VbeL3UUAAAAJ), [Adam D. Cobb](https://adamcobb.github.io/), [Brian Jalaian](https://brianjalaian.netlify.app/), [Tarek Abdelzaher](http://abdelzaher.cs.illinois.edu/) and [Benjamin M. Marlin](https://people.cs.umass.edu/~marlin).

This paper will be presented at [MLSys '22](https://mlsys.org/). An initial version of this paper was presented at ICML '20 Workshop on [Uncertainty and Robustness in Deep Learning](https://sites.google.com/view/udlworkshop2020/home). More updates to this repo will follow soon!

## Code references:

* Model implementations:
  - PreResNet: https://github.com/bearpaw/pytorch-classification
  - WideResNet: https://github.com/meliketoy/wide-resnet.pytorch

* The included inference schemes have been adapted from the following repos:
  - SWA https://github.com/timgaripov/swa/
  - SWAG https://github.com/wjmaddox/swa_gaussian

* For HMC, we use https://github.com/AdamCobb/hamiltorch.
* Some metrics incorporate code from https://github.com/bayesgroup/pytorch-ensembles

Please cite our work if you find this approach useful in your research:
```bibtex
@inproceedings{MLSYS2022_3ef81541,
	author = {Vadera, Meet P. and Li, Jinyang and Cobb, Adam and Jalaian, Brian and Abdelzaher, Tarek and Marlin, Benjamin},
	booktitle = {Proceedings of Machine Learning and Systems},
	editor = {D. Marculescu and Y. Chi and C. Wu},
	pages = {217--237},
	title = {URSABench: A System for Comprehensive Benchmarking of Bayesian Deep Neural Network Models and Inference methods},
	url = {https://proceedings.mlsys.org/paper/2022/file/3ef815416f775098fe977004015c6193-Paper.pdf},
	volume = {4},
	year = {2022},
	bdsk-url-1 = {https://proceedings.mlsys.org/paper/2022/file/3ef815416f775098fe977004015c6193-Paper.pdf}}
```

## Acknowledgements

Research reported in this paper was sponsored in part by the CCDC Army Research Laboratory under Cooperative Agreement W911NF-17-2-0196 (ARL IoBT CRA). The views and conclusions contained in this document are those of the authors and should not be interpreted as representing the official policies, either expressed or implied, of the Army Research Laboratory or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for Government purposes notwithstanding any copyright notation herein.
