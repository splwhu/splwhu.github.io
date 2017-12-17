---
layout: post
title:  "Sparse Bayesian learning for image rectification with transform invariant low-rank textures!"
date:   2017-12-17 14:55:24 +0800
categories: jekyll update
---

## Sparse Bayesian learning for image rectification with transform invariant low-rank textures ##


###Abstract###
 Comparing to the low-level local features, transform invariant low-rank textures (TILT) can in some sense
globally rectify a large class of low-rank textures in 2D images, and thus more accurate and robust. However,
the existing algorithms based on the alternating direction method (ADM) and the linearized alternating
direction method with adaptive penalty (LADMAP), suffer from the weak robustness and the
local minima, especially with plenty of corruptions and occlusions. In this paper, instead of exploiting
optimization methods, we propose to build a hierarchical Bayesian model to TILT and then a variational
method is implemented for Bayesian inference. Instead of point estimation, the proposed Bayesian approach
introduces the uncertainty of the parameters, which has been proven to have much less local
minima. Experimental results on both synthetic and real data indicate that our new algorithm outperforms
the existing algorithms especially for the case with corruptions and occlusions


### Introduction ###
We focus on the third situation when the userspecified patches are with too many corruptions. Although TILT is designed to be robust to corruptions and occlusions, there is an assumption essentially necessary that the amount of corruptions and occlusions can not be enormous. This is due to the fact that the ADM and LADMAP are both trying to solve the nuclear and 1-norm constrained optimization problem, which has been proven to have lots of local minima when with too many noises and perturbations. Moreover, the aforementioned optimization algorithms need to set the trade-off parameters in advance, which are not known a priori. What’s worse, naive fixed values will lose uncertainty in the parameters, which induces uncertainty in predictions. These may drive the optimization to a mistaken end under this kind of complex situation.
![](https://i.imgur.com/om3yKwb.png)



![](https://i.imgur.com/tM3UoUh.png)


### Experiments ###

![](https://i.imgur.com/yh095G2.png)

### Conclusion ###

We propose a
robust algorithm for better solving TILT problem. Hierarchical Bayesian modelings are exploited to our model to impose lowrankness inducing prior and sparsity-inducing prior on corresponding entries, variational Bayesian inference is implemented to compute the estimations of related parameters. Besides less local
minima, nonparameter Bayesian approach introduces the uncertainty in the parameters to make our new algorithm can handle more complex situations in natural sceneries, especially the cases with plenty of corruptions and occlusions. Experimental results show that our new algorithm significantly outperforms the existing
methods, both artificial synthetic images and real images are
utilized in experiments.


### Reference ###
[Sparse Bayesian learning for image rectification with transform invariant low-rank textures](https://www.sciencedirect.com/science/article/pii/S0165168417300701)
