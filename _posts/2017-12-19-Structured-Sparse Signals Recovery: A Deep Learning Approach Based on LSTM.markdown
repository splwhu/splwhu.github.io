---
layout: post
title:  "Structured-Sparse Signals Recovery: A Deep Learning Approach Based on LSTM"
date:   2017-12-19 14:55:24 +0800
categories: jekyll update
---

## Structured-Sparse Signals Recovery:                  A Deep Learning Approach Based on LSTM ##



### Introduction

Compressive sensing nowadays is one of the hottest research topics in signal processing field, which also plays a significant role in signal sampling and recovering. 

In the conventional sense, only sparse prior on the property of signals is adopted to guarantee the exact recovery. In the general CS framework, the canonical form of CS could be written as follows:
$$
\mathbf{y} = \mathbf{Ax} + \mathbf{e}
$$

where \\(\mathbf{y}\in \mathcal{R}^{M}\\) is the measurement matrix, \\(\mathbf{A}\in \mathcal{R}^{M\times N}\\) is a random sensing matrix with \\(M\ll N\\) satisfying the so-called [RIP](https://www.sciencedirect.com/science/article/pii/S1631073X08000964), \\(\mathbf{x}\in \mathcal{R}^{N}\\) is the original sparse signal needed to be recovered with no more than \\(K\\)\\((K<M)\\) nonzero elements and \\(\mathbf{e}\\) is the error term consists of the possible noise and perturbations.

Besides the sparsity, however, the structures which are used to describe the correlation of nonzero elements could also be utilized to improve the performance of sparse signals recovery algorithms, such as block-sparse, tree-structure, uniform-sparse and so forth. There are numerous algorithms have been proposed specific to the certain structure, for instance,  [ block-CoSaMP ](http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=5437428) and [ block-sparse Bayesian learning ](http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6415293) for block-sparse signals, [ TSW-CS ](http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=4907073) for tree-structure signals, etc. 

In spite of aforementioned model-driven methods achieve excellent performance, they still have some insurmountable disadvantages: (1) the highest level that model-driven methods could attain is no more than the comprehension of researchers; (2) model-driven methods are specific to the certain structure and without the universality. 

In consideration of the development of deep learning and big data analysis in recent few years, we design a data-driven method which could recover structured-sparse signals with any structure patterns. The main contribution of our work is that we develop a novel algorithm dealing with the problem of structured-sparse signals recovery based on recurrent neural networks. It should be noted especially that our proposed methoed is not only specific to the certain structure but any structure patterns. Extensive experimental results demonstrate the superiority of our proposed algorithm over any other existing model-driven algorithms.



### Why and How the LSTM works

#### Why the LSTM works

Along with their great success in natural language processing (NLP), recurrent neural networks (RNNs) have been a significant kind of network in processing the sequential signal. However, the existance of the problem that RNNs are difficult to train and can not handle long-term dependencies over time because of the exploding and vanishing gradient problem that plagues all RNNs in training seriously hindered the development of RNNs in the early phase. The long short-term momery (LSTM), which could recusively maintain the state that is a compact summarization of all the important information from the past for a long period of time, is designed to avoid the long-term dependency problem, allowing larger and deeper networks to be created.

There is an abundance of work attempts to interpret internal mechanisms of the LSTM. Basically, the LSTM can in principal use its memory cells to remember long-range information and keep track of various attributes of information it is currently processing. Let us take a look inside the standard LSTM block below, which is the most general and serves in hidden layers of RNNs in many work. The input gate can allow the input information to alter the memory state or block it. If the new input is relevant for the internal state, the information of the input will be saved in the activation of the memory state. The output gate can allow the memory state to be revealed or prevent its effect on the next neuron. Thus the information stored in the memory state is readable when the output gate is active. The forget gate can update the momery state by erasing or retaining the cell's previous state. Since the peephole mechanism does not have great help to improve the recovery performance, we have removed these peephole connections to shorten the training time in this work. Aforementioned powerful multiplicative interactions enable the LSTM to capture richer contextual information as it goes along the sequence.


![Block diagram of the LSTM](https://github.com/yuanqw/yuanqw.github.io/blob/master/image/lvcc/lstm.png?raw=true)
### How the LSTM works

In this paper, we adopt the framework of the greedy algorithm -- [OMP](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.55.1254&rep=rep1&type=pdf). Instead of searching indices of nonzero elements by solving a maximization problem in OMP algorithm, we replace this step with learning approaches. Our proposed algorithm, the flow diagram of which is shown in below, could be split into five steps : 

![algorithm](https://github.com/yuanqw/yuanqw.github.io/blob/master/image/lvcc/algorithm.png?raw=true)

* initialize the residual  \\(\mathbf{r}\\) by the measurement matrix \\(\mathbf{y}\\) and regard it as the input to the LSTM; 
* consider the output of the LSTM \\(\mathbf{s}\\) as the input of softmax layer, which is used to calculate the probability of each element of signals be nonzero;
* select  the maxinum probability in these probabilities and employ its corresponding index as the position of a certain nonzero element; 
* estimate the value of this nonzero element via solving the following least square (LS) problem: 
$$
\hat{\mathbf{x}} = \arg\min_{\mathbf{x}}|| \mathbf{y}-\mathbf{A}^{\Omega} \mathbf{x}||\_2^2
$$





where \\(\mathbf{A}^{\Omega}\\) is a matrix that includes only those columns of \\(\mathbf{A}\\) that are members of the support of \\(\mathbf{x}\\); 

* calculate the new residual by using \\(\hat{\mathbf{x}}\\) in step 4 

$$
\mathbf{r} = \mathbf{y}-\mathbf{A}^{\Omega} \hat{\mathbf{x}}
$$

this new residual will serve as the input to the LSTM in the next iteration. Of particular note is that the     output of the LSTM will be inputted into the next LSTM as well. 

## Training Details

### Training Samples Generation

Now we state concrete steps for generating training samples: (1) randomly give a structured-sparse signal \\(\mathbf{x}\\) with \\(K\\) nonzero elements and calculate \\(\mathbf{y}\\) using **(1)**; (2) find the element that has the maximum value in \\(\mathbf{x}\\) and set it to zero, then assume that the index of this element is \\({p\_0}\\), thus we get a new structured-sparse signal with \\(K-1\\) nonzero elements; (3) calculate the residual vector by
$$
\mathbf{r}=\mathbf{y}-\mathbf{a}_{p\_0}x(p\_0)
$$
where \\(\mathbf{a}\_{p\_0}\\) is the \\(p\_0\\)-th column of the sensing matrix \\(\mathbf{A}\\) and \\(x(p\_0)\\) is the \\(p\_0\\)-th element of \\(\mathbf{x}\\). Notably, the generation of the residual vector is because of not having the other \\(K-1\\) nonzero elements of \\(\mathbf{x}\\), in which the second largest value in these \\(K-1\\) nonzero elements principally contributes to the residual vector \\(\mathbf{r}\\) in **(4)**; (4) assume that the index of the second largest value of \\(\mathbf{x}\\) is \\(p\_1\\), then define a one-hot vector \\(\mathbf{h}\\) at \\(p\_1\\)-th entry. Finally, a training sample pair \\((\mathbf{r},\mathbf{h})\\) is obtained. Next, we set the second largest value of \\(\mathbf{x}\\) to zero and repeat the above procedures until the vector \\(\mathbf{x}\\) does not have any nonzero element. 

### Training Regimes

The main purpose of training is to find a set of weights and bias values so that computed outputs closely match the known outputs of a collection of training data. The consequent neural network can make predictions on new data once a set of fine weights and bias values have been found.

After employing the cross-entropy as the loss function over the training samples in this work, what we faced is the optimization problem as follows:

$$
\mathcal{J}(\Lambda)=\min\_{\Lambda}\left(\sum\_{n=1}^{nB}\sum_{i=1}^{B\text{size}}\sum\_{j=1}^{N}\mathcal{J}\_{n,i,j}(\Lambda)\right)
$$
$$
\mathcal{J}\_{n,i,j}(\Lambda)=-\mathbf{h}\_{n,i}(j)\log(\mathbf{s}\_{n,i}(j))
$$

where \\(\mathbf{s}\\) is the output of the softmax layer, \\(nB\\) is the number of structured-sparse signals for generating training samples, \\(B\text{size}\\) is the number of sample pairs generated from the same given structured-sparse signal, \\(N\\) is the length of structured-sparse signals and \\(\Lambda\\) denotes the collection of parameters in the LSTM block.

The whole training procedure is shown in below. Green, red and blue color points are non-zero elements of the given structured-sparse signal, which generate a batch of training pairs \\((\mathbf{r}\_1, \textbf{h}\_1)\\), \\((\mathbf{r}\_2, \textbf{h}\_2)\\), \\(\cdots\\), \\((\mathbf{r}\_s,\textbf{h}\_s)\\). The residual \\(\mathbf{r}\\)will serve as an input of the LSTM block, then we obtain an estimated index after the softmax layer. Compare it with groundtruth \\(\mathbf{h}\\), we acquire the cross-entropy \\(\mathbf{J}\\). Finally, parameters \\(\Lambda\\) will be updated via back-propagation algorithm. Please note, again, that the output of the LSTM will be inputted into the next LSTM as well. 

![train](https://github.com/yuanqw/yuanqw.github.io/blob/master/image/lvcc/train.jpg?raw=true)



## Experimental Results and Analysis

In this section, three typical structured-sparse signals are selected for our specific experiments, including block-sparse signals, tree-structured signals and uniform-sparse signals. Since there are a lot of algorithms direct at block-sparse signals, we will put our focus on recovery of block-sparse signals while give some comparisons with competitive algorithms on the other two structured-sparse signals. As a matter of convenience, we denote the proposed algorithm SSSR-LSTM, abbreviation of \textit{Structured-Sparse Signals Recovery based on LSTM}. In particular, all the sensing matrice \\(\mathbf{A}\\) in the following experiments are randomly generated with each entry drawn from standard normal distribution independently, columns of which are  orthonormal. All error terms added to the measurement \\(\mathbf{y}\\) is additive white Gaussian noise. Otherwise, due to the limited experiment circumstances, the number of given structured-sparse signals for generating training samples and test samples will be set to 5000 and 500, respectively. We have reasons to believe that the performance of our proposed algorithm will be further enhanced along with the increase of training samples. Finally, we introduce Normalized Mean Square Error (NMSE) to evaluate the performance of all algorithms,
$$
\text{NMSE}=\frac{\|\mathbf{x}-\hat{\mathbf{x}}\|\_F}{\|\mathbf{x}\|\_F}
$$

### Block-Sparse Signals

Let us start with block-sparse signals, the nonzero elements of which are occurring in clusters. We examine our proposed algorithm on four aspects, including sparsity, the quantity of blocks, sampling ratio and noise intensity. To generally evaluate the performance of our proposed algorithm, we compare it with some recently developed block-sparse signals recovery algorithms: [CluSS](http://www.sciencedirect.com/science/article/pii/S0165168411002490), [MBCS-LBP](https://www.sciencedirect.com/science/article/pii/S0165168414004411), [PCSBL](http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6967808), [EBSBL](http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6415293).  

For synthetic data, which is a series of one-dimensional block-sparse signals generated in a similar way to [EBSBL](http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6415293), we initially set the length of signals \\(N=200\\), the number of linear measurements \\(M=100\\), the sparsity of signals \\(S=30\\) and the quantity of blocks \\(T=5\\), the standard deviation of noise \\(\sigma = 0.05\\).

![exp1](https://github.com/yuanqw/yuanqw.github.io/blob/master/image/lvcc/exp1.png?raw=true)

For real-world data, we employ MNIST handwritten digit database, images of which can be regarded as two-dimensional block-sparse signals. Notably, we down-sample these images by ratio 0.5 for faster computational speed, thus the size of all training and test images is 14\\(\times\\)14 pixels. We set the sampling ratio \\(M/N = 0.5\\) while add the noise with standard deviation 0.1 to the measurement vector \\(\mathbf{y}\\) to simulate the error term. The recovery results of respective algorithms are demonstrated in Table1 and Figure 6. It could be visually observed that our proposed algorithm provides the most accurate estimation of original images. In addition, we could obviously find that EBSBL does not perform well on two-dimensional signals as it is on one-dimensional signals while results of other algorithms on two-dimensional signals turn in a consistent performance as they are on one-dimensional signals.

![exp2](https://github.com/yuanqw/yuanqw.github.io/blob/master/image/lvcc/exp2.png?raw=true)



![exp3](https://github.com/yuanqw/yuanqw.github.io/blob/master/image/lvcc/exp3.png?raw=true)

### Tree-Struture Signals

Generally, the wavelet coefficients of an image after a wavelet transform tend to be tree-structured and each wavelet coefficient serves as a parent for four children coefficients. Due to there are few algorithms specific to the tree-struture signals, we compare our proposed algorithm with [TSW-CS](http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=4907073). The standard test image Cameraman is employed for our experiment. We firstly resize the image to 32\\(\times\\)32 pixels, then the image is decomposed by wavelet transform on Daubechies 1 basis. To simulate the perturbation, we similarly add the noise with standard deviation 0.05 to the measurement vector \\(\mathbf{y}\\). The sampling ratio we used is 0.5.

![exp4](https://github.com/yuanqw/yuanqw.github.io/blob/master/image/lvcc/exp4.png?raw=true)

### Uniform-Sparse Signals

As for uniform-sparse signals, whose interval of two adjacent nonzero elements is uniform. Since we do not seek out any algorithms aimed at this type of structure, we have to compare our proposed algorithm with some classical CS reconstruction algorithms: [MP](https://www.di.ens.fr/~mallat/papiers/MallatPursuit93.pdf), [OMP](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.55.1254&rep=rep1&type=pdf), [BCS](http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=4524050), [LASSO](https://statweb.stanford.edu/~tibs/lasso/lasso.pdf). We set the length of signals is 200, interval of two adjacent nonzero elements is 5. In this subsection, we only compare the performance of aforementioned algorithms versus sampling ratio (\\(\sigma = 0.1\\)) and noise intensity (\\(M/N = 0.5\\)).

![exp5](https://github.com/yuanqw/yuanqw.github.io/blob/master/image/lvcc/exp5.png?raw=true)

## Conclusions

In this paper, a novel data-driven method based on recurrent neural networks was developed, which could flexibly cope with the structured-sparse signals recovery problem with any structure patterns. Particularly, the long short-term memory was introduced to the proposed algorithm for precisely capturing the correlations and dependencies among nonzero elements of sparse signals. Unlike some other conventional approaches that rely on the certain priori knowledge, SSSR-LSTM is prior-free, which is the reason why our proposed algorithm has the robust ability to explore and exploit intra-structure correlation with any structure patterns. Extensive experiments on three typical structured-sparse signals show that our proposed algorithm achieves a significant performance improvement as compared with the conventional approaches under any experiment settings. It also demonstrates the superiority of our proposed algorithm over other existing state-of-the-art methods.

<script type="text/javascript"
   src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
</script>
