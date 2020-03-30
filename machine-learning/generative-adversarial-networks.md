---
description: GANs (mainly in image synthesis)
---

# Generative Adversarial Networks

## Survey Papers / Repos

* Are GANs Created Equal? A Large-Scale Study [\[1711.10337\]](https://arxiv.org/abs/1711.10337)
* Which Training Methods for GANs do actually Converge? [\[1801.04406\]](https://arxiv.org/abs/1801.04406)
* A Large-Scale Study on Regularization and Normalization in GANs [\[1807.04720\]](https://arxiv.org/abs/1807.04720)
* [hindupuravinash/the-gan-zoo](https://github.com/hindupuravinash/the-gan-zoo)

## Resources

* [google/compare\_gan](https://github.com/google/compare_gan)
* [TF-GAN](https://github.com/tensorflow/gan): TensorFlow-GAN
* [torchgan](https://github.com/torchgan/torchgan)
* [PyTorch-GAN](https://github.com/eriklindernoren/PyTorch-GAN)
* [lzhbrian/metrics](https://github.com/lzhbrian/metrics): IS, FID implementation in TF, PyTorch

## Models

### Loss functions

* Vanilla GAN [\[1406.2661\]](https://arxiv.org/abs/1406.2661)
* EBGAN [\[1609.03126\]](https://arxiv.org/abs/1609.03126)
* LSGAN [\[1611.04076\]](https://arxiv.org/abs/1611.04076)
* WGAN [\[1701.07875\]](https://arxiv.org/abs/1701.07875)
* BEGAN [\[1703.10717\]](https://arxiv.org/abs/1703.10717)
* Hinge Loss [\[1705.02894\]](https://arxiv.org/abs/1705.02894)

### Regularization

* Gradient Penalty [\[1704.00028\]](https://arxiv.org/abs/1704.00028)
* DRAGAN [\[1705.07215\]](https://arxiv.org/abs/1705.07215)
* Consistency Regularization [\[1910.12027\]](https://arxiv.org/abs/1910.12027)

### Architecture

* Deep Convolution GAN \(DCGAN\) [\[1511.06434\]](https://arxiv.org/abs/1511.06434)
* Progressive Growing of GANs \(PGGAN\) [\[1710.10196\]](https://arxiv.org/abs/1710.10196)
* Self Attention GAN \(SAGAN\) [\[1805.08318\]](https://arxiv.org/abs/1805.08318)
* BigGAN [\[1809.11096\]](https://arxiv.org/abs/1809.11096)
* Style based Generator \(StyleGAN\) [\[1812.04948\]](https://arxiv.org/abs/1812.04948)
* Mapping Network \(StyleGAN\) [\[1812.04948\]](https://arxiv.org/abs/1812.04948)
* LOGAN: Latent Optimisation for Generative Adversarial Networks [\[1912.00953\]](https://arxiv.org/abs/1912.00953)

### Conditional GANs

* Vanilla Conditional GANs [\[1411.1784\]](https://arxiv.org/abs/1411.1784)
* Auxiliary Classifer GAN \(ACGAN\) [\[1610.09585\]](https://arxiv.org/abs/1610.09585)

## Others

### Tricks

* Two time-scale update rule \(TTUR\) [\[bioinf-jku/TTUR\]](https://github.com/bioinf-jku/TTUR) [\[1706.08500\]](https://arxiv.org/abs/1706.08500)
* Self-Supervised GANs via Auxiliary Rotation Loss \(SS-GAN\) [\[1811.11212\]](https://arxiv.org/abs/1811.11212)

### Metrics \(my implementation: [lzhbrian/metrics](https://github.com/lzhbrian/metrics)\)

* Inception Score [\[1606.03498\]](https://arxiv.org/abs/1606.03498) [\[1801.01973\]](https://arxiv.org/abs/1801.01973)
  * Assumption
    * **MEANINGFUL**: The generated image should be clear, the output probability of a classifier network should be \[0.9, 0.05, ...\] \(largely skewed to a class\). $$p(y|\mathbf{x})$$ is of low entropy.
    * **DIVERSITY**: If we have 10 classes, the generated image should be averagely distributed. So that the marginal distribution $$p(y) = \frac{1}{N} \sum_{i=1}^{N} p(y|\mathbf{x}^{(i)})$$ __is of high entropy.
    * Better models: KL Divergence of $$p(y|\mathbf{x})$$ and $$p(y)$$ should be high.
  * Formulation
    * $$\text{IS} = \exp (\mathbb{E}_{\mathbf{x} \sim p_g} D_{KL} [p(y|\mathbf{x}) || p(y)] )$$
    * where
      * $$\mathbf{x}$$ is sampled from generated data
      * $$p(y|\mathbf{x})​$$ is the output probability of Inception v3 when input is $$\mathbf{x}​$$
      * $$p(y) = \frac{1}{N} \sum_{i=1}^{N} p(y|\mathbf{x}^{(i)})$$ is the average output probability of all generated data \(from InceptionV3, 1000-dim vector\)
      * $$D_{KL} (\mathbf{p}||\mathbf{q}) = \sum_{j} p_{j} \log \frac{p_j}{q_j}$$, where $$j$$ is the dimension of the output probability.
  * Reference
    * Official TF implementation is in [openai/improved-gan](https://github.com/openai/improved-gan)
    * Pytorch Implementation: [sbarratt/inception-score-pytorch](https://github.com/sbarratt/inception-score-pytorch)
    * TF seemed to provide a [good implementation](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/gan/python/eval/python/classifier_metrics_impl.py)
    * [scipy.stats.entropy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.entropy.html)
    * [zhihu: Inception Score 的原理和局限性](https://zhuanlan.zhihu.com/p/54146307)
    * [A Note on the Inception Score](https://arxiv.org/abs/1801.01973)
* FID Score [\[1706.08500\]](https://arxiv.org/abs/1706.08500)
  * Formulation
    * $$\text{FID} = ||\mu_r - \mu_g||^2 + Tr(\Sigma_{r} + \Sigma_{g} - 2(\Sigma_r \Sigma_g)^{1/2})​$$
    * where
      * $$Tr$$ is [trace of a matrix \(wikipedia\)](https://en.wikipedia.org/wiki/Trace_%28linear_algebra%29)
      * $$X_r \sim \mathcal{N}(\mu_r, \Sigma_r)$$ and $$X_g \sim \mathcal{N}(\mu_g, \Sigma_g)$$ are the 2048-dim activations  the Inception v3 pool3 layer
      * $$\mu_r$$ is the mean of real photo's feature
      * $$\mu_g$$ is the mean of generated photo's feature
      * $$\Sigma_r$$ is the covariance matrix of real photo's feature
      * $$\Sigma_g$$ is the covariance matrix of generated photo's feature
  * Reference
    * Official TF implementation: [bioinf-jku/TTUR](https://github.com/bioinf-jku/TTUR)
    * Pytorch Implementation: [mseitzer/pytorch-fid](https://github.com/mseitzer/pytorch-fid)
    * TF seemed to provide a [good implementation](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/gan/python/eval/python/classifier_metrics_impl.py)
    * [zhihu: Frechet Inception Score \(FID\)](https://zhuanlan.zhihu.com/p/54213305)
    * [Explanation from Neal Jean](https://nealjean.com/ml/frechet-inception-distance/)

