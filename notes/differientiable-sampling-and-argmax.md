# Differientiable Sampling and Argmax

**Softmax** is a commonly used function for turning an unnormalized log probability into a normalized probability.

$$
softmax(\mathbf{o}) = \frac{e^{\mathbf{o}}}{\sum_{j} e^{o_j}}
$$

After softmax, we usually **sample** from this distribution, or taking an **argmax** function to select the index. However, one can notice that neither the sample nor the argmax is **indifferientiable**.

Researchers have 

### Reference

* Softmax
* Soft argmax [\[NIPSW 2016\]](https://zhegan27.github.io/Papers/textGAN_nips2016_workshop.pdf)
* Gumbel Softmax [\[1611.01144\]](https://arxiv.org/abs/1611.01144)



