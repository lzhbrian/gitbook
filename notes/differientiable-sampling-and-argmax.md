---
description: 'WIP, last updated: 2019.12.6'
---

# Differientiable Sampling and Argmax

## Introduction

**Softmax** is a commonly used function for turning an **unnormalized log probability** into a normalized probability \(or **categorical distribution**\).

$$
\mathbf{\pi} = \text{softmax}(\mathbf{o}) = \frac{e^{\mathbf{o}}}{\sum_{j} e^{o_j}},\\o_j \in (-\infty, +\infty)
$$

Say $$\mathbf{o}$$ is the output of a neural network before softmax, we call $$\mathbf{o}$$ the **unnormalized log probability.**

After softmax, we usually **sample** from this categorical distribution, or taking an **argmax** function to select the index. However, one can notice that neither the **sampling** nor the **argmax** is **differientiable**.

Researchers have proposed several works to make this possible. I am going to discuss them here.

## Sampling

I will introduce Gumbel Softmax [\[1611.01144\]](https://arxiv.org/abs/1611.01144), which have made the **sampling** procedure differentiable.

### Gumbel Max

First, we need to introduce **Gumbel Max**. In short, Gumbel Max is a trick to use gumbel distribution to sample a categorical distribution.

Say we want to sample from a categorical distribution $$\mathbf{\pi}$$.  
The usual way of doing this is using $$\pi$$ to separate $$[0, 1]$$ into intervals, sampling from a uniform distribution $$\text{U} \sim[0, 1]$$, and see where it locates.

The Gumbel Max trick provides an alternative way of doing this. It use **Reparameterization Trick** to avoid the stochastic node during backpropagation.

$$
y = \arg \max_{i} (o_i +g_i)
$$

where $$g_i \sim \text{Gumbel}(0, 1)$$, which can be sampled by 

$$
-\log(-\log(\text{Uniform}[0, 1]))
$$

We can prove that $$y$$ is distributed according to $$\mathbf{\pi}$$.

{% hint style="info" %}
### **Prove that**

$$y = \arg \max_{i} (o_i +g_i)$$, where $$g_i \sim \text{Gumbel}(0, 1)$$, which sampled by 

$$
-\log(-\log(\text{Uniform}[0, 1]))
$$

is distributed with

$$
\pi_i = \text{softmax}(o_i) = \frac{e^{o_i}}{\sum_{j} e^{o_j}}
$$

### **Prerequisites**

**Gumbel Distribution** \(param by location ****$$\mu$$, and scale $$\beta>0$$\) \([wikipedia](https://en.wikipedia.org/wiki/Gumbel_distribution)\)  
**CDF:** $$F(x; \mu, \beta) = e^{-e^{(x-\mu)/\beta}}$$  
**PDF:** $$f(x; \mu, \beta) = \frac{1}{\beta} e^{-(z+e^{-z})}, z = \frac{x-\mu}{\beta}$$  
**Mean:** $$\text{E}(X) = \mu+\gamma\beta, \gamma \approx 0.5772$$is the [Euler–Mascheroni constant](https://en.wikipedia.org/wiki/Euler%E2%80%93Mascheroni_constant).  
**Quantile Function:** $$Q(p) = \mu-\beta \log(-\log(p))$$\([Quantile Function](https://en.wikipedia.org/wiki/Quantile_function) is used to sample random variables from a distribution given CDF, it is also called inverse CDF\)

### Proof

We actually want to prove that $$\text{Gumbel}(\mu=o_i, \beta=1)$$ is distributed with $$\pi_i = \frac{e^{o_i}}{\sum_{j} e^{o_j}}$$.

 $$\text{Gumbel}(\mu=o_i, \beta=1)$$ has the following PDF and CDF

$$
\begin{align}
f(x; \mu, 1) &= e^{-(x-\mu) – e^{-(x-\mu)}}\\
F(x; \mu, 1) &= e^{-e^{-(x-\mu)}}
\end{align}
$$

.The Probability that all other $$\pi_{j \neq i}$$ are less than $$\pi_i$$ is:

$$
\Pr(\pi_i ~\text{is the largest} | \pi_i, \{o_{j}\}) = \prod_{j \neq i} e^{-e^{-(\pi_i - o_j)}}
$$

We know the marginal distribution over $$\pi_i$$ and we need to integrate it out to find the overall probability:

$$
\begin{align}  
\Pr(\text{$i$ is largest}|\{o_{j}\}) &= \int e^{-(\pi_i-o_i)-e^{-(\pi_i-o_i)}} \times \prod_{j\neq i}e^{-e^{-(\pi_i-o_j)}} \mathrm{d}\pi_i  \\
&=\int e^{-\pi_i + o_i -e^{-\pi_i} \sum_{j} e^{o_j}}\mathrm{d}\pi_i \\
&=\frac{e^{o_i}}{\sum_{j}e^{o_j}}
\end{align}
$$

which is exactly a softmax probablity. QED.

**Reference:** [**https://lips.cs.princeton.edu/the-gumbel-max-trick-for-discrete-distributions/**](https://lips.cs.princeton.edu/the-gumbel-max-trick-for-discrete-distributions/)\*\*\*\*
{% endhint %}

### Gumbel Softmax

Notice that there is still an argmax in Gumbel Max, which still makes it indifferentiable. Therefore, we use a softmax function to approximate this argmax procedure.

$$
\mathbf{y} = \frac{e^{(o_i+g_i) / \tau}}{\sum_{j}e^{(o_j+g_j) / \tau}}
$$

where $$\tau \in (0, \infty)$$ is a temparature hyperparameter.

We note that the output of Gumbel Softmax function here is a vector which sum to 1, which somewhat looks like a one-hot vector \(but it's not\).  
So by far, this does not actually replace the argmax function.

To actually get a pure one-hot vector, we need to use a **Straight-Through \(ST\) Gumbel Trick**.  
Let's directly see an [implementation of Gumbel Softmax in PyTorch](https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.gumbel_softmax)  
\(We use the hard mode, soft mode does not get a pure one-hot vector\).

```python
def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
    # type: (Tensor, float, bool, float, int) -> Tensor
    r"""
    Samples from the Gumbel-Softmax distribution (`Link 1`_  `Link 2`_) and optionally discretizes.

    Args:
      logits: `[..., num_features]` unnormalized log probabilities
      tau: non-negative scalar temperature
      hard: if ``True``, the returned samples will be discretized as one-hot vectors,
            but will be differentiated as if it is the soft sample in autograd
      dim (int): A dimension along which softmax will be computed. Default: -1.

    Returns:
      Sampled tensor of same shape as `logits` from the Gumbel-Softmax distribution.
      If ``hard=True``, the returned samples will be one-hot, otherwise they will
      be probability distributions that sum to 1 across `dim`.

    .. note::
      This function is here for legacy reasons, may be removed from nn.Functional in the future.

    .. note::
      The main trick for `hard` is to do  `y_hard - y_soft.detach() + y_soft`

      It achieves two things:
      - makes the output value exactly one-hot
      (since we add then subtract y_soft value)
      - makes the gradient equal to y_soft gradient
      (since we strip all other gradients)

    Examples::
        >>> logits = torch.randn(20, 32)
        >>> # Sample soft categorical using reparametrization trick:
        >>> F.gumbel_softmax(logits, tau=1, hard=False)
        >>> # Sample hard categorical using "Straight-through" trick:
        >>> F.gumbel_softmax(logits, tau=1, hard=True)

    .. _Link 1:
        https://arxiv.org/abs/1611.00712
    .. _Link 2:
        https://arxiv.org/abs/1611.01144
    """
    if eps != 1e-10:
        warnings.warn("`eps` parameter is deprecated and has no effect.")

    gumbels = -torch.empty_like(logits).exponential_().log()  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret
```

When fowarding, the code use an argmax to get an actual one-hot vector.   
And it uses `ret = y_hard - y_soft.detach() + y_soft`, `y_hard` has no grad, and by minusing `y_soft.detach()` and adding `y_soft`, it achieves a grad from `y_soft` without modifying the forwarding value.

So eventually, we are able to get a pure one-hot vector in forward pass, and a grad when back propagating, which **makes the sampling procedure differientiable**.

Finally, let's look at how $$\tau$$affects the sampling procedure. The below image shows the sampling distribution \(which is also called the Concrete Distribution [\[1611.00712\]](https://arxiv.org/abs/1611.00712)\) and one random sample instance when using different hyperparameter $$\tau$$.

![image from https://arxiv.org/abs/1611.01144](../.gitbook/assets/image%20%282%29.png)

> when $$\tau \rightarrow 0$$, the softmax becomes an argmax and the Gumbel-Softmax distribution becomes the categorical distribution. During training, we let $$\tau > 0$$ to allow gradients past the sample, then gradually anneal the temperature $$\tau$$ \(but not completely to 0, as the gradients would blow up\).
>
> from Eric Jang. [https://blog.evjang.com/2016/11/tutorial-categorical-variational.html](https://blog.evjang.com/2016/11/tutorial-categorical-variational.html)

## Argmax

How to make argmax differentiable?

{% hint style="info" %}
Intuitively, the **Straight-Through Trick** is also applicable for softmax+argmax \(or softargmax + argmax\).  
I am still not sure, needs more digging in the literature.
{% endhint %}

Some have introduced the soft-argmax function. It doesn't actually makes it differentiable, but use a continuous function to approximate the softmax+argmax procedure.

$$
\mathbf{\pi} = \text{soft-argmax}(\mathbf{o}) = \frac{e^{\mathbf{\beta o}}}{\sum_{j} e^{\beta o_j}}
$$

where $$\beta$$ can be a large value to make $$\mathbf{\pi}$$ very much "look like" a one-hot vector.



## Discussion

* Goal
  * **softmax + argmax** is used for classification, we only want the index with the highest probability.
  * **gumbel softmax + argmax** is used for sampling, we may want to sample an index not with the highest probability.
* Deterministic
  * **softmax + argmax** is deterministic. Get the index with the highest probablity.
  * **gumbel softmax + argmax** is stochastic. We need to sample from a gumbel distribution in the beginning.
* Output vector
  * **softmax** and **gumbel softmax** aboth output a vector sum to 1.
  * **softmax** outputs a _normalized probability distribution_.
  * **gumbel softmax** outputs a _sample_ somewhat more similar to a one-hot vector.\(can be controlled by $$\tau$$\)
* **Straight-Through Trick** can actually be applied to both **softmax + argmax** and **gumbel softmax + argmax**, which can make both of them differentiable. \(?\)

## Reference

* Gumbel Softmax [\[1611.01144\]](https://arxiv.org/abs/1611.01144)
* Concrete Distribution \(Gumbel Softmax Distribution\) [\[1611.00712\]](https://arxiv.org/abs/1611.00712)
* Eric Jang official blog: [https://blog.evjang.com/2016/11/tutorial-categorical-variational.html](https://blog.evjang.com/2016/11/tutorial-categorical-variational.html)
* PyTorch Implementation of Gumbel Softmax: [https://pytorch.org/docs/stable/nn.functional.html\#torch.nn.functional.gumbel\_softmax](https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.gumbel_softmax)
* [https://timvieira.github.io/blog/post/2014/07/31/gumbel-max-trick/](https://timvieira.github.io/blog/post/2014/07/31/gumbel-max-trick/)
* [https://lips.cs.princeton.edu/the-gumbel-max-trick-for-discrete-distributions/](https://lips.cs.princeton.edu/the-gumbel-max-trick-for-discrete-distributions/)



