# SDE basics

Consider the $It\hat o$ SDE

$$
\dfrac{dx}{dt} = f(x(t),t;\theta) + g(x(t),t;\theta)\dfrac{dw}{dt}
$$

* $g$: **diffusion** term
* $f$ **drift** term
* $w(t)$: **Wiener process** (Brownian), $(w(t+\delta)-w(t)) \sim \mathcal{N}(0,\delta)$

we aim to compute $\theta$ given observations of $x,f$ and $g$.

**Examples**

*  (**Ornstein–Uhlenbeck**) $\dfrac{dx}{dt} = \theta(\mu - x(t)) + \sigma\dfrac{dw}{dt}$


# MLE

$$
\mathcal{L}(\theta) = \int \log p_{x_t|x_{t-1};\theta}(x) ~dx
$$

where $p_{x_t|x_{t-1};\theta}$ follows the **Fokker-Planck** PDE

$$
\dfrac{\partial p}{\partial t} = \dfrac{\partial}{\partial x}\Big[ f(x(t);\theta)p \Big] + \dfrac{1}{2} \dfrac{\partial^2 p}{\partial x^2}\Big[ g(x;\theta)^2p \Big] 
$$

**Assumptions**: Markov chain, ergodicity, stationarity, uniform sampling

## Euler-Maruyama

$$
x_{i+1} = x_{i+1} + f(x_i;\theta )\delta + g(x_i;\theta)(w_{i+1}-w_i)
$$

and

$$
p(x_{i+1}|x_i;\theta) \approx \mathcal{N}(x_i+\delta,~g(x_i)^2\delta) 
$$

# Neural SDEs as Infinite-Dimensional GANs
<http://arxiv.org/abs/2102.03657>

Since SDEs are inherently stochastic systems, they can be thought of as generative systems that produces samples (time-series) from noise. Therefore, this paper proposes (Wasserstein) GANs optimization schema to learn SDE parameters.

$$
\underset{\phi}{\min}~ \underset{\theta}{\max} ~ \mathbb{E}_{p_x}\big[ D(x;\phi) \big] + \mathbb{E}_{p_w}\big[ D(G(w(t);\theta);\phi) \big]
$$

where the Generator $G$ produces $\hat{x}(t)$ as follows

$$
\hat{x}(t) = \alpha_{\theta} x(t) + \beta_{\theta}\\
\dfrac{dx}{dt} = \mu(x(t),t;\theta) + \sigma(x(t),t;\phi)\dfrac{dw}{dt}\\
h(0) = \xi_{\phi}(\hat{x}(0))
$$

and the discriminator $D$ produces $d$ based only on the last state of a respective SDE.

$$
d = m_{\phi} h(t)\\
\dfrac{dh}{dt} = f(h(t),t;\phi) + g(h(t),t;\phi)\dfrac{d\hat {x}}{dt}\\
h(0) = \xi_{\phi}(\hat{x}(0))
$$

**Limitation**: Sparse time sampling