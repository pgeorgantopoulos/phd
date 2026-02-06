# The Generative Problem

We have samples from the random variable $x \sim p_x, \in \mathcal{X}$ from which we would like to sample.

**Idea**: Condition on a known variable $z\sim p_z, \in \mathcal{Z}$ and let there be a mapping (**Generator/Decoder**) $f:\mathcal{Z} \rightarrow \mathcal{X}$.

$$
x = f(z)
$$

where $f$ is a Universal Function Approximator (UFA). In common practice $f$ is approximated by a Neural Network (NN):

$$
f_\theta:\mathcal{Z} \rightarrow \mathcal{X}
$$ 

**Limitation**: 
* the Lipschitz continuouity of $\hat f_{\theta}$
* the geometric complexity of $p_x$ with respect to the choice of $p_z$
* the dimensional discrepancy of $z$ and $x$
affect the "*expressivity*" of $p_{x|z;\theta}$ and therefore the marginal $p_{x;\theta}$ ("mode collapse").

## Generative Adversarial Neural Networks

??

## Variational Autoencoders

VAEs (https://arxiv.org/pdf/1312.6114) **propose**: sample from a $p_{z|x;\phi} \approx p_{z|x}$ which maximizes the marginal likelihood $p_{x;\theta}$, or at least its *Evidence Lower Bound* (ELBO)

$$
\mathcal{L}(\phi,\theta;x) = \int p_{z|x;\phi} \log(p_{x|z;\theta})~ dz - \int p_{z|x;\phi} \log\Big(\dfrac{p_{z|x;\phi}}{p_{z;\theta}}\Big)~ dz\\
= \mathbb{E}_{ p_{z|x;\phi} } \Big[ \log(p_{x|z;\theta}) \Big] + D_{KL}\Big( p_{z|x;\phi}, p_{z;\theta} \Big)
$$

<!-- where
$$
p_{z|x;\theta} = \dfrac{p_{x|z;\theta}~ p_z}{p_x}
$$
is the Generator's posterior -->

Observe, VAEs introduce the (**Encoder**) mapping

$$
f_\phi:\mathcal{X} \rightarrow \mathcal{Z}
$$

~ **Unstructured notes on ELBO derivation**

Let $\hat{x} = \mathcal{D}(z;\theta)$ and $z = \mathcal{E}(x;\phi)$ where $X\sim p(x)$ and $p_{\theta,\phi}(\hat{x})$ its estimated density through a Encoder-Decoder schema.

$$
log ~p(x) \approx log ~p_{\theta,\phi}(\hat{x}) = E_{q_{\phi}(x)}[ log ~p_{\theta,\phi}(\hat{x}) ] = E_{q_{\phi}(x)}[ log ~\dfrac{p_{\theta,\phi}(\hat{x},z)}{p_{\theta}(z|x)} ] = E_{q_{\phi}(x)}[ log ~\dfrac{p_{\theta,\phi}(\hat{x},z)q_{\phi}(z|x)}{p_{\theta}(z|x)q_{\phi}(z|x)} ]\\ 
$$
$$
= E_{q_{\phi}(x)}[ log ~\dfrac{p_{\theta,\phi}(\hat{x},z)}{q_{\phi}(z|x)}] + E_{q_{\phi}(x)}[ log~\dfrac{q_{\phi}(z|x)}{p_{\theta}(z|x)} ]
$$

where $p_{\theta}(z|x)p(x) = p_{\theta}(x|z)p(z)$.~

In addition VAEs **propose**: make p_{x|z;\theta} stochastic to densify samples ins $\mathcal{X}$. This is done by densifying $\mathcal{Z}$ as follows

$$
p_{z|x;\phi} = \mathcal{N}(z;~ \mu_{\phi}(x), \sigma_{\phi}(x))
$$

which we can think as applying a Gaussian kernel on each sample $z$.

## Denoising Diffusion Probabilistic Models

DDPM (https://arxiv.org/pdf/2006.11239) instead of better choosing $p_z$, propose structuring $p_x$ and consequently $f_{\theta}$. Specifically as a *Markov Chain* with a Gaussian step. Another way to think of DDPM is that it replaces $\mathcal{Z}$ with $\mathcal{X}$. Instead of $p_{z|x;\phi}$ we have the predetermined (needs no training) forward model $p_{x_{t}|x_{t-1}}$. The **Denoiser** (used to be *Decoder*) implements the backward model $p_{x_{t-1}|x_{t};\theta}$ instead of $p_{x|z;\theta}$.

$$
p_{x_0;\theta} = p_{x_T}\prod_{t=0}^{T-1} p_{x_t|x_{t+1};\theta}
$$

where $p_{x_T}:=p_z=\mathcal{N}\Big(x_T;0,1\Big)$ and

$$
p_{x_t|x_{t+1};\theta}=\mathcal{N}\Big(x_t;\mu_{\theta}(x_t,t), \sigma_{\theta}(x_t,t)\Big)
$$

$$
p_{x_t|x_{t-1}} = \mathcal{N}\Big(\sqrt{1-\beta_t} x_{t-1}, \beta_t)\Big)
$$

Observe,

$$
f_{\theta}: \mathcal{T}\times\mathcal{Z} \rightarrow \mathcal{X}
$$

which is optimized as in VAE, by maximizing the ELBO of the likelihood $p_{x|z;\theta}$, $\mathcal{L}(\theta;x)$. Only now, there is no approximation of $p_{z|x}$.

$$
\mathcal{L}(\theta;x) = \int p_{x_t|x_{t-1}} \Big[ p_{x_T} + \sum\limits_{t=0}^{T-1} \log\Big(\dfrac{p_{x_t|x_{t+1};\theta}}{p_{x_t|x_{t-1}}}\Big)\Big]~ dx_t
$$

Remember, the original generative problem requires sampling from $\mathcal{X}$. DDPM allows for sampling from $\mathcal{X}$ at any $t$, since it can be shown that

$$
p_{x_t|x_0} = \mathcal{N}\Big(x_t; \sqrt{\bar \alpha_t} x_0, (1 − \bar \alpha_t)\Big),\quad \bar\alpha_t:=\prod_s^t\alpha_s, \quad \alpha_t := 1-\beta_t
$$

That enables constraining the loss function per $t$-step, as follows

$$
\mathcal{L}(\theta) = \mathbb{E}_{t,x_0,\epsilon} \Big[ \| \epsilon - \epsilon_{\theta}(\sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t}\epsilon,~t) \|^2 \Big]
$$

\* this loss is derived after assumptions on $\sigma_{\theta}$ and $x_t(x_0,\epsilon)$. 

Sampling is then performed by

$$
x_{i-1} = \dfrac{1}{\sqrt{1-\beta_i}}\big(x_i + \beta_i\epsilon_{\theta}(x_i,i)\big) + \sqrt{\beta_i}\epsilon_i,\quad i=N,\dots,1
$$

## Score-based Generative Models


1908 ["Langevin dynamics"](https://en.wikipedia.org/wiki/Langevin_dynamics) \
2002 [Langevin-dynamics: Sampling with gradient-based Markov Chain Monte Carlo approaches](https://github.com/alisiahkoohi/Langevin-dynamics)\
2011 ["Bayesian Learning via Stochastic Gradient Langevin"](https://www.stats.ox.ac.uk/~teh/research/compstats/WelTeh2011a.pdf)\
2018 ["On sampling from a log-concave density using kinetic Langevin diffusions"](https://arxiv.org/abs/1807.09382) \
2019 ["Generative Modeling by Estimating Gradients of the Data Distribution"](https://arxiv.org/abs/1907.05600) \
2020 ["Generative Modeling with Denoising Auto-Encoders and Langevin Sampling"](https://arxiv.org/abs/2002.00107)\ 
2021 ["Diffusion model"](https://en.wikipedia.org/wiki/Diffusion_model)\
2022 ["SCORE-BASED GENERATIVE MODELING"](https://openreview.net/pdf?id=CzceR82CYc)\
2023 ["Discrete Langevin Samplers via Wasserstein Gradient Flow"](https://proceedings.mlr.press/v206/sun23f.html)

Score-based models make use of **Langevin dynamics** to explore samples within $\mathcal{X}$ ["Bayesian learning via stochastic gradient Langevin dynamics", (2011)](https://www.stats.ox.ac.uk/~teh/research/compstats/WelTeh2011a.pdf).

$$
\dfrac{dx}{dt} = \nabla_x \log p_x + \dfrac{dz}{dt}
$$

$\nabla_x \log p_x(x)$ is called the "**score**" of $p_x$. 

The *Euler–Maruyama* **discretization** of Langevin dynamics is

$$
x_t = x_{t-1} + \dfrac{\epsilon}{2} \nabla_x \log p_{x_{t-1}} + \sqrt{\epsilon}z_{t},\quad z\sim\mathcal{N}(0,1)
$$

Therefore, the overall score-based methodology includes two steps: approximating $\nabla_x \log p_x$ and sampling.

### Score matching 

["Estimation of non-normalized statistical models by score matching", (2005)](https://jmlr.org/papers/volume6/hyvarinen05a/hyvarinen05a.pdf)

["Sliced score matching: A scalable approach to density and score estimation", (2019)](https://proceedings.mlr.press/v115/song20a/song20a.pdf)

Here $\mathcal{L}(\theta)$ captures the direction of higher data density (notation is abused as we move to vector derivatives):

$$
\underset{\theta}{argmin}\ \mathcal{L}(\theta) = \underset{\theta}{argmin}\ \mathbb{E}_{p_x}\Big[ \| s_{\theta}(x) - \nabla_x \log p_x(x) \|^2 \Big] = \underset{\theta}{argmin}\ \mathbb{E}_{p_x}\Big[ \dfrac{1}{2} \| s_{\theta}(x) \|^2 + tr\big( \nabla_x s_{\theta}(x) \big) \Big]
$$

**Limitation**: Gradient information is zero where $\mathcal{X}$ doesn't admit any density. This is common in low-dimensional $x$ (e.g. physical systems). Score matching fails (analytically) for the same reason.

**Limitation**: Computing $\dfrac{dp_x}{dx}$ is prohibiting for high-dimensional $x$.

["A connection between score matching and denoising autoencoders", (2011)](https://www.iro.umontreal.ca/~vincentp/Publications/smdae_techreport.pdf) propose marginilizing $p_x$ to a knwon (Gaussian) $p_{\hat{x}} = \int p_{\hat{x}|x} p_x~ dx$

$$
\mathcal{L}(\theta) = \dfrac{1}{2} \mathbb{E}_{p_{\hat{x}|x}, p_x}\Big[ \| s_{\theta}(\hat x) - \nabla_{\hat x} \log p_{\hat{x}|x} \|^2 \Big]
$$

$$
= \sum\limits_{i=1}^{N} \sigma_i^2 \mathbb{E}_{p(x)}\mathbb{E}_{p_{\hat x|x}} \Big[ \| s_{\theta}(\hat x,\sigma_i) - \nabla_{\hat x} \log p_{\hat x|x;\sigma_i} \|^2 \Big]
$$

with $p_{\hat x|x} := \mathcal{N}(\hat x; x,\sigma^2I)$. This is analytically proven to work for small perturbations ["Bayesian learning via stochastic gradient Langevin dynamics", (2011)](https://www.stats.ox.ac.uk/~teh/research/compstats/WelTeh2011a.pdf). $s_{\theta}(\hat x,\sigma)$ is called a **Noise Conditioned Score Network** (NCSN).

The discretization from which we sample from then becomes

$$
x_i^m = x_{i}^{m-1} + \epsilon_i s_{\theta}(x_i^{m-1},\sigma_i) + \sqrt{2\epsilon_i}z_{t}^m,\quad i=1,\dots,N, m=1,\dots,M
$$

Note that M steps are required for a single sample $x_i$.

### Score-based Generative Models with SDEs

["Score-Based Generative Modeling through Stochastic Differential Equations", (2021)](https://arxiv.org/abs/2011.13456) consider 

* the oringinal $It\hat o$-type SDE of Langevin dynamics.
* the Noise Conditioned Score Network

Sampling requires inverting the SDE wrt $t$, which is again an SDE *"Reverse-time diffusion equation models", (1982)* of the frorm

$$
\dfrac{dx}{dt} = \dfrac{d}{dt}\big[ f(x,t) - g(t)^2 \nabla_x \log p_{x;t} \big] + g(t)\dfrac{dz}{dt},\quad t\in[T,0]
$$
where $p_{x;T}$ is known

??

Sampling is done by numerically solving

$$
\dfrac{dx}{dt} = -\dfrac{1}{2} \beta(t)x + \sqrt{\beta(t)(1-\exp{-1\int_{0}^{t}\beta(s)~ds})} \dfrac{dz}{dt}
$$

<!-- ## Normalizing Flows

NFs aim to constraint the design of f, such that using if auto-regressively applied is should in theory converge to $p_x$. This circles back around the Central Limit Theorem, since a NNs are already auto-regressive functions - however, their hidden state space is not constrained hardly. NFs, assume the hidden state space is structured (same dimensionality for all layers), so that the *Change of Variables Theorem* can be applied at each step.




## Consistency Models

DDPM requires a lot of iterative steps (~10K), which is costly. Consistency models, formulated the iterative processes as the solution of a DE from which solving the DE can be done more effectively by any method in the bibliography.

## Distillation of DDPMs

Forumlating diffusion DE and using solvers, suffers the usual local truncation errors, which add up after multiple steps. To bypass that, Distillation methods employ a  learn a model to solve that DE in fewer steps. That improves sampling the sampling time of DDPMs. -->