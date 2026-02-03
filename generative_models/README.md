# The Generative Problem

We have samples from the random variable $x \sim p_x, \in \mathcal{X}$ from which we would like to sample.

**Idea**: Condition on a known variable $z\sim p_z, \in \mathcal{Z}$ and let there be a mapping (**Generator/Decoder**) $f:\mathcal{Z} \rightarrow \mathcal{X}$.

$$
x = f(z)
$$

where $f$ is a Universal Function Approximator (UFA). In practive $\f\$ is approximated by a Neural Network.

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

VAEs (<https://arxiv.org/pdf/1312.6114>) **propose**: sample from a $p_{z|x;\phi} \approx p_{z|x}$ which maximizes the marginal likelihood $p_{x;\theta}$, or at least its *Evidence Lower Bound* (ELBO)

$$
\begin{align}

\mathcal{L(\phi,\theta;x)} &= \int p_{z|x;\phi} \log(p_{x|z;\theta})~ dz - \int p_{z|x;\phi} \log\Big(\dfrac{p_{z|x;\phi}}{p_{z;\theta}}\Big)~ dz\\
& = \mathbb{E}_{ p_{z|x;\phi} } \Big[ \log(p_{x|z;\theta}) \Big] + D_{KL}\Big( p_{z|x;\phi}, p_{z;\theta} \Big)

\end{align}
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
\begin{align}

log ~p(x) \approx log ~p_{\theta,\phi}(\hat{x}) &= E_{q_{\phi}(x)}[ log ~p_{\theta,\phi}(\hat{x}) ] = E_{q_{\phi}(x)}[ log ~\dfrac{p_{\theta,\phi}(\hat{x},z)}{p_{\theta}(z|x)} ] = E_{q_{\phi}(x)}[ log ~\dfrac{p_{\theta,\phi}(\hat{x},z)q_{\phi}(z|x)}{p_{\theta}(z|x)q_{\phi}(z|x)} ]\\ &= E_{q_{\phi}(x)}[ log ~\dfrac{p_{\theta,\phi}(\hat{x},z)}{q_{\phi}(z|x)}] + E_{q_{\phi}(x)}[ log~\dfrac{q_{\phi}(z|x)}{p_{\theta}(z|x)} ]

\end{align}
$$

where $p_{\theta}(z|x)p(x) = p_{\theta}(x|z)p(z)$.~

In addition VAEs **propose**: make p_{x|z;\theta} stochastic to densify samples ins $\mathcal{X}$. This is done by densifying $\mathcal{Z}$ as follows

$$
p_{z|x;\phi} = \mathcal{N}(z;~ \mu_{\phi}(x), \sigma_{\phi}(x))
$$

which we can think as applying a Gaussian kernel on each sample $z$.

## Denoising Diffusion Probabilistic Models

DDPM (<https://arxiv.org/pdf/2006.11239>) instead of better choosing $p_z$, propose structuring $p_x$ and consequently $f_{\theta}$. Specifically as a *Markov Chain* with a Gaussian step  

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

which is optimized as in VAE, by maximizing the ELBO of the likelihood $p_{x|z;\theta}$, $\mathcal{L(\theta;x)}$. Only now, there is no approximation of $p_{z|x}$.

$$
\mathcal{L(\theta;x)} = \int p_{x_t|x_{t-1}} \Big[ p_{x_T|} + \sum\limits_{t=0}^{T-1} \log\Big(\dfrac{p_{x_t|x_{t+1};\theta}}{p_{x_t|x_{t-1}}}\Big)\Big]~ dz
$$

<!-- ## Normalizing Flows

NFs aim to constraint the design of f, such that using if auto-regressively applied is should in theory converge to $p_x$. This circles back around the Central Limit Theorem, since a NNs are already auto-regressive functions - however, their hidden state space is not constrained hardly. NFs, assume the hidden state space is structured (same dimensionality for all layers), so that the *Change of Variables Theorem* can be applied at each step.




## Consistency Models

DDPM requires a lot of iterative steps (~10K), which is costly. Consistency models, formulated the iterative processes as the solution of a DE from which solving the DE can be done more effectively by any method in the bibliography.

## Distillation of DDPMs

Forumlating diffusion DE and using solvers, suffers the usual local truncation errors, which add up after multiple steps. To bypass that, Distillation methods employ a  learn a model to solve that DE in fewer steps. That improves sampling the sampling time of DDPMs. -->