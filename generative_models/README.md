We have samples $x \sim p_x, \int \mathcal{X}$ from which we would like to sample

Condition on a known variable $z\sim p_z, \int \mathcal{Z}$ and suppose there is mapping $f:\mathcal{Z} \rightarrow \mathcal{X}$.

$$
x = f(z)
$$

If $f$ is a Universal Function Approximator, by the *Central Limit Theorem*, we should be able to train this down to 0 error. In practive $\f\$ is a Neural Network. There we should expect some limitations

**Limitation**: $\hat f:\mathcal{\hat X} \rightarrow \mathcal{\hat Z}$ 

where  $\mathcal{\hat X} \subset \mathcal{X}, \mathcal{\hat Z} \subset \mathcal{Z}$. Therefore we can theorize on how other design choices affect the results. 

One **idea** is to examine the choice of $p_z$: if it is Gaussian most of $\mathcal{\hat X}$ is accumulated around some mean. Therefore at best, this works if $\mathcal{\hat Z}$ has low variance as well. If we chose a more complex $p_z$ then we could account for that, but it there is no proof a certain $p_z$ covers all $p_x$.

# Variational Autoencoders

The idea is that $p_z$ should be conditioned on $p_x$. Rather that sampling from $p_z$ we should sample from $p_{z|x}$. To design this density VAEs propose a second NN such that

$$
p_{z|x} = \mathcal{N}(z;\mu_{\phi}(x),\sigma_{\phi}(x))
$$

$$
p_{x|z} = \mathcal{N}(x;\mu_{\theta}(z),\sigma_{\theta}(z))
$$


# Generative Adversarial Neural Networks

GANs aim to better design the loss function with respect to VAEs. The Discriminator $D: \tilde x \rightarrow [0,1]$ learns to classify samples either originating from p_x or not.

# Normalizing Flows

NFs aim to constraint the design of f, such that using if auto-regressively applied is should in theory converge to $p_x$. This circles back around the Central Limit Theorem, since a NNs are already auto-regressive functions - however, their hidden state space is not constrained hardly. NFs, assume the hidden state space is structured (same dimensionality for all layers), so that the *Change of Variables Theorem* can be applied at each step.



# Denoising Diffusion Probabilistic Models

DDPM is a generative model: that is a sampling method for intractable distributions from which we only have a example draws. It employes a neural network to estimate the noise that needs to be removed from the image, at for different noise levels. The theory that backs this up, is the inverse processes of adding the same noise (with known distrubtion) to a clean image, iteratevely. After sufficient iterations, the outcome is a plain noise image of known distribution. The inverse processes requires drawing from the noise distrubtion (easy to do) and removing the noise step-by-step.

# Consistency Models

DDPM requires a lot of iterative steps (~10K), which is costly. Consistency models, formulated the iterative processes as the solution of a DE from which solving the DE can be done more effectively by any method in the bibliography.

# Distillation of DDPMs

Forumlating diffusion DE and using solvers, suffers the usual local truncation errors, which add up after multiple steps. To bypass that, Distillation methods employ a  learn a model to solve that DE in fewer steps. That improves sampling the sampling time of DDPMs.


