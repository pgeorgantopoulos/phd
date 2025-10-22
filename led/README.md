# Adaptive Learning of Effective Dynamics (AdaLED)

Kičić, I., Vlachas, P. R., Arampatzis, G., Chatzimanolakis, M., Guibas, L., & Koumoutsakos, P. (2023). Adaptive learning of effective dynamics for online modeling of complex systems. Computer Methods in Applied Mechanics and Engineering, 415, 116204.

<https://github.com/cselab/adaled>

## Learning Effective Dynamics (LED)

* ref:24

"Learning Effective Dynamics"

Vlachas, P. R., Arampatzis, G., Uhler, C., & Koumoutsakos, P. (2022). Multiscale simulations of complex systems by learning their effective dynamics. Nature Machine Intelligence, 4(4), 359-366.

$$
s_{n} = \mathcal{F}(s_{n}) \in R^{d_s}
$$

Approximation of $s_n$ is done with an Autoencoder
$$
z_{n} = \mathcal{E}(s_{n}) \in R^{d_z} \\
\tilde{s}_{n} = \mathcal{D}(z_{n})\\
d_s >> d_z
$$

trained with

$$
\underset{\mathcal{D},\mathcal{E}}{argmin}\ || s_{n} - \mathcal{D}(\mathcal{E}(s_{n})) ||_2^2
$$

Evolution of $z_n$ is done with an RNN

$$
h_{n} = \mathcal{H}(z_{n},h_{n-1})\\
\tilde{z}_{n} = \mathcal{R}(h_{n-1})
$$

trained with

$$
\underset{\mathcal{R},\mathcal{H}}{argmin}\ || z_{n} - \mathcal{R}(\mathcal{H}(z_{n},h_{n-1})) ||_2^2
$$

or my maximizing the likelihood of the observed state

$$
\underset{\mathcal{D},\mathcal{E}}{argmax}\ log\ p(s_{n}|\mathcal{D}(\mathcal{E}(s_{n})))
$$

Adaptive LED (AdaLED) contribute a dynamical state model

$$
s_{n} = \mathcal{F}(s_{n},f_{n}) \in R^{d_s}
$$

therefore the RNN is adapted

$$
h_{n} = \mathcal{H}(z_{n},h_{n-1},f_{n})\\
\tilde{z}_{n} = \mathcal{R}(h_{n-1})
$$


## Other Coarse/Fine-grain solvers

* ref:12

"Equation-Free Framework"

Bar-Sinai, Y., Hoyer, S., Hickey, J., & Brenner, M. P. (2019). Learning data-driven discretizations for partial differential equations. Proceedings of the National Academy of Sciences, 116(31), 15344-15349.

* ref:14
  
"Heterogenous Multiscale Methods"

Weinan, E., Engquist, B., Li, X., Ren, W., & Vanden-Eijnden, E. (2007). Heterogeneous multiscale methods: a review. Communications in computational physics, 2(3), 367-450.

* ref:15

"FLow AVeraged integratOR (FLAVOR)"

Tao, M., Owhadi, H., & Marsden, J. E. (2010). Nonintrusive and structure preserving multiscale integration of stiff ODEs, SDEs, and Hamiltonian systems with hidden slow dynamics via flow averaging. Multiscale Modeling & Simulation, 8(4), 1269-1324.

## Interesting Applications

* ref:27

"Latent Evolution of Partial Differential Equations"

Wu, T., Maruyama, T., & Leskovec, J. (2022). Learning to accelerate partial differential equations via latent global evolution. Advances in Neural Information Processing Systems, 35, 2240-2253.

* ref:43

"Reduced Order Model - Uncertainty Quantification"
  
Galbally, D., Fidkowski, K., Willcox, K., & Ghattas, O. (2010). Non‐linear model reduction for uncertainty quantification in large‐scale inverse problems. International journal for numerical methods in engineering, 81(12), 1581-1608.

 