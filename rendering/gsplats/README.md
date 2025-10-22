# Gaussian splats

Improves NeRFs rendering time using gaussian modeling at the pixel level.

$$
\pmb \mu' = P V \pmb \mu
$$

$$
\Sigma' = J V \Sigma V^T J^T
$$

$V$: camera matrix \
$P$: raster projection \
$J$: Jacobian of V 

and the final color of a pixel x is

$$
C(\pmb x) = \sum_i c_i \alpha_i(\pmb(x)) \prod_j (1-\alpha_j(\pmb x))
$$

$$
\alpha_i(\pmb x) = \sigma_i exp(-\dfrac{1}{2}(\pmb x - \pmb \mu_i')^T \Sigma_i' ((\pmb x - \pmb \mu_i')^))
$$

$\pmb x$: pixel coordinates \
$c_i$: color of pixel i \
$\sigma_i$: opacity of pixel i \
$\pmb \mu_i'$: mean \
$\Sigma_i'$: covariance

\* Note that C() in NeRF is a function rays, while in G-Splats it is a function of pixels.

G-Splats render 3D Gaussians (aka rasteraziation) while NeRF project by ray marching.