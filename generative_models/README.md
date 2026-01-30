# Denoising Diffusion Probabilistic Models

DDPM is a generative model: that is a sampling method for intractable distributions from which we only have a example draws. It employes a neural network to estimate the noise that needs to be removed from the image, at for different noise levels. The theory that backs this up, is the inverse processes of adding the same noise (with known distrubtion) to a clean image, iteratevely. After sufficient iterations, the outcome is a plain noise image of known distribution. The inverse processes requires drawing from the noise distrubtion (easy to do) and removing the noise step-by-step.

# Consistency Models

DDPM requires a lot of iterative steps (~10K), which is costly. Consistency models, formulated the iterative processes as the solution of a DE from which solving the DE can be done more effectively by any method in the bibliography.

# Distillation of DDPMs

Forumlating diffusion DE and using solvers, suffers the usual local truncation errors, which add up after multiple steps. To bypass that, Distillation methods employ a  learn a model to solve that DE in fewer steps. That improves sampling the sampling time of DDPMs.


