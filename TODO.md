# TODO
[X] switch to algorithm 2 (RandPDLowRankApprox) in https://arxiv.org/pdf/2211.08597v2.pdf
[ ] use larger batch size (and gradient accumulation) to stablize the largest singular value of the Hessian estimation
[ ] update the Hessian less frequently to save time
[ ] calculate the spectral norm of the difference between successive Hessian estimations (through power iteration?)
[ ] do a grid search over log(rho); the suggested rho = 0.001 is for the convex setting

# NOTE
* `jax.remat` (rematerialize) prevents JAX from saving the inner activations (for reverse-mode AD), which seems helpful
* `jax.ad_checkpoint.print_saved_residuals` prints saved activations for debugging
* https://jax.readthedocs.io/en/latest/notebooks/autodiff_remat.html
