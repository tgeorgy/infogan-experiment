# InfoGAN experiment on MNIST dataset

This is a simplified implementation of the experiment. It is based on two projects:  
1. The experiment details were taken from this paper https://arxiv.org/abs/1606.03657 and corresponding project https://github.com/openai/InfoGAN with same changes.  
2. And DCGAN torch implementation details were taken from https://github.com/soumith/dcgan.torch, thanks to its simplicity.

## Loss function
In original paper they maximize aproximated mutual information (MI) between latent parameters and generated images. Here I do the same thing, but I don't evaluate complete MI and deriving gradients from it. Instead, I keep the only part which is relevent to computing gradients.

### 1. Continues latent parameters loss
For normally distributed parameters I assume that the variance is fixed (just like it is done in original experiment). So it is enough to reduce MSE between latent and estimated parameters. The resulting loss is proportional to MI, so the gradinet is same.

### 2. Categorical latent parameters loss
Same thing is here - the result is equevalent to minimizing negative log-likelihood.

### Random sample
![random_sample](images/random_sample.png "Random sample")

### Symbol incline
Changing c1 from -2 to 2 in each column and activating different categorical parameter per row:
![symbol_incline](images/symbol_incline.png "Symbol incline")

### Stroke width
Changing c2 from -2 to 2:
![stroke_width](images/stroke_width.png "Stroke width")
