# simple-vae
Tensorflow implementation of Variational Autoencoder concept

### Model
Both encoder and decoder consist of two fully connected hidden layers. Various ways of VAE implementation is possible in TF, but I computed both losses after forward pass, which means model provides both encoder and decoder outputs.
Training is done on MNIST training set, using Adam optimizer with learning rate 1e-3 for maximum of 15 epochs. Depending on latent channel capacity, overfitting is possible, so early stopping via visual inspection is advised.
I experimented with different formulations of re-parametrization trick and found that $z = \mu + \log(1+\exp(\rho)) \odot  \epsilon$ is more stable than $z = \mu + \sigma  \odot  \epsilon$, although both produce nice outcomes. 

### Usage
```
cd src
python main.py
```
VAE settings ($\beta$ and latent dimension) can easily be modified inside main.py.


### Experiments

In experiments below, latent space visualization is obtained by TSNE on encoder outputted means for each sample in the training set.

#### KL-Reconstruction Loss Trade-off
$\beta$ = 0.0, optimising only reconstruction loss: no kl divergence so latent space idea is useless because encoder can put each digit in the space separate places with very small variations. test images can perfectly reconstructed even if they are not used in training. (2d_fold, latent_space, new_samples, test samples) comment on the plots.
$\beta$ = 200.0, optimising only kl loss: without even caring reconstruction quality, all samples will be encoded to have unit gaussian parameters, thus in the latent space no label(or similarity) based clustering will be observed. comment on the plots.
$\beta$ = 12.0, both clustering nature of reconstruction loss and dense packing nature of kl loss observed.