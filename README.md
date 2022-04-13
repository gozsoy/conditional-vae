# simple-vae
Tensorflow implementation of Variational Autoencoder concept

### Usage
```
cd src
python main.py
```
VAE settings (&beta; and latent dimension) can easily be modified inside main.py.


### Experiments

In experiments below, latent space visualization is obtained by TSNE on encoder outputted means for each sample in the training set.

#### KL Loss vs. Reconstruction Loss
+ &beta; = 0.0 (optimising only reconstruction loss): Latent space idea is not used because encoder can put each sample in separate places with punctual variations. Test image reconstruction quality is high even if they are not used in training, but generation ability is very low.
<p align="center">
<img src="./results/tsne_transformed_latent_space_beta0.png" height="270px">
<img src="./results/generated_test_images_beta0.png" height="270px">
<img src="./results/generated_new_images_beta0.png" height="270px">
</p>
<p align="center">
Left: 2d latent space, center: reconstructed test set images, right: newly generated images
</p>

+ &beta; = 200.0 (optimising only kl loss): Without reconstruction pressure, all samples will have unit gaussian parameters, thus in the latent space no label(or similarity)-based clustering will be observed. Test image reconstruction quality, and generation ability are very low.
<p align="center">
<img src="./results/tsne_transformed_latent_space_beta200.png" height="270px">
<img src="./results/generated_test_images_beta200.png" height="270px">
<img src="./results/generated_new_images_beta200.png" height="270px">
</p>
<p align="center">
Left: 2d latent space, center: reconstructed test set images, right: newly generated images
</p>

+ &beta; = 12.0 (optimising both losses): Both clustering nature of reconstruction loss and dense packing nature of kl loss observed.
<p align="center">
<img src="./results/tsne_transformed_latent_space_beta8.png" height="450px">
<img src="./results/digit_manifold_2d_beta12.png" height="450px">
</p>
<p align="center">
Left: 2d latent space, right: digit manifold when latent dimension = 2
</p>

### Model
Both encoder and decoder consist of two fully connected hidden layers. Various ways of VAE implementation is possible in TF, but I computed both losses after forward pass, which means model provides both encoder and decoder outputs.

Training is done on MNIST training set, using Adam optimizer with learning rate 1e-3 for maximum of 15 epochs. Depending on latent channel capacity, overfitting is possible, so early stopping via visual inspection is advised.

I experimented with different formulations of re-parametrization trick and found that z = &mu; + &sigma; &#8857; &epsilon; is less stable than z = &mu; + log(1 + exp(&rho;)) &#8857; &epsilon;, although both produce nice outcomes. 
