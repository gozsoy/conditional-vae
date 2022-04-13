import os
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE,Isomap
from sklearn.model_selection import StratifiedKFold
import warnings

warnings.filterwarnings("ignore")


def visualize_latent_space(z_mu_list,label_list):

    # cannot work on full training samples so get a small and stratified portion
    skf = StratifiedKFold(n_splits=20)
    for _, test_index in skf.split(z_mu_list, label_list):
        mini_z_mu_list, mini_label_list = z_mu_list[test_index], label_list[test_index]
        break

    tsne = TSNE(n_jobs=-1,learning_rate='auto',init='pca')
    embedded_mus = tsne.fit_transform(mini_z_mu_list)

    df = pd.DataFrame({'dim1':embedded_mus[:,0],'dim2':embedded_mus[:,1],'label':mini_label_list})
    fig = plt.figure(figsize=(8, 8))
    sns_plot = sns.scatterplot(data=df, x="dim1", y="dim2", hue="label",palette='bright')
    sns_plot.figure.savefig("results/tsne_transformed_latent_space.png") 

    return


def generate_images(model,dataset_mean,dataset_std,temp_x_test=None):

    def reshape_and_save(preds,plt_name):
        preds = tf.reshape(preds,[-1,28,28])
        # destandardization
        preds = (preds * dataset_std) + dataset_mean
        preds = preds * 255.

        fig = plt.figure(figsize=(5, 5))
        for i in range(preds.shape[0]):
            plt.subplot(4, 4, i + 1)
            plt.imshow(preds[i, :, :], cmap='gray')
            plt.axis('off')

        plt.savefig(plt_name)
        return
  
    if temp_x_test is None:  # sample from prior
        z = tf.random.normal(shape=(16,model.encoder_block.output[0].shape[1]),mean=0.0,stddev=1.0)
        preds = model.decoder_block(z)
        reshape_and_save(preds,plt_name='results/generated_new_images.png')

    else:  # encode and decode given test samples
        _,_,preds = model(temp_x_test)
        reshape_and_save(preds,plt_name='results/generated_test_images.png')

        if not os.path.isfile('results/original_test_images.png'):
            reshape_and_save(temp_x_test,plt_name='results/original_test_images.png')

    return


# slightly modified version of https://www.tensorflow.org/tutorials/generative/cvae#display_a_2d_manifold_of_digits_from_the_latent_space
def plot_latent_images(model,dataset_mean,dataset_std):
    n = 20
    if model.latent_dim != 2:
        raise Exception('latent space is not 2-D.')
    norm = tfp.distributions.Normal(0, 1)
    grid_x = norm.quantile(np.linspace(0.05, 0.95, n))
    grid_y = norm.quantile(np.linspace(0.05, 0.95, n))

    image = np.zeros((28*n, 28*n))

    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z = np.array([[xi, yi]])
            x_decoded = model.decoder_block(z)

            digit = tf.reshape(x_decoded[0],[-1,28,28])
            digit = (digit * dataset_std) + dataset_mean
            digit = digit * 255.
            image[i * 28: (i + 1) * 28,
                j * 28: (j + 1) * 28] = digit.numpy()

    fig = plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap='Greys_r')
    plt.axis('Off')
    plt.savefig('results/digit_manifold_2d.png')

    return