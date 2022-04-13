import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")


def generate_conditioned_digits(model,dataset_mean,dataset_std):
    n = 20  # number of generation per digit

    image = np.zeros((28*10, 28*n))

    for digit in range(10):
        for gen_idx in range(n):

            label_onehot = np.zeros((1,10))
            label_onehot[:,digit] = 1.0

            z = tf.random.normal(shape=(1,model.encoder_block2.output[0].shape[1]),mean=0.0,stddev=1.0)
            z_lbl_concat = np.concatenate((z,label_onehot),axis=1)
            preds = model.decoder_block(z_lbl_concat)

            generated_digit = tf.reshape(preds[0],[-1,28,28])
            generated_digit = (generated_digit * dataset_std) + dataset_mean
            generated_digit = generated_digit * 255.
            image[digit * 28: (digit + 1) * 28,
                gen_idx * 28: (gen_idx + 1) * 28] = generated_digit.numpy()

    fig = plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap='Greys_r')
    plt.axis('Off')
    plt.savefig('results/generated_conditoned_digits.png')

    return