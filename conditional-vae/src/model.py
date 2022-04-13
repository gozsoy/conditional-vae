import numpy as np
import tensorflow as tf

# processes input image and flattens feature maps
def get_conditional_encoder1():
    inputs = tf.keras.Input(shape = (28,28,1))
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu')(inputs)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)

    return tf.keras.Model(inputs=inputs,outputs=[x])

# gets flattened feature maps, and one hot label vector and outputs mu and rho
def get_conditional_encoder2(latent_dim,input_size):
    inputs = tf.keras.Input(shape = (input_size + 10,))
    mu = tf.keras.layers.Dense(units=latent_dim)(inputs)
    rho = tf.keras.layers.Dense(units=latent_dim)(inputs)

    return  tf.keras.Model(inputs=inputs,outputs=[mu,rho])

# classical vae decoder
def get_conditional_decoder(latent_dim):
    z = tf.keras.Input(shape = (latent_dim+10,))
    x= tf.keras.layers.Dense(units=7*7*32, activation='relu')(z)
    x=tf.keras.layers.Reshape(target_shape=(7, 7, 32))(x)
    x=tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same',activation='relu')(x)
    x=tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same',activation='relu')(x)
    decoded_img=tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=3, strides=1, padding='same')(x)
    
    return tf.keras.Model(inputs=z,outputs=[decoded_img])

class Conditional_VAE(tf.keras.Model):
    def __init__(self,latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder_block1 = get_conditional_encoder1()
        # 2304 is specific to conv layers, not the best practice to hardcode it
        self.encoder_block2 = get_conditional_encoder2(latent_dim=latent_dim,input_size=2304)
        self.decoder_block = get_conditional_decoder(latent_dim)

    def call(self,img,labels):
        # encoder q(z|x,y)
        enc1_output = self.encoder_block1(img)
        # concat feature maps and one hot label vector
        img_lbl_concat = np.concatenate((enc1_output,labels),axis=1)
        z_mu,z_rho = self.encoder_block2(img_lbl_concat)

        # sampling
        epsilon = tf.random.normal(shape=z_mu.shape,mean=0.0,stddev=1.0)
        z = z_mu + tf.math.softplus(z_rho) * epsilon

        # decoder p(x|z,y)
        z_lbl_concat = np.concatenate((z,labels),axis=1)
        decoded_img = self.decoder_block(z_lbl_concat)

        return z_mu,z_rho,decoded_img