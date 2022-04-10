import tensorflow as tf

def get_encoder(latent_dim):
    inputs = tf.keras.Input(shape = (784,))
    x = tf.keras.layers.Dense(units=500, activation='relu')(inputs)
    x = tf.keras.layers.Dense(units=120, activation='relu')(x)
    mu = tf.keras.layers.Dense(units=latent_dim)(x)
    rho = tf.keras.layers.Dense(units=latent_dim)(x)
    Encoder = tf.keras.Model(inputs=inputs,outputs=[mu,rho])
    
    return Encoder

def get_decoder(latent_dim):
    z = tf.keras.Input(shape = (latent_dim,))
    x = tf.keras.layers.Dense(units=120, activation='relu')(z)
    x = tf.keras.layers.Dense(units=500, activation='relu')(x)
    decoded_img = tf.keras.layers.Dense(units=784)(x)
    Decoder = tf.keras.Model(inputs=z,outputs=[decoded_img])
    
    return Decoder

class VAE(tf.keras.Model):
    def __init__(self,latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder_block = get_encoder(latent_dim)
        self.decoder_block = get_decoder(latent_dim)

    def call(self,img):
        z_mu,z_rho = self.encoder_block(img)

        epsilon = tf.random.normal(shape=z_mu.shape,mean=0.0,stddev=1.0)
        z = z_mu + tf.math.softplus(z_rho) * epsilon

        decoded_img = self.decoder_block(z)

        return z_mu,z_rho,decoded_img