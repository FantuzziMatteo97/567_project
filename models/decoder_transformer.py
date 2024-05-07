import keras_nlp
import tensorflow as tf

from keras import layers, Model
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from models.base_model import BaseModel
from models.transformer_building_blocks import Decoder

@tf.keras.utils.register_keras_serializable()
def average_mse(label, pred):
    mse = tf.keras.losses.mean_squared_error(label, pred)
    average_mse = tf.reduce_mean(mse)
    return average_mse
    
class DecoderTransformer(BaseModel):

    def __init__(self, scaler, input_shape, optimizer='adam', loss=average_mse):
        super().__init__(scaler, input_shape, optimizer, loss)

    def build_model(self):
        d_k = 4
        d_v = 4
        n_heads = 1
        d_ff = 4

        sequence_length = self.input_shape[0]
        d_model = self.input_shape[1]
        in_seq = layers.Input(shape=self.input_shape)

        # add positional embedding to input sequence
        position_embeddings = keras_nlp.layers.PositionEmbedding(sequence_length)(in_seq)
        x = in_seq + position_embeddings

        # add decoder block(s)s
        # each decoder block consists of:
        #   - masked multi-headed attention
        #   - normalization
        #   - feed-forward network with non-linear activation
        #   - dropout
        #   - normalization
        decoder_block1 = Decoder(d_k, d_v, n_heads, d_ff, d_model)
        x = decoder_block1((x, x, x))

        # use output of decoder block(s) to do regression
        x = layers.GlobalAveragePooling1D(data_format='channels_first')(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.1)(x)
        out = layers.Dense(sequence_length, activation='linear')(x)

        self.model = Model(inputs=in_seq, outputs=out)

    def train(self, inputs_train, targets_train, inputs_val, targets_val, epochs=50, batch_size=32):
        checkpoint_filepath = f'.best_{self.__class__.__name__.lower()}.keras'
        checkpoint = ModelCheckpoint(
            filepath=checkpoint_filepath,
            monitor='val_loss',
            mode='min',
            save_best_only=True)

        self.model.compile(optimizer='adam', loss=average_mse)

        history = self.model.fit(x=inputs_train, y=targets_train, batch_size=batch_size, epochs=epochs, validation_data=(inputs_val, targets_val), callbacks=[checkpoint])

        self.model = load_model(checkpoint_filepath)
        return history
    
    def predict(self, X):
        return self.scaler.inverse_transform(self.model.predict(X)[:,-1].reshape(-1, 1))