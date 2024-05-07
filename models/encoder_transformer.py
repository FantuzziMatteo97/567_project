from keras import layers, Model
from models.base_model import BaseModel
from models.transformer_building_blocks import Encoder, Time2Vector

class EncoderTransformer(BaseModel):
    def __init__(self, scaler, input_shape, optimizer='adam', loss='mean_squared_error'):
        super().__init__(scaler, input_shape, optimizer, loss)

    def build_model(self):
        d_k = 8
        d_v = 8
        n_heads = 2
        d_ff = 8

        timesteps = self.input_shape[0]
        time_embedding = Time2Vector(timesteps)

        # input sequence is of shape (timesteps, num_features)
        in_seq = layers.Input(shape=self.input_shape)

        # calculate time/positional embedding of shape (timesteps, 2)
        # uses Time2Vec that computes 'linear' and 'periodic' components of time embedding
        x = time_embedding(in_seq)

        # concatenate time embedding with input sequence
        x = layers.Concatenate(axis=-1)([in_seq, x])
        d_model = x.shape[2]

        # add encoder block(s)
        # each encoder block consists of:
        #   - multi-headed attention
        #   - dropout
        #   - normalization
        #   - feed-forward network with non-linear activation
        #   - feed-forward network without activation
        #   - dropout
        #   - normalization
        attn_layer1 = Encoder(d_k, d_v, n_heads, d_ff, d_model)
        x = attn_layer1((x, x, x))

        # use output of encoder block(s) to do regression prediction
        x = layers.GlobalAveragePooling1D(data_format='channels_first')(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.1)(x)
        out = layers.Dense(1, activation='linear')(x)

        self.model = Model(inputs=in_seq, outputs=out)