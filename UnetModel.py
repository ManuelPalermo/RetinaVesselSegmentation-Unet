
from keras.models import *
from keras.layers import *
from keras.optimizers import Adam, SGD
from keras import backend as K


class Unet:
    def __init__(self, pretrained_weights=None, input_size=(160, 160, 1)):
        self.model = self.unet(pretrained_weights, input_size)


    def summary(self):
        self.model.summary()


    def channel_attention(self, input_feature, ratio=8):
        channel = input_feature._keras_shape[-1]

        shared_layer_one = Dense(channel // ratio,
                                 activation='relu',
                                 kernel_initializer='he_normal',
                                 use_bias=True,
                                 bias_initializer='zeros')
        shared_layer_two = Dense(channel,
                                 kernel_initializer='he_normal',
                                 use_bias=True,
                                 bias_initializer='zeros')

        avg_pool = GlobalAveragePooling2D()(input_feature)
        avg_pool = Reshape((1, 1, channel))(avg_pool)
        avg_pool = shared_layer_one(avg_pool)
        avg_pool = shared_layer_two(avg_pool)

        max_pool = GlobalMaxPooling2D()(input_feature)
        max_pool = Reshape((1, 1, channel))(max_pool)
        max_pool = shared_layer_one(max_pool)
        max_pool = shared_layer_two(max_pool)

        cbam_feature = Add()([avg_pool, max_pool])
        cbam_feature = Activation('sigmoid')(cbam_feature)
        return multiply([input_feature, cbam_feature])


    def spatial_attention(self, input_feature, kernel_size=7):
        avg_pool = Lambda(lambda x: K.mean(x, axis=-1, keepdims=True))(input_feature)
        max_pool = Lambda(lambda x: K.max(x, axis=-1, keepdims=True))(input_feature)
        concat = Concatenate(axis=-1)([avg_pool, max_pool])
        cbam_feature = Conv2D(filters=1,
                              kernel_size=kernel_size,
                              strides=1,
                              padding='same',
                              activation='sigmoid',
                              kernel_initializer='he_normal',
                              use_bias=False)(concat)
        return multiply([input_feature, cbam_feature])


    def cbam_block(self, cbam_feature, ratio=2):
        # https://github.com/kobiso/CBAM-keras/blob/master/models/attention_module.py
        cbam_feature = self.channel_attention(cbam_feature, ratio)
        cbam_feature = self.spatial_attention(cbam_feature)
        return cbam_feature


    def unet(self, pretrained_weights=None, input_size=(128, 128, 1)):
        def dice_coef(y_true, y_pred, smooth=1):
            intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
            return (2. * intersection + smooth) / (
                    K.sum(K.square(y_true), -1) + K.sum(K.square(y_pred), -1) + smooth)

        def dice_coef_loss(y_true, y_pred):
            return 1 - dice_coef(y_true, y_pred)

        inputs = Input(input_size)
        
        d1 = Conv2D(32, 3, activation='relu', padding='same')(inputs)
        d1 = BatchNormalization()(d1)
        d1 = Conv2D(32, 3, activation='relu', padding='same')(d1)
        d1 = BatchNormalization()(d1)
        d1 = SpatialDropout2D(0.1)(d1)
        d1 = self.spatial_attention(d1)

        d2 = MaxPooling2D(pool_size=(2, 2))(d1)
        d2 = Conv2D(64, 3, activation='relu', padding='same')(d2)
        d2 = BatchNormalization()(d2)
        d2 = Conv2D(64, 3, activation='relu', padding='same')(d2)
        d2 = BatchNormalization()(d2)
        d2 = SpatialDropout2D(0.1)(d2)
        d2 = self.spatial_attention(d2)

        d3 = MaxPooling2D(pool_size=(2, 2))(d2)
        d3 = Conv2D(128, 3, activation='relu', padding='same')(d3)
        d3 = BatchNormalization()(d3)
        d3 = Conv2D(128, 3, activation='relu', padding='same')(d3)
        d3 = BatchNormalization()(d3)
        d3 = SpatialDropout2D(0.25)(d3)
        d3 = self.spatial_attention(d3)

        d4 = MaxPooling2D(pool_size=(2, 2))(d3)
        d4 = Conv2D(256, 3, activation='relu', padding='same')(d4)
        d4 = BatchNormalization()(d4)
        d4 = Conv2D(256, 3, activation='relu', padding='same')(d4)
        d4 = BatchNormalization()(d4)
        d4 = SpatialDropout2D(0.4)(d4)
        d4 = self.spatial_attention(d4)

        u3 = UpSampling2D(size=(2, 2))(d4)
        u3 = Conv2D(128, 2, activation='relu', padding='same')(u3)
        u3 = BatchNormalization()(u3)
        u3 = concatenate([d3, u3], axis=-1)
        u3 = Conv2D(128, 3, activation='relu', padding='same')(u3)
        u3 = BatchNormalization()(u3)
        u3 = Conv2D(128, 3, activation='relu', padding='same')(u3)
        u3 = BatchNormalization()(u3)
        u3 = self.spatial_attention(u3)

        u2 = UpSampling2D(size=(2, 2))(u3)
        u2 = Conv2D(64, 2, activation='relu', padding='same')(u2)
        u2 = BatchNormalization()(u2)
        u2 = concatenate([d2, u2], axis=-1)
        u2 = Conv2D(64, 3, activation='relu', padding='same')(u2)
        u2 = BatchNormalization()(u2)
        u2 = Conv2D(64, 3, activation='relu', padding='same')(u2)
        u2 = BatchNormalization()(u2)
        u2 = self.spatial_attention(u2)

        u1 = UpSampling2D(size=(2, 2))(u2)
        u1 = Conv2D(32, 2, activation='relu', padding='same')(u1)
        u1 = BatchNormalization()(u1)
        u1 = concatenate([d1, u1], axis=-1)
        u1 = Conv2D(32, 3, activation='relu', padding='same')(u1)
        u1 = BatchNormalization()(u1)
        u1 = Conv2D(32, 3, activation='relu', padding='same')(u1)
        u1 = BatchNormalization()(u1)
        u1 = self.spatial_attention(u1)

        out = Conv2D(3, 3, activation='relu', padding='same')(u1)
        out = BatchNormalization()(out)
        out = self.spatial_attention(out)

        out = Conv2D(1, 1, activation='sigmoid')(out)

        model = Model(inputs=inputs, outputs=out)

        # optimizer = Adam(lr=0.001)
        optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["acc", dice_coef])

        if pretrained_weights is not None:
            model.load_weights(pretrained_weights)

        return model


    def get_model(self):
        return self.model
