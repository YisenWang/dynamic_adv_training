from __future__ import absolute_import
from __future__ import print_function
import keras.backend as K
from keras.layers import Input, Conv2D, Dense, MaxPooling2D, Flatten, BatchNormalization, Activation
from resnet import ResNet, WideResNet
from keras.models import Model
from keras.regularizers import l2

NUM_CLASS = {'mnist': 10, 'cifar-10': 10}


def get_model(dataset='mnist', input_tensor=None, input_shape=None, n_class=-1, softmax=True):
    """
    Define the models used for different datasets.
    """
    if n_class == -1:
        n_class = NUM_CLASS[dataset]

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_shape):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    if dataset == 'mnist':
        x = Conv2D(32, (3, 3), padding='same', name='conv1')(img_input)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(x)

        x = Conv2D(64, (3, 3), padding='same', name='conv2')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(x)

        x = Flatten()(x)

        x = Dense(128, kernel_initializer="he_normal", name='features')(x)
        x = Activation('relu')(x)

        x = Dense(n_class, name='logits')(x)

    elif dataset == 'cifar-10':
        x = WideResNet(depth=5, n_class=n_class, input_tensor=img_input)
        # # Block 1
#         x = Conv2D(64, (3, 3), padding='same', name='block1_conv1')(img_input)
#         x = BatchNormalization()(x)
#         x = Activation('relu')(x)
#         x = Conv2D(64, (3, 3), padding='same', name='block1_conv2')(x)
#         x = BatchNormalization()(x)
#         x = Activation('relu')(x)
#         x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool1')(x)

#         # Block 2
#         x = Conv2D(128, (3, 3), padding='same', name='block2_conv1')(x)
#         x = BatchNormalization()(x)
#         x = Activation('relu')(x)
#         x = Conv2D(128, (3, 3), padding='same', name='block2_conv2')(x)
#         x = BatchNormalization()(x)
#         x = Activation('relu')(x)
#         x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool1')(x)

#         # Block 3
#         x = Conv2D(196, (3, 3), padding='same', name='block3_conv1')(x)
#         x = BatchNormalization()(x)
#         x = Activation('relu')(x)
#         x = Conv2D(196, (3, 3), padding='same', name='block3_conv2')(x)
#         x = BatchNormalization()(x)
#         x = Activation('relu')(x)
#         x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool1')(x)

#         x = Flatten(name='flatten')(x)

#         x = Dense(256, kernel_initializer="he_normal",
#                   kernel_regularizer=l2(0.01),
#                   bias_regularizer=l2(0.01),
#                   name='features')(x)
#         x = BatchNormalization()(x)
#         x = Activation('relu')(x)

#         x = Dense(n_class, name='logits')(x)
    else:
        # total layers = depth*6 + 2
        x = ResNet(depth=5, n_class=n_class, input_tensor=img_input)

    if softmax:
        x = Activation('softmax')(x)
    model = Model(inputs=img_input, outputs=x)
    return model



