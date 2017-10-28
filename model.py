from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    Dense,
    Flatten
)
from keras.layers.advanced_activations import PReLU
from keras.regularizers import l2

from keras.layers.convolutional import (
    Conv2D,
    MaxPooling2D,
    AveragePooling2D
)

from keras.layers.merge import add

from keras.layers.normalization import BatchNormalization

from subpixel import Subpixel

def _bn_prelu (input):
    normalized_input = BatchNormalization(axis = 3)(input) # tf is backend. channel is the thirs axis starting from 0
    return  Activation("relu")(normalized_input)



def _conv_bn(input, **conv_params):
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    conv = Conv2D(filters=filters, kernel_size=kernel_size,
                  strides=strides, padding=padding,
                  kernel_initializer=kernel_initializer,
                  kernel_regularizer=kernel_regularizer)(input)
    normalized_input = BatchNormalization(axis=3)(conv)  # tf is backend. channel is the thirs axis starting from 0

    return normalized_input


def _conv_bn_prelu(input, **conv_params):
    return Activation('relu')(_conv_bn(input,**conv_params))


def basic_block (input):
    bottle_neck =  _conv_bn(_conv_bn_prelu(input, filters = 64, kernel_size = 3 ), filters = 64, kernel_size = 3)

    return add([bottle_neck, input])


def build_model(input_shape,residual_blocks_count = 21):

    input = Input(shape=input_shape)
    conv1 = Conv2D(filters=64, kernel_size = 9, kernel_initializer= "he_normal", padding="same", kernel_regularizer=l2(1.e-4), activation="relu")(input)

    inp = conv1
    for i in range(residual_blocks_count):
        inp = basic_block(inp)

    inp = _conv_bn(inp,filters = 64, kernel_size= 3)

    res_output= add([inp,conv1])

    first_ups = Subpixel(64, (3, 3), 2, activation='relu')(res_output)

    second_ups = Subpixel(64, (3, 3), 2, activation='relu')(first_ups)

    output = Conv2D(filters = 3, kernel_size=9, kernel_initializer= "he_normal", padding= "same", kernel_regularizer=l2(1.e-4))(second_ups)

    model = Model (inputs= input, outputs= output)

    return  model







