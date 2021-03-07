from tensorflow.python.keras import layers
from tensorflow.python.keras import backend
from tensorflow.python.keras.engine import training
from tensorflow.python.keras.utils import layer_utils
''' part of code from https://github.com/taki0112/Densenet-Tensorflow'''


def bottleneck_layer(x, name, filters, dropout_rate):
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu', name=name + '_relu1')(x)
    x = layers.Conv2D(4 * filters, 1, padding='same', name=name + '_conv1')(x)
    x = layers.Dropout(rate=dropout_rate)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu', name=name + '_relu2')(x)
    x = layers.Conv2D(filters, 1, padding='same', name=name + '_conv2')(x)
    x = layers.Dropout(rate=dropout_rate)(x)
    return x


def transition_block(x, name, dropout_rate, reduction=0.5):
    x = layers.BatchNormalization(name=name + '_bn')(x)
    x = layers.Activation('relu', name=name + '_relu')(x)
    x = layers.Conv2D(int(x.get_shape().as_list()[-1] * reduction),
                      1,
                      padding='same',
                      name=name + '_conv')(x)
    x = layers.Dropout(rate=dropout_rate)(x)
    x = layers.AveragePooling2D(strides=2, name=name + '_pool')(x)
    return x


def dense_block(x, blocks, filters, dropout_rate, name):
    for i in range(blocks):
        x1 = bottleneck_layer(x,
                              filters=filters,
                              dropout_rate=dropout_rate,
                              name=name + '_bottleN_' + str(i))
        x = layers.Concatenate(axis=3)([x, x1])
    return x


def DenseNet(blocks,
             classes,
             filters,
             dropout_rate,
             include_top=True,
             weights=None,
             input_tensor=None,
             input_shape=None,
             pooling=None,
             classifier_activation=None):
    # Determine proper input shape
    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)))(x)
    x = layers.Conv2D(2 * filters,
                      3,
                      strides=2,
                      padding='same',
                      name='conv1/conv')(img_input)
    # x = layers.BatchNormalization(axis=bn_axis,
    #                               epsilon=1.001e-5,
    #                               name='conv1/bn')(x)
    # x = layers.Activation('relu', name='conv1/relu')(x)
    # x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=2, name='pool1')(x)

    # x = dense_block(x,
    #                 blocks=6,
    #                 filters=filters,
    #                 dropout_rate=dropout_rate,
    #                 name='dense_1')
    # x = transition_block(x, dropout_rate=dropout_rate, name='trans_2')
    # x = dense_block(x,
    #                 blocks=12,
    #                 filters=filters,
    #                 dropout_rate=dropout_rate,
    #                 name='dense_2')
    # x = transition_block(x, dropout_rate=dropout_rate, name='trans_3')
    # x = dense_block(x,
    #                 blocks=24,
    #                 filters=filters,
    #                 dropout_rate=dropout_rate,
    #                 name='dense_3')
    # x = transition_block(x, dropout_rate=dropout_rate, name='trans_4')
    # x = dense_block(x,
    #                 blocks=16,
    #                 filters=filters,
    #                 dropout_rate=dropout_rate,
    #                 name='dense_4')

    for i in range(blocks):
        x = dense_block(x,
                        blocks=6,
                        filters=filters,
                        dropout_rate=dropout_rate,
                        name='dense_' + str(i))
        x = transition_block(x,
                             dropout_rate=dropout_rate,
                             name='trans_' + str(i))
    x = dense_block(x,
                    blocks=12,
                    filters=filters,
                    dropout_rate=dropout_rate,
                    name='dense_final')

    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu', name='relu')(x)

    if include_top:
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        x = layers.Flatten()(x)
        x = layers.Dense(classes,
                         activation=classifier_activation,
                         name='predictions')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D(name='max_pool')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = layer_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = training.Model(inputs, x, name='NET.model')

    return model
