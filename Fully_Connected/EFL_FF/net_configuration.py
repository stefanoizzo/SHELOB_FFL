#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Gio Lug 27 10:33:20 2023

@author: Anonymous
"""

# %%
import numpy as np
import tensorflow as tf

# %%

def model_configuration(input_size, output_size=None, model_type='convdense', 
                        conv_reg=None, dense_reg=None):#, dropout=None):
    if isinstance(input_size, int): 
        assert output_size is not None, 'output_size must be specified'
        # assert model_type == 'dense', 'model_type must be dense'
    elif len(input_size) == 2: 
        input_size = (*input_size, 1)
    if conv_reg is not None:
        conv_reg = tf.keras.regularizers.l2(conv_reg)
    if dense_reg is not None:
        dense_reg = tf.keras.regularizers.l2(dense_reg)

    if model_type == 'conv':
        dims = {#'Input': input_size,
                'Block_0': {'Input' : {'input_shape': input_size},
                            'Conv2D_0': {'filters': 8, 'kernel_size': 3, 'use_bias' : False, 'kernel_regularizer': conv_reg},
                            'Conv2D_1': {'filters': 16, 'kernel_size': 3, 'use_bias' : False, 'kernel_regularizer': conv_reg},
                            'Flatten': {},
                            'Dense_0': {'units': np.prod(input_size[:2]), 'use_bias' : False, 'kernel_regularizer':dense_reg}
                            },
                'Block_1': {'Input' : {'input_shape': (np.prod(input_size[:2]),)},
                            'L2_normalization': {'axis': 1},
                            'Reshape': {'target_shape': (*input_size[:2], 1)},
                            'Conv2D_0': {'filters': 8, 'kernel_size': 3, 'use_bias' : False, 'kernel_regularizer': conv_reg},
                            'Conv2D_1': {'filters': 16, 'kernel_size': 3, 'use_bias' : False, 'kernel_regularizer': conv_reg},
                            'Flatten': {},
                            'Dense_0': {'units': np.prod(input_size[:2]), 'use_bias' : False, 'kernel_regularizer':dense_reg}
                            },
                'Block_2': {'Input' : {'input_shape': (np.prod(input_size[:2]),)},
                            'L2_normalization': {'axis': 1},
                            'Reshape': {'target_shape': (*input_size[:2], 1)},
                            'Conv2D_0': {'filters': 8, 'kernel_size': 3, 'use_bias' : False, 'kernel_regularizer': conv_reg},
                            'Conv2D_1': {'filters': 16, 'kernel_size': 3, 'use_bias' : False, 'kernel_regularizer': conv_reg},
                            'Flatten': {},
                            'Dense_0': {'units': np.prod(input_size[:2]), 'use_bias' : False, 'kernel_regularizer':dense_reg}
                            },
                'Block_3': {'Input' : {'input_shape': (np.prod(input_size[:2]),)},
                            'L2_normalization': {'axis': 1},
                            'Reshape': {'target_shape': (*input_size[:2], 1)},
                            'Conv2D_0': {'filters': 8, 'kernel_size': 3, 'use_bias' : False, 'kernel_regularizer': conv_reg},
                            'Conv2D_1': {'filters': 16, 'kernel_size': 3, 'use_bias' : False, 'kernel_regularizer': conv_reg},
                            'Flatten': {},
                            'Dense_0': {'units': np.prod(input_size[:2]), 'use_bias' : False, 'kernel_regularizer':dense_reg}
                            },
                'Block_4': {'Input' : {'input_shape': (np.prod(input_size[:2]),)},
                            'L2_normalization': {'axis': 1},
                            'Reshape': {'target_shape': (*input_size[:2], 1)},
                            'Conv2D_0': {'filters': 8, 'kernel_size': 3, 'use_bias' : False, 'kernel_regularizer': conv_reg},
                            'Conv2D_1': {'filters': 16, 'kernel_size': 3, 'use_bias' : False, 'kernel_regularizer': conv_reg},
                            'Flatten': {},
                            'Dense_0': {'units': np.prod(input_size[:2]), 'use_bias' : False, 'kernel_regularizer':dense_reg}
                            },
                'train_mode': 'conv'
                }
    elif model_type == 'convdense':
        if output_size is None:
            output_size = np.prod(input_size[:2])
        dims = {#'Input': input_size,
                'Block_0': {'Input' : {'input_shape': input_size},
                            'Conv2D_0': {'filters': 8, 'kernel_size': 3, 'use_bias' : False, 'kernel_regularizer': conv_reg},
                            'Conv2D_1': {'filters': 16, 'kernel_size': 3, 'use_bias' : False, 'kernel_regularizer': conv_reg},
                            'Flatten': {},
                            'Dense_0': {'units': np.prod(input_size[:2]), 'use_bias' : False, 'kernel_regularizer':dense_reg}
                            },
                'Block_1': {'Input' : {'input_shape': (np.prod(input_size[:2]),)},
                            'L2_normalization': {'axis': 1},
                            'Reshape': {'target_shape': (*input_size[:2], 1)},
                            'Conv2D_0': {'filters': 8, 'kernel_size': 3, 'use_bias' : False, 'kernel_regularizer': conv_reg},
                            'Conv2D_1': {'filters': 16, 'kernel_size': 3, 'use_bias' : False, 'kernel_regularizer': conv_reg},
                            'Flatten': {},
                            'Dense_0': {'units': np.prod(input_size[:2]), 'use_bias' : False, 'kernel_regularizer':dense_reg}
                            },
                'Block_2': {'Input' : {'input_shape': (output_size,)},
                            'L2_normalization': {'axis': 1},
                            'Dense_0': {'units': output_size, 'use_bias' : False, 'kernel_regularizer':dense_reg},
                            'Dense_1': {'units': output_size, 'use_bias' : False, 'kernel_regularizer':dense_reg},
                            'Dense_2': {'units': output_size, 'use_bias' : False, 'kernel_regularizer':dense_reg},
                            },
                'Block_3': {'Input' : {'input_shape': (output_size,)},
                            'L2_normalization': {'axis': 1},
                            'Dense_0': {'units': output_size, 'use_bias' : False, 'kernel_regularizer':dense_reg},
                            'Dense_1': {'units': output_size, 'use_bias' : False, 'kernel_regularizer':dense_reg},
                            'Dense_2': {'units': output_size, 'use_bias' : False, 'kernel_regularizer':dense_reg},
                            },
                'Block_4': {'Input' : {'input_shape': (output_size,)},
                            'L2_normalization': {'axis': 1},
                            'Dense_0': {'units': output_size, 'use_bias' : False, 'kernel_regularizer':dense_reg},
                            'Dense_1': {'units': output_size, 'use_bias' : False, 'kernel_regularizer':dense_reg},
                            'Dense_2': {'units': output_size, 'use_bias' : False, 'kernel_regularizer':dense_reg},
                            },
                'train_mode': 'conv'
                }
    elif model_type == 'dense':
        dims = {
            'Block_0': {'Input' : {'input_shape': (input_size,)},
                        'Dense_0': {'units': output_size, 'use_bias' : False, 'kernel_regularizer':dense_reg},
                        'Dense_1': {'units': output_size, 'use_bias' : False, 'kernel_regularizer':dense_reg},
                        'Dense_2': {'units': output_size, 'use_bias' : False, 'kernel_regularizer':dense_reg},
                        },
            'Block_1': {'Input' : {'input_shape': (output_size,)},
                        'L2_normalization': {'axis': 1},
                        'Dense_0': {'units': output_size, 'use_bias' : False, 'kernel_regularizer':dense_reg},
                        'Dense_1': {'units': output_size, 'use_bias' : False, 'kernel_regularizer':dense_reg},
                        'Dense_2': {'units': output_size, 'use_bias' : False, 'kernel_regularizer':dense_reg},
                        },
            'Block_2': {'Input' : {'input_shape': (output_size,)},
                        'L2_normalization': {'axis': 1},
                        'Dense_0': {'units': output_size, 'use_bias' : False, 'kernel_regularizer':dense_reg},
                        'Dense_1': {'units': output_size, 'use_bias' : False, 'kernel_regularizer':dense_reg},
                        'Dense_2': {'units': output_size, 'use_bias' : False, 'kernel_regularizer':dense_reg},
                        },
            'Block_3': {'Input' : {'input_shape': (output_size,)},
                        'L2_normalization': {'axis': 1},
                        'Dense_0': {'units': output_size, 'use_bias' : False, 'kernel_regularizer':dense_reg},
                        'Dense_1': {'units': output_size, 'use_bias' : False, 'kernel_regularizer':dense_reg},
                        'Dense_2': {'units': output_size, 'use_bias' : False, 'kernel_regularizer':dense_reg},
                        },
            'Block_4': {'Input' : {'input_shape': (output_size,)},
                        'L2_normalization': {'axis': 1},
                        'Dense_0': {'units': output_size, 'use_bias' : False, 'kernel_regularizer':dense_reg},
                        'Dense_1': {'units': output_size, 'use_bias' : False, 'kernel_regularizer':dense_reg},
                        'Dense_2': {'units': output_size, 'use_bias' : False, 'kernel_regularizer':dense_reg},
                        },
            'Block_5': {'Input' : {'input_shape': (output_size,)},
                        'L2_normalization': {'axis': 1},
                        'Dense_0': {'units': output_size, 'use_bias' : False, 'kernel_regularizer':dense_reg},
                        'Dense_1': {'units': output_size, 'use_bias' : False, 'kernel_regularizer':dense_reg},
                        'Dense_2': {'units': output_size, 'use_bias' : False, 'kernel_regularizer':dense_reg},
                        },
            'Block_6': {'Input' : {'input_shape': (output_size,)},
                        'L2_normalization': {'axis': 1},
                        'Dense_0': {'units': output_size, 'use_bias' : False, 'kernel_regularizer':dense_reg},
                        'Dense_1': {'units': output_size, 'use_bias' : False, 'kernel_regularizer':dense_reg},
                        'Dense_2': {'units': output_size, 'use_bias' : False, 'kernel_regularizer':dense_reg},
                        },
            'train_mode': 'dense'
            }
    elif model_type == 'convdense2':
        if output_size is None:
            output_size = np.prod(input_size[:2])
        dims = {#'Input': input_size,
                'Block_0': {'Input' : {'input_shape': input_size},
                            'Conv2D_0': {'filters': 16, 'kernel_size': 5, 'use_bias' : False, 'kernel_regularizer': conv_reg},
                            'MaxPooling2D_0': {'pool_size': 3, 'strides': 1},
                            'Conv2D_1': {'filters': 8, 'kernel_size': 5, 'use_bias' : False, 'kernel_regularizer': conv_reg},
                            'Flatten': {},
                            'Dense_0': {'units': np.prod(input_size[:2]), 'use_bias' : False, 'kernel_regularizer': dense_reg}
                            },
                'Block_1': {'Input' : {'input_shape': (np.prod(input_size[:2]),)},
                            'L2_normalization': {'axis': 1},
                            'Reshape': {'target_shape': (*input_size[:2], 1)},
                            'Conv2D_0': {'filters': 16, 'kernel_size': 5, 'use_bias' : False, 'kernel_regularizer': conv_reg},
                            'MaxPooling2D_0': {'pool_size': 3, 'strides': 1},
                            'Conv2D_1': {'filters': 8, 'kernel_size': 5, 'use_bias' : False, 'kernel_regularizer': conv_reg},
                            'Flatten': {},
                            'Dense_0': {'units': np.prod(input_size[:2]), 'use_bias' : False, 'kernel_regularizer': dense_reg}
                            },
                # 'Block_2': {'Input' : {'input_shape': (np.prod(input_size[:2]),)},
                #             'L2_normalization': {'axis': 1},
                #             'Reshape': {'target_shape': (*input_size[:2], 1)},
                #             'Conv2D_0': {'filters': 4, 'kernel_size': 5, 'use_bias' : False, 'kernel_regularizer': conv_reg},
                #             'Conv2D_1': {'filters': 6, 'kernel_size': 5, 'use_bias' : False, 'kernel_regularizer': conv_reg},
                #             'Flatten': {},
                #             'Dense_0': {'units': np.prod(input_size[:2]), 'use_bias' : False, 'kernel_regularizer': dense_reg}
                #             },
                'Block_2': {'Input' : {'input_shape': (output_size,)},
                            'L2_normalization': {'axis': 1},
                            'Dense_0': {'units': output_size, 'use_bias' : False, 'kernel_regularizer': dense_reg},
                            'Dense_1': {'units': output_size, 'use_bias' : False, 'kernel_regularizer': dense_reg},
                            'Dense_2': {'units': output_size, 'use_bias' : False, 'kernel_regularizer': dense_reg},
                            },
                # 'Block_4': {'Input' : {'input_shape': (output_size,)},
                #             'L2_normalization': {'axis': 1},
                #             'Dense_0': {'units': output_size, 'use_bias' : False, 'kernel_regularizer': dense_reg},
                #             'Dense_1': {'units': output_size, 'use_bias' : False, 'kernel_regularizer': dense_reg},
                #             'Dense_2': {'units': output_size, 'use_bias' : False, 'kernel_regularizer': dense_reg},
                #             },
                'train_mode': 'conv'
                }
    elif model_type == 'denseconv':
        dims = {#'Input': input_size,
                'Block_0': {'Input' : {'input_shape': (input_size,)},
                            'Dense_0': {'units': np.prod(output_size), 'use_bias' : False, 'kernel_regularizer': dense_reg},
                            'Reshape_0': {'target_shape': output_size},
                            'Conv2D_0': {'filters': 8, 'kernel_size': 5, 'use_bias' : False, 'kernel_regularizer': conv_reg},
                            'MaxPooling2D_0': {'pool_size': 2, 'strides': 1},
                            'Conv2D_1': {'filters': 16, 'kernel_size': 5, 'use_bias' : False, 'kernel_regularizer': conv_reg},
                            # 'Reshape_1': {'target_shape': -1},
                            'Flatten_0': {},
                            'Dense_1': {'units': np.prod(output_size), 'use_bias' : False, 'kernel_regularizer': dense_reg},
                            },
                'Block_1': {'Input' : {'input_shape': (np.prod(output_size),)},
                            'Flatten_0': {},
                            'L2_normalization': {'axis': 1},
                            'Dense_0': {'units': np.prod(output_size), 'use_bias' : False, 'kernel_regularizer': dense_reg},
                            'Reshape_0': {'target_shape': output_size},
                            'Conv2D_0': {'filters': 8, 'kernel_size': 5, 'use_bias' : False, 'kernel_regularizer': conv_reg},
                            'MaxPooling2D_0': {'pool_size': 2, 'strides': 1},
                            'Conv2D_1': {'filters': 16, 'kernel_size': 5, 'use_bias' : False, 'kernel_regularizer': conv_reg},
                            'Flatten_1': {},
                            'Dense_1': {'units': np.prod(output_size), 'use_bias' : False, 'kernel_regularizer': dense_reg},
                            },
                # 'Block_2': {'Input' : {'input_shape': (np.prod(input_size[:2]),)},
                #             'L2_normalization': {'axis': 1},
                #             'Reshape': {'target_shape': (*input_size[:2], 1)},
                #             'Conv2D_0': {'filters': 4, 'kernel_size': 5, 'use_bias' : False, 'kernel_regularizer': conv_reg},
                #             'Conv2D_1': {'filters': 6, 'kernel_size': 5, 'use_bias' : False, 'kernel_regularizer': conv_reg},
                #             'Flatten': {},
                #             'Dense_0': {'units': np.prod(input_size[:2]), 'use_bias' : False, 'kernel_regularizer': dense_reg}
                #             },
                'Block_2': {'Input' : {'input_shape': (np.prod(output_size),)},
                            'L2_normalization': {'axis': 1},
                            'Dense_0': {'units': np.prod(output_size), 'use_bias' : False, 'kernel_regularizer': dense_reg},
                            'Dense_1': {'units': np.prod(output_size), 'use_bias' : False, 'kernel_regularizer': dense_reg},
                            'Dense_2': {'units': np.prod(output_size), 'use_bias' : False, 'kernel_regularizer': dense_reg},
                            },
                # 'Block_4': {'Input' : {'input_shape': (output_size,)},
                #             'L2_normalization': {'axis': 1},
                #             'Dense_0': {'units': output_size, 'use_bias' : False, 'kernel_regularizer': dense_reg},
                #             'Dense_1': {'units': output_size, 'use_bias' : False, 'kernel_regularizer': dense_reg},
                #             'Dense_2': {'units': output_size, 'use_bias' : False, 'kernel_regularizer': dense_reg},
                #             },
                'train_mode': 'dense'
                }
    
    dims['n_blocks'] = len([f for f in dims.keys() if f.startswith('Block_')])
    dims['n_layers'] = 0
    dims['layers_in_block'] = []
    for block in dims.keys():
        if block.startswith('Block_'):
            n_layers = len([f for f in dims[block].keys() if f.startswith(('Conv2D', 'Dense'))])
            dims['n_layers'] += n_layers
            dims['layers_in_block'].append(n_layers)

    if all(i == dims['layers_in_block'][0] for i in dims['layers_in_block']):
        dims['layers_in_block'] = dims['layers_in_block'][0]

    return dims

# %%

def dense_configuration(input_size, output_size):
    dims = {
            'Block_0': {'Input' : {'input_shape': (input_size,)},
                        'Dense_0': {'units': output_size, 'use_bias' : False},
                        'Dense_1': {'units': output_size, 'use_bias' : False},
                        'Dense_2': {'units': output_size, 'use_bias' : False},
                        },
            'Block_1': {'Input' : {'input_shape': (output_size,)},
                        'L2_normalization': {'axis': 1},
                        'Dense_0': {'units': output_size, 'use_bias' : False},
                        'Dense_1': {'units': output_size, 'use_bias' : False},
                        'Dense_2': {'units': output_size, 'use_bias' : False},
                        },
            'Block_2': {'Input' : {'input_shape': (output_size,)},
                        'L2_normalization': {'axis': 1},
                        'Dense_0': {'units': output_size, 'use_bias' : False},
                        'Dense_1': {'units': output_size, 'use_bias' : False},
                        'Dense_2': {'units': output_size, 'use_bias' : False},
                        },
            'Block_3': {'Input' : {'input_shape': (output_size,)},
                        'L2_normalization': {'axis': 1},
                        'Dense_0': {'units': output_size, 'use_bias' : False},
                        'Dense_1': {'units': output_size, 'use_bias' : False},
                        'Dense_2': {'units': output_size, 'use_bias' : False},
                        },
            'Block_4': {'Input' : {'input_shape': (output_size,)},
                        'L2_normalization': {'axis': 1},
                        'Dense_0': {'units': output_size, 'use_bias' : False},
                        'Dense_1': {'units': output_size, 'use_bias' : False},
                        'Dense_2': {'units': output_size, 'use_bias' : False},
                        },
            'Block_5': {'Input' : {'input_shape': (output_size,)},
                        'L2_normalization': {'axis': 1},
                        'Dense_0': {'units': output_size, 'use_bias' : False},
                        'Dense_1': {'units': output_size, 'use_bias' : False},
                        'Dense_2': {'units': output_size, 'use_bias' : False},
                        },
            'Block_6': {'Input' : {'input_shape': (output_size,)},
                        'L2_normalization': {'axis': 1},
                        'Dense_0': {'units': output_size, 'use_bias' : False},
                        'Dense_1': {'units': output_size, 'use_bias' : False},
                        'Dense_2': {'units': output_size, 'use_bias' : False},
                        },
            'train_mode': 'dense'
            }
    
    dims['n_blocks'] = len([f for f in dims.keys() if f.startswith('Block_')])
    dims['n_layers'] = 0
    dims['layers_in_block'] = []
    for block in dims.keys():
        if block.startswith('Block_'):
            n_layers = len([f for f in dims[block].keys() if f.startswith(('Conv2D', 'Dense'))])
            dims['n_layers'] += n_layers
            dims['layers_in_block'].append(n_layers)

    if all(i == dims['layers_in_block'][0] for i in dims['layers_in_block']):
        dims['layers_in_block'] = dims['layers_in_block'][0]

    return dims
# %%