#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Anonymous
"""

# %%

def model_configuration(channels, dims, params):
    num_classes = params.pop('n_classes')
    Preprocessing = {
                    'Augmentation': {
                        'RandomCrop': {'size': dims, 'padding': 4}, 
                        'RandomRotation': {'degrees': 15}
                        },
                    'Embedding': {
                        'num_embeddings': num_classes,
                        'embedding_dim': dims * dims
                        }, 
                    }
    config = {
            'Block_0': {'in_channels': channels, 'out_channels': 64, 
                        'index': 0, 'prev': Preprocessing, 'dims': dims, 
                        **params, 'maxpool': False
                        },
            'Block_1': {'in_channels': 64, 'out_channels': 128, 
                        'index': 1, 'prev': 'Block_0', 'dims': dims, 
                        **params, 'maxpool': True
                        },
            'Block_2': {'in_channels': 128, 'out_channels': 128, 
                        'index': 2, 'prev': 'Block_1', 'dims': dims//2, 
                        **params, 'maxpool': False
                        },
            'Block_3': {'in_channels': 128, 'out_channels': 128, 
                        'index': 3, 'prev': 'Block_2', 'dims': dims//2, 
                        **params, 'maxpool': False
                        },
            'Block_4': {'in_channels': 128, 'out_channels': 128, 
                        'index': 4, 'prev': 'Block_3', 'dims': dims//2, 
                        **params, 'maxpool': False
                        },
            'Block_5': {'in_channels': 128, 'out_channels': 256,
                        'index': 5, 'prev': 'Block_4', 'dims': dims//2, 
                        **params, 'maxpool': True
                        },
            'Block_6': {'in_channels': 256, 'out_channels': 256,
                        'index': 6, 'prev': 'Block_5', 'dims': dims//4, 
                        **params, 'maxpool': False
                        },
            'Block_7': {'in_channels': 256, 'out_channels': 256,
                        'index': 7, 'prev': 'Block_6', 'dims': dims//4, 
                        **params, 'maxpool': False
                        },
            'Block_8': {'in_channels': 256, 'out_channels': 256,
                        'index': 8, 'prev': 'Block_7', 'dims': dims//4, 
                        **params, 'maxpool': False
                        },
            'Block_9': {'in_channels': 256, 'out_channels': 512,
                        'index': 9, 'prev': 'Block_8', 'dims': dims//4, 
                        **params, 'maxpool': True
                        },
            'Block_10': {'in_channels': 512, 'out_channels': 512,
                        'index': 10, 'prev': 'Block_9', 'dims': dims//8, 
                        **params, 'maxpool': False
                        },
            'Block_11': {'in_channels': 512, 'out_channels': 512,
                        'index': 11, 'prev': 'Block_10', 'dims': dims//8, 
                        **params, 'maxpool': False
                        },
            }
    
    configuration = {'configuration': config, 'dims': dims, 'n_classes': num_classes, 'train_mode': 'conv'}
    configuration['n_blocks'] = len([f for f in config.keys() if f.startswith('Block_')])
    
    return configuration

# %%