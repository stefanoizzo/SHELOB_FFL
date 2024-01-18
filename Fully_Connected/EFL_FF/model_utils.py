#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Gio Lug 13 16:50:12 2023

@author: Anonymous
"""
# %%
# IMPORT SECTION
import time
import numpy as np
import tensorflow as tf
from copy import deepcopy
from EFL_FF.net import Network, train
from EFL_FF.utils import mahalanobis

# %%
def train_EFLFF(input_dict):
    # Get the inputs
    t0 = time.time()
    number_client = input_dict.get('number_client')
    multiprocess_dict = input_dict['multiprocess_dict']
    batch_size = input_dict.get('batch_size', 50)
    global_parameters_ = input_dict.get('global_parameters')
    global_parameters, global_parameters_dict = global_parameters_
    verbose = input_dict.get('verbose', False)
    block_threshold = input_dict.get('block_threshold', 1e-1)
    patience = input_dict.get('patience', 10)

    model_config = {}
    model_config['layers_in_block'] = input_dict.get('layers_in_block', 2)
    model_config['n_classes'] = input_dict.get('n_classes', None)
    model_config['hard_negatives'] = input_dict.get('hard_negatives', True)
    model_config['theta'] = input_dict.get('theta', 10.)
    model_config['lr'] = input_dict.get('lr', 1e-3)
    model_config['epochs'] = input_dict.get('epochs', 100)
    model_config['configuration'] = input_dict.get('configuration')
    model_config['n_blocks'] = input_dict.get('n_blocks', 5)
    model_config['idx_blocks'] = input_dict.get('idx_blocks', list(range(model_config['n_blocks'])))
    model_config['mu'] = input_dict.get('mu', 0.1)
    model_config['fedprox'] = input_dict.get('fedprox', False)
    # model_config['n_layers'] = input_dict.get('n_layers', model_config['n_blocks']*model_config['layers_in_block'])
    
    train_dataset = input_dict.get('train_dataset')
    test_dataset = input_dict.get('test_dataset')

    # Prepare the data
    if train_dataset is None:
        raise ValueError('Train dataset is None')
    elif isinstance(train_dataset, (list, tuple)):
        # Split in batches
        x_train, y_train = train_dataset
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)
                                    ).shuffle(10000).batch(batch_size)
    
    if isinstance(test_dataset, (list, tuple)):
        x_test, y_test = test_dataset
        test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)
                                    ).batch(batch_size) 
    
    auxiliary_model = None
    if model_config['idx_blocks'][0] != 0:
        with tf.device('/CPU:0'):
            assert global_parameters is not None, 'global_parameters must be not None'
            auxiliary_config = deepcopy(model_config)
            auxiliary_config['idx_blocks'] = list(range(model_config['idx_blocks'][0]))
            auxiliary_model = Network(**auxiliary_config, name='auxiliary_client_'+str(number_client))
            auxiliary_parameters = []
            for i_gp in auxiliary_config['idx_blocks']:
                # for i_l in range(auxiliary_config['layers_in_block']):
                auxiliary_parameters.extend(global_parameters_dict['Block_'+str(i_gp)])
                    # auxiliary_parameters.append(global_parameters[i_gp*auxiliary_config['layers_in_block']+i_l])
            auxiliary_model.my_build(next(iter(train_dataset)), parameters=auxiliary_parameters)
            auxiliary_model.trainable = False
            if number_client == 0:
                print('Auxiliary model summary:')
                auxiliary_model.summary()
    
    # Select which global parameters are needed for the client model
    # e.g. if the client model has the first 2 blocks with 3 layers on each block, 
    # the global parameters selected are the first 6 parameters
    if global_parameters is not None:
        selected_parameters = []
        for idxb_gp in model_config['idx_blocks']:
            # selected_parameters.extend(global_parameters_dict['Block_'+str(idxb_gp)])
            block_parameters = []
            for gpb in global_parameters_dict['Block_'+str(idxb_gp)]:
                block_parameters.append(tf.constant(gpb))
            selected_parameters.append(block_parameters)
        
        model_config['global_parameters'] = selected_parameters
        # Create the model
        model = Network(**model_config, name='client_'+str(number_client))
        
        if auxiliary_model is not None:
            batch_builder = auxiliary_model(*next(iter(train_dataset)), last=True)
            model.my_build(batch_builder, parameters=selected_parameters)
        else:
            model.my_build(next(iter(train_dataset)), parameters=selected_parameters)
    else:
        # Create the model
        model = Network(**model_config, name='client_'+str(number_client))
        model.my_build(next(iter(train_dataset)))

    # Train the model
    model, losses, l2_losses, accs, ces, tec, report = train(model, train_dataset, test_dataset, auxiliary_model=auxiliary_model, verbose=verbose)

    if isinstance(accs, list): acc = accs[-1]
    else: acc = accs.numpy()

    # Dictionary to store the output
    output = {}
    if 'report' in multiprocess_dict[number_client].keys():
        output['report'] = multiprocess_dict[number_client]['report']
        output['accuracy'] = multiprocess_dict[number_client]['accuracy']
        output['report'].append(report)
        output['accuracy'].append(acc)
    else:
        output['report'] = [report]
        output['accuracy'] = [acc]

    output['time'] = time.time()-t0
    if tf.reduce_any(tf.math.is_nan(losses)):
        output['parameters'] = global_parameters_
    elif auxiliary_model is not None: 
        output['parameters'] = [auxiliary_model.my_get_weights()[0] + model.my_get_weights()[0]]
        temp = auxiliary_model.my_get_weights()[1]
        temp.update(model.my_get_weights()[1])
        output['parameters'].append(temp)
    else:
        output['parameters'] = model.my_get_weights()

    # Compute the l2 norm different between the global parameters and the client parameters
    if global_parameters_dict is not None:
        key_block = 'Block_'+ str(model_config['idx_blocks'][0])
        assert key_block == list(model.my_get_weights()[1].keys())[0]
        l2_norm = np.mean([np.linalg.norm(glob_w - local_w)/np.linalg.norm(glob_w)
                    for glob_w, local_w in zip(global_parameters_dict[key_block], output['parameters'][1][key_block])])
        # mahalanobis_mean = np.mean([np.mean(mahalanobis(glob_w, local_w)) 
        #                 for glob_w, local_w in zip(global_parameters_dict[key_block], output['parameters'][1][key_block])])
    else: 
        l2_norm = np.inf
        # mahalanobis_mean = np.inf

    # Save the output list controll in the multiprocess_dict
    if 'l2_norm' in multiprocess_dict[number_client].keys():
        output['l2_norm'] = multiprocess_dict[number_client]['l2_norm']
        output['l2_norm'].append(l2_norm)
        output['freeze_wait'] = multiprocess_dict[number_client]['freeze_wait']
    else:
        output['freeze_wait'] = 0
        output['l2_norm'] = [l2_norm]
    
    if 'freeze' not in multiprocess_dict[number_client].keys():
        output['freeze'] = False
    else:
        output['freeze'] = multiprocess_dict[number_client]['freeze']
    
    if len(output['l2_norm']) > 1 and np.abs(output['l2_norm'][-1]-output['l2_norm'][-2]) < block_threshold:
        output['freeze_wait'] += 1
        if output['freeze_wait'] >= patience:
            # output['freeze_wait'] = 0
            output['freeze'] = True
    else:
        output['freeze'] = False
        output['freeze_wait'] = 0

    # # Save the output list controll in the multiprocess_dict
    # if 'mahalanobis' in multiprocess_dict[number_client].keys():
    #     output['mahalanobis'] = multiprocess_dict[number_client]['mahalanobis']
    #     output['mahalanobis'].append(mahalanobis_mean)
    # else:
    #     output['mahalanobis'] = [mahalanobis_mean]

    multiprocess_dict[number_client] = output

    print(f'I\'m client {number_client}, Loss: {losses[-1]}, ',
          f'Accuracy: {acc}, L2_loss: {l2_losses[-1]}, time: {time.time()-t0} s')

# %%