#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Anonymous
"""
# %%
# IMPORT SECTION
import time
import torch
import numpy as np
from copy import deepcopy
import multiprocessing as mp
from torchvision.transforms import *
from EFL_FF.net import Network, My_Trainer
from torch.utils.data import DataLoader, TensorDataset

# %%
def train_EFLFF(input_dict, use_multiprocessing=False):
    # Get the inputs
    t0 = time.time()
    number_client = input_dict.get('number_client')
    multiprocess_dict = input_dict['multiprocess_dict']
    batch_size = input_dict.get('batch_size', 50)
    global_state_dict = input_dict.get('global_state_dict')
    block_threshold = input_dict.get('block_threshold', 1e-1)
    patience = input_dict.get('patience', 10)

    model_config = {}
    model_config['n_classes'] = input_dict.get('n_classes', None)
    epochs = input_dict.get('epochs', 100)
    model_config['configuration'] = input_dict.get('configuration')
    model_config['n_blocks'] = input_dict.get('n_blocks', 5)
    model_config['idx_blocks'] = input_dict.get('idx_blocks', list(range(model_config['n_blocks'])))
    fedprox = input_dict.get('fedprox', False)
    
    train_dataset = input_dict.get('train_dataset')

    device = input_dict.get('device', 'cpu')

    # Prepare the data
    if train_dataset is None:
        raise ValueError('Train dataset is None')
    elif isinstance(train_dataset, (list, tuple)):
        assert len(train_dataset)==2, 'Train dataset is not composed by two elements (x_train, y_train)'
        # Split in batches
        train_dataset = DataLoader(
                                TensorDataset(
                                train_dataset[0], train_dataset[1], 
                                ),
                        batch_size=batch_size, shuffle=True)
    else:
        raise ValueError('Train dataset is not a list or tuple')
    
    auxiliary_model = None
    if model_config['idx_blocks'][0] != 0:
        assert isinstance(global_state_dict, dict), 'global_parameters must be a dictionary of parameters'
        auxiliary_config = deepcopy(model_config)
        auxiliary_config['idx_blocks'] = list(range(model_config['idx_blocks'][0]))
        auxiliary_config['name_blocks'] = ['Block_'+str(aci) for aci in auxiliary_config['idx_blocks']]
        auxiliary_config['configuration'] = {acn: auxiliary_config['configuration'][acn] 
                                        for acn in auxiliary_config['name_blocks']}
        auxiliary_model = Network(auxiliary_config).to('cpu') 
        auxiliary_parameters = {acn: global_state_dict[acn] for acn in auxiliary_config['name_blocks']}
        auxiliary_model.my_set_state_dict(auxiliary_parameters)
        auxiliary_model.eval()

    
    # Select which global parameters are needed for the client model
    # e.g. if the client model has the first 2 blocks with 3 layers on each block, 
    # the global parameters selected are the first 6 parameters
    model_config['name_blocks'] = ['Block_'+str(idxb) for idxb in model_config['idx_blocks']]
    if global_state_dict is not None:
        
        model_config['global_state_dict'] = {mcnb: global_state_dict[mcnb] for mcnb in model_config['name_blocks']}
        model_config['configuration'] = {mcnb: model_config['configuration'][mcnb] for mcnb in model_config['name_blocks']}
        
        # Create the model
        model = Network(model_config) 
        model.my_set_state_dict(model_config['global_state_dict'])
        if fedprox:
            model.preprare_fedprox()
    else:
        # Create the model
        model_config['configuration'] = {mcnb: model_config['configuration'][mcnb] for mcnb in model_config['name_blocks']}
        model = Network(model_config) 

    if fedprox:
        model.preprare_fedprox()
    
    model = model.to(device)
    model.train()

    # Train the model
    trainer = My_Trainer(
        epochs, 
        number_client, 
        model, 
        auxiliary_model,
    )

    metrics = trainer.fit(train_dataset) 
    model = trainer.model.to('cpu')

    # Dictionary to store the output
    output = {}
    if 'report' in multiprocess_dict[number_client].keys():
        output['report'] = multiprocess_dict[number_client]['report']
        output['accuracy'] = multiprocess_dict[number_client]['accuracy']
        output['report'].append(metrics['report'])
        output['accuracy'].append(metrics['accuracy'])
    else:
        output['report'] = [metrics['report']]
        output['accuracy'] = [metrics['accuracy']]

    output['time'] = time.time()-t0
    if torch.isnan(metrics['loss']):
        print(f'Client {number_client}: Loss is NaN')
        output['parameters'] = global_state_dict
    else:
        output['parameters'] = model.my_get_state_dict()

    # Compute the l2 norm different between the global parameters and the client parameters
    if global_state_dict is not None:
        first_key_block = list(output['parameters'].keys())[0]
        global_state_dict_first = global_state_dict[first_key_block]
        local_state_dict_first = output['parameters'][first_key_block]
        l2_norm = []
        for k_gsdf, v_gsdf in global_state_dict_first.items():
            if k_gsdf.split(sep='.')[-1] != 'num_batches_tracked':
                global_norm = np.linalg.norm(v_gsdf.data)
                if global_norm !=0:
                    l2_norm.append(np.linalg.norm(v_gsdf.data - local_state_dict_first[k_gsdf].data)/global_norm)
                else:
                    l2_norm.append(np.inf)
        l2_norm = np.mean(l2_norm)
    else: 
        l2_norm = np.inf

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
            output['freeze'] = True
    else:
        output['freeze'] = False
        output['freeze_wait'] = 0

    multiprocess_dict[number_client] = output

    print(f'I\'m client {number_client}, Loss: {metrics["loss"]:.5f}, ',
            f'Accuracy: {metrics["accuracy"]:.5f}, time: {time.time()-t0:.5f}s')

    if not use_multiprocessing:
        return multiprocess_dict
    
# %%    

def run_train(num_clients, clients, train_config, model_config, multiprocess_dict, 
            max_simultaneous_clients=1):
    if max_simultaneous_clients > 1:
        ctx = mp.get_context('spawn')
        multiprocess_dict = ctx.Manager().dict(multiprocess_dict)
        for i_client in range(num_clients): 
            input_dict = {'number_client': i_client, 
                        'multiprocess_dict': multiprocess_dict
                        }
            
            input_dict.update(train_config)
            input_dict.update(clients[i_client]['data'])
            input_dict.update(model_config)
            input_dict['device'] = clients[i_client]['device']
            
            clients[i_client]['process'] = ctx.Process(target=train_EFLFF, 
                        args=(input_dict, True))
        
        for n_group_clients in range(0, num_clients, max_simultaneous_clients):
            for client in list(clients.values())[n_group_clients:n_group_clients+max_simultaneous_clients]:
                if len(client['data']['idx_blocks']) != 0:
                    client['process'].start()

            for client in list(clients.values())[n_group_clients:n_group_clients+max_simultaneous_clients]:
                if len(client['data']['idx_blocks']) != 0:
                    client['process'].join()
        
        return dict(multiprocess_dict)
    
    else:
        for i_client in range(num_clients): 
            input_dict = {'number_client': i_client, 
                        'multiprocess_dict': multiprocess_dict}
            
            input_dict.update(train_config)
            input_dict.update(clients[i_client]['data'])
            input_dict.update(model_config)
            input_dict['device'] = clients[i_client]['device']
            
            multiprocess_dict = train_EFLFF(input_dict, False) 

        return multiprocess_dict


def run_evaluate(global_model, test_dataset, device):
    predict_dict = mp.Manager().dict()
    ctx_pred = mp.get_context('spawn')
    
    predict_process = ctx_pred.Process(target=global_model.evaluate, 
                    args=(test_dataset, device, predict_dict, None))
    
    predict_process.start()
    predict_process.join()

    return dict(predict_dict)


# %%

