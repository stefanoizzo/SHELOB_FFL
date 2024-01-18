#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Gio Lug 13 09:52:43 2023

@author: Anonymous
"""

# %%
# IMPORT SECTION
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'
import yaml
import pickle
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_datasets as tfds

import multiprocessing as mp
from EFL_FF.net import Network
from EFL_FF.model_utils import *
from IPython.display import display
from EFL_FF import model_configuration
from tqdm.notebook import tqdm, trange
from sklearn.metrics import classification_report
from EFL_FF import separate_data, myshow, plot_block_assignment

import warnings
warnings.filterwarnings("ignore")

# %% PRELIMINARY DEFINITIONS
if __name__ == '__main__':
    seed = 42
    CVD = 0
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(CVD)

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    dataset_name = 'mnist' # 'fashion_mnist' 'cifar10' 'mnist'
    train_mode = 'dense' # 'dense' 'conv' 'convdense' 'denseconv'
    fedprox = False # if True, use FedProx instead of FedAvg
    save_results = True

    # Training configuration
    if train_mode in ['dense', 'denseconv']:
        batch_size = 256 # batch size for each client
    elif train_mode in ['conv', 'convdense', 'convdense2']:
        batch_size = 64 # batch size for each client
    n_rounds = 300 # number of rounds
    num_clients = 10 # 20 # number of clients
    max_simultaneous_clients = 10 # maximum number of simultaneous clients
    train_size = 0.75 # merge original training set and test set, then split it manually. 
    least_samples = batch_size / (1-train_size) # least samples for each client

    # How assign the blocks to the clients
    block_assignment = 'all_in' # 'random' 'ordered' 'max3' 'all_in'
    
    # How to partition the dataset
    niid = True # If we want to have non-iid unbalance data must be True, else for iid balance data must be False
    alpha = .5 # for Dirichlet distribution

    if niid:
        # If we want to have iid balance data
        partition = 'dir'
        balance = False
    else:
        # If we want to have non-iid unbalance data
        partition = 'pat' 
        balance = True

    # Training configuration
    train_config = dict(
        block_threshold = 0.001, # threshold for the block freezing
        patience = 15, # patience for the block freezing
        batch_size = batch_size, # batch size
        verbose = False, # if True, print the accuracy of each client
        conv_reg = 1e-6, # regularization parameter for convolutional layers
        dense_reg = 1e-6, # regularization parameter for dense layers
    )

    # Model configuration
    model_config = dict(
        # configuration = configuration, # configuration of the network
        hard_negatives = True, # if True, the loss function is computed also on the hardest negative
        theta = 10., # hyperparameter for the goodness loss function
        epochs = 5, # number of epochs
        # n_classes = num_classes, # number of classes
        lr = 5e-5, # learning rate
        fedprox = fedprox, # if True, use FedProx instead of FedAvg
        mu = 0.1, # hyperparameter for fedprox regularization
    )

    if save_results:
        output_path = f'OUTPUT/{dataset_name}/{train_mode}'
        if num_clients == 1:
            output_path += f'_single/'
            if fedprox:
                print('WARNING: FedProx is not available for single client training')
                print('It\'s set to False')
                fedprox = False
        else:
            if block_assignment == 'random':
                output_path += f'_random'
            elif block_assignment == 'max3':
                output_path += f'_max3'
            elif block_assignment == 'all_in':
                output_path += f'_all_in'
            elif block_assignment == 'ordered':
                pass
            else:
                raise ValueError(f'block_assignment must be "random", "max3" or "ordered", not {block_assignment}')
            output_path += f'_nC={num_clients}_fedprox={fedprox}/lr={model_config["lr"]}_thresh={train_config["block_threshold"]}'
            if niid:
                output_path += f'_niid={niid}_alpha={alpha}/'
            else:
                output_path += f'_niid={niid}_balance={balance}/'
    else:
        output_path = 'OUTPUT/Test/'

    os.makedirs(output_path, exist_ok=True)

    if dataset_name == 'mnist':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    elif dataset_name == 'cifar10':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    elif dataset_name == 'fashion_mnist':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    else:
        (x_train, y_train), (x_test, y_test) = tfds.as_numpy(tfds.load(
                        dataset_name,
                        split=['train', 'test'],
                        batch_size=-1,
                        as_supervised=True,
                    ))

    x_train = x_train.squeeze()
    x_test = x_test.squeeze()
    y_train = y_train.squeeze()
    y_test = y_test.squeeze()
    
    num_classes = len(np.unique(y_train)) # number of classes
    model_config['n_classes'] = num_classes

    # Normalize the dataset
    x_train = tf.cast(x_train, tf.float32) / 255.
    x_test = tf.cast(x_test, tf.float32) / 255.

    y_test = tf.keras.utils.to_categorical(y_test, num_classes)
    # Split in batches
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)
        
    if train_mode == 'dense':
        input_size = int(np.prod(x_train.shape[1:])+num_classes)
        output_size = int(np.prod(x_train.shape[1:]))
    elif train_mode == 'conv':
        input_size = x_train.shape[1:]
        output_size = None
    elif train_mode.startswith('convdense'):
        input_size = x_train.shape[1:]
        output_size = None
    elif train_mode.startswith('denseconv'):
        input_size = int(np.prod(x_train.shape[1:])+num_classes)
        output_size = (20,20,1)
    else:
        raise ValueError('train_mode must be either "dense" or "conv"')

    configuration = model_configuration(input_size, output_size, model_type=train_mode, 
                                        conv_reg=train_config['conv_reg'], dense_reg=train_config['dense_reg'])
    model_config['configuration'] = configuration

    model_config['n_blocks'] = model_config['configuration'].pop('n_blocks')
    train_config['n_layers'] = model_config['configuration'].pop('n_layers')
    model_config['layers_in_block'] = model_config['configuration'].pop('layers_in_block')
    
    X, y, statistic = separate_data((x_train.numpy(), y_train), num_clients, num_classes, niid=niid, 
                    least_samples=least_samples, partition=partition, 
                    alpha=alpha, balance=balance, save_fig=output_path)

    # one hot encoding
    y = {i:tf.keras.utils.to_categorical(y[i], num_classes) for i in range(num_clients)}
    
# %%
if __name__ == '__main__':
    clients = {}
    pd_block_assignment = {}
    multiprocess_dict = mp.Manager().dict()
    for iclient in range(num_clients):
        # Inizialized the multiprocess_dict
        multiprocess_dict[iclient] = {}
        # Inizialized the clients dict with the data and the idx of blocks
        clients.setdefault(iclient, {'data':{}})
        clients[iclient]['data']['train_dataset'] = (X[iclient], y[iclient])
        clients[iclient]['data']['number_of_samples'] = X[iclient].shape[0]
        if num_clients == 1:
            clients[iclient]['data']['n_blocks'] = model_config['n_blocks']
        else:
            if block_assignment == 'random':
                clients[iclient]['data']['n_blocks'] = random.randint(1, model_config['n_blocks'])
            elif block_assignment == 'max3':
                clients[iclient]['data']['n_blocks'] = min(3, random.randint(1, model_config['n_blocks']))
            elif block_assignment == 'all_in':
                clients[iclient]['data']['n_blocks'] = model_config['n_blocks']
            elif block_assignment == 'ordered':
                clients[iclient]['data']['n_blocks'] = (iclient % model_config['n_blocks']) + 1
                # clients[iclient]['data']['n_blocks'] = iclient + 2
        pd_block_assignment[iclient] = clients[iclient]['data']['n_blocks']
        clients[iclient]['data']['idx_blocks'] = list(range(clients[iclient]['data']['n_blocks']))
        print(f'Client {iclient} has {clients[iclient]["data"]["idx_blocks"]} blocks')

    # Plot the block assignment
    pd_block_assignment = pd.DataFrame.from_dict(pd_block_assignment, orient='index', columns=['n_blocks'])

    plot_block_assignment(pd_block_assignment, output_path, 
                            num_blocks=model_config['n_blocks'], 
                            name=block_assignment)

    # Create the global model
    global_model = Network(**model_config)
    global_model.my_build(next(iter(test_dataset)))
    global_model.summary()
    
    train_config['global_parameters'] = global_model.my_get_weights()
    
    global_parameters_len = len(train_config['global_parameters'][0])
    global_parameters_keys = train_config['global_parameters'][1].keys()

    accs = []; ends_clients = 0
    reports = []; freeze_report = []
    local_freezed_blocks = {idx_fb : 0 for idx_fb in range(model_config['n_blocks'])}
    global_freezed_blocks = -1

    for i_round in trange(1, n_rounds+1, desc='Rounds'): 
        ctx = mp.get_context('spawn')
        for i_client in range(num_clients): 
            input_dict = {'number_client': i_client, 
                        'multiprocess_dict': multiprocess_dict}
            
            input_dict.update(train_config)
            input_dict.update(clients[i_client]['data'])
            input_dict.update(model_config)
            
            clients[i_client]['process'] = ctx.Process(target=train_EFLFF, 
                        args=(input_dict,))
        
        for n_group_clients in range(0, num_clients, max_simultaneous_clients):
            for client in list(clients.values())[n_group_clients:n_group_clients+max_simultaneous_clients]:
                if len(client['data']['idx_blocks']) != 0:
                    client['process'].start()

            for client in list(clients.values())[n_group_clients:n_group_clients+max_simultaneous_clients]:
                if len(client['data']['idx_blocks']) != 0:
                    client['process'].join()

        # Convert multiprocess_dict to dict
        multiprocess_dict = dict(multiprocess_dict)
        # Compute the new global parameters with the weighted average of the parameters of each client
        new_parameters = {}
        for j in range(model_config['n_blocks']): 
            parameters_layer = {}
            number_of_samples = 0
            for kmp, vmp in multiprocess_dict.items():
                if f'Block_{j}' in vmp['parameters'][1].keys():
                    number_of_samples += clients[kmp]['data']['number_of_samples']
                    parameters_layer[kmp] = [f*clients[kmp]['data']['number_of_samples'] for f in vmp['parameters'][1][f'Block_{j}']]
            if len(list(parameters_layer.values())) > 0: 
                n_layers_in_block_ = len(list(parameters_layer.values())[0])
                parameters_layer_in_block = {i_layer_block:0 for i_layer_block in range(n_layers_in_block_)}
                for i_layer_block in range(n_layers_in_block_):
                    for kpl, vpl in parameters_layer.items():
                        parameters_layer_in_block[i_layer_block] += vpl[i_layer_block]/number_of_samples
                new_parameters[f'Block_{j}'] = list(parameters_layer_in_block.values())


        if len(new_parameters) < len(global_parameters_keys): # Not all blocks have been trained in at least one client
            global_weights_old = global_model.my_get_weights()
            for k_old_weight, v_old_weight in global_weights_old[1].items():
                if k_old_weight not in new_parameters.keys():
                    new_parameters[k_old_weight] = v_old_weight

        global_model.my_set_weights(new_parameters)
        train_config['global_parameters'] = global_model.my_get_weights()

        # Evaluate the model
        with tf.device('/cpu:0'):
            preds, acc = global_model.my_predict(x_test, y_test, return_acc=True)
            preds_train, acc_train = global_model.my_predict(x_train, y_train, return_acc=True)
            n_label = list(range(global_model.n_classes))
            report = classification_report(np.argmax(y_test, axis=1), preds, labels=n_label, output_dict=True, zero_division=0)

        accs.append(acc)
        reports.append(report)

        if i_round > 1 and len(accs) > 1:
            plt.figure()
            plt.title('Accuracy on the global model')
            plt.plot(accs)
            if len(freeze_report) > 0:
                for idx_fb, i_round_fb in freeze_report:
                    plt.axvline(x=i_round_fb, color='r', linestyle='--')
            if save_results:
                os.makedirs(f'{output_path}/temp/', exist_ok=True)
                plt.savefig(f'{output_path}/temp/accuracy.png')
            myshow()

            if i_round % 10 == 0:
                # display report
                idxmax = np.argmax(accs)
                df = pd.DataFrame(reports[idxmax]).transpose()
                # save report as csv
                df.to_csv(f'{output_path}/temp/final_report.csv')
                df.to_excel(f'{output_path}/temp/final_report.xlsx')

        print(f'\nRound {i_round}, Server Accuracy: {acc}, Accuracy on train: {acc_train}\n' )
        
        n_client_want_freeze = 0
        for kmp, vmp in multiprocess_dict.items():
            n_client_want_freeze += vmp['freeze']
        if n_client_want_freeze >= np.ceil(2*num_clients/3).astype(int):
            global_freezed_blocks +=1
            freeze_report.append((global_freezed_blocks, i_round))
            print(f"Block {global_freezed_blocks} has been frozen globally")
            for k_client in clients.keys():
                clients[k_client]['data'].setdefault('freeze_blocks', []).append(global_freezed_blocks)
                clients[k_client]['data']['idx_blocks'] = [idxb+1 for idxb in client['data']['idx_blocks'] if idxb+1 < model_config['n_blocks']]
                multiprocess_dict[k_client]['freeze'] = False
                multiprocess_dict[k_client]['freeze_wait'] = 0
        if global_freezed_blocks == model_config['n_blocks']-1:
            break
        
        # Convert multiprocess_dict to multiprocess dict
        multiprocess_dict = ctx.Manager().dict(multiprocess_dict)

# %%
if __name__ == '__main__':
    plt.figure(figsize=(10, 10))
    plt.title('Final accuracy on the global model')
    plt.plot(accs)
    if len(freeze_report) > 0:
        for idx_fb, i_round_fb in freeze_report:
            plt.axvline(x=i_round_fb, color='r', linestyle='--')
    if save_results:
        plt.savefig(f'{output_path}/global_accuracy.png')
    myshow()

    for kmp, vmp in multiprocess_dict.items():
        plt.figure(figsize=(10, 10))
        plt.plot(vmp['l2_norm'])
        plt.title(f'Client {kmp} l2 norm difference between pre and post training parameters on first block')
        if save_results:
            os.makedirs(f'{output_path}/l2_norm/', exist_ok=True)
            plt.savefig(f'{output_path}/l2_norm/client_{kmp}.png')
        myshow()
    
    for kmp, vmp in multiprocess_dict.items():
        norm_l2 = np.array(vmp['l2_norm'])
        plt.figure(figsize=(10, 10))
        plt.plot(np.abs(norm_l2[1:] - norm_l2[:-1]))
        plt.title(f'Client {kmp} l2 variation norm between pre and post training parameters on first block')
        if save_results:
            os.makedirs(f'{output_path}/l2_norm_variation/', exist_ok=True)
            plt.savefig(f'{output_path}/l2_norm_variation/client_{kmp}.png')
        myshow()

    if save_results:
        with open(f'{output_path}/global_results.pkl', 'wb') as f:
            pickle.dump({'accs': accs, 'reports': reports, 'freeze_report': freeze_report}, f)
        with open(f'{output_path}/multiprocess_dict.pkl', 'wb') as f:
            pickle.dump(dict(multiprocess_dict), f)

        global_model.save_weights(f'{output_path}/global_model_weights.h5')
        with open(f'{output_path}/global_model_config.pkl', 'wb') as f:
            pickle.dump(global_model.my_get_config(), f)

        with open(f'{output_path}/train_config.pkl', 'wb') as f: 
            pickle.dump(train_config, f)

        with open(f'{output_path}/model_config.pkl', 'wb') as f: 
            pickle.dump(model_config, f)
        
        config_orig = {}
        for k, v in train_config.items():
            if k != 'global_parameters':
                config_orig[k] = v
        for k, v in model_config.items():
            if k != 'configuration':
                config_orig[k] = v
        
        with open(f'{output_path}/config_orig.yml', 'w') as f: 
            yaml.dump(config_orig, f)
        
        with open(f'{output_path}/net_configuration.yml', 'w') as f:
            yaml.dump(configuration, f)
        
        # display report
        idxmax = np.argmax(accs)
        df = pd.DataFrame(reports[idxmax]).transpose()
        display(df)

        # save report as csv
        df.to_csv(os.path.join(output_path, 'final_report.csv'))
        df.to_excel(os.path.join(output_path, 'final_report.xlsx'))
        
    print('Done')
    
# %%

