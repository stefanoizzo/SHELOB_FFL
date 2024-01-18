#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Gio Lug 13 09:52:43 2023

@author: Anonymous
"""

# %%
# IMPORT SECTION
if __name__ == '__main__':
    import os
    from pathlib import Path
    script_path = Path(__file__).parent
    os.chdir(script_path)
    root_path = script_path.parent

    import yaml
    import torch
    import pickle
    import random
    import numpy as np
    import pandas as pd
    from copy import deepcopy
    import matplotlib.pyplot as plt
    from EFL_FF.net import Network
    from IPython.display import display
    from torchvision.transforms import *
    from tqdm.notebook import tqdm, trange
    from EFL_FF.model_utils import run_train, run_evaluate
    from torch.utils.data import DataLoader, TensorDataset
    from EFL_FF.net_configuration import model_configuration
    from EFL_FF.utils import separate_data, myshow, plot_block_assignment
    from torchvision.datasets import MNIST, FashionMNIST, CIFAR10 #, SVHN

    import warnings
    warnings.filterwarnings("ignore")

# %% 
# PRELIMINARY DEFINITIONS
if __name__ == '__main__':
    seed = 42
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    CVD = 0
    if torch.cuda.device_count() == 0 and CVD != 0:
        print("WARNING: Only one CUDA device is found")
        CVD = 0
    device = torch.device(f"cuda:{CVD}" if torch.cuda.is_available() else "cpu")

    dataset_name = 'cifar10' # 'fashion_mnist' 'cifar10' 'mnist'
    fedprox = True                             # if True, use FedProx instead of FedAvg
    freeze = True
    save_results = True

    # Training configuration
    batch_size = 128                              # batch size for each client
    global_batch_size = 2048                     # batch size for the global model
    n_rounds = 5000                               # number of rounds
    num_clients = 10                             # number of clients
    max_simultaneous_clients = 10                # maximum number of simultaneous clients
    train_size = 0.75                            # merge original training set and test set, then split it manually. 
    least_samples = batch_size / (1-train_size)  # least samples for each client

    # How assign the blocks to the clients
    block_assignment = 'min2_ordered' # 'random' 'ordered' 'max3' 'all_in' min2_ordered

    # How to partition the dataset
    niid = True # If we want to have non-iid unbalance data must be True, else for iid balance data must be False
    beta = .1 # for Dirichlet distribution

    if num_clients == 1:
        batch_size = 512
        niid = False
        beta = 1.

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
        patience = 10, # patience for the block freezing
        batch_size = batch_size, # batch size
        verbose = False, # if True, print the accuracy of each client
        # conv_reg = 1e-6, # regularization parameter for convolutional layers
        # dense_reg = 1e-6, # regularization parameter for dense layers
        epochs = 6, # number of epochs
        fedprox = fedprox, # if True, use FedProx instead of FedAvg
    )
    
    # Model configuration
    mu = 0.01 if fedprox else 0.0
    block_config = dict(
        lr = 1e-3,    # learning rate
        scale = 1,    # scale of the block
        alpha = 4.0,  # the scale parameter for SymBa, use None for the original loss
        bn = True,    # True or False for batch-norm and layer-norm respectively
        olu = True,   # True or False for olu or layer-wise respectively
        tau = 2.0,    # the threshold parameter for the original loss
        mu = mu,  # hyperparameter for fedprox regularization
    )
    
    assert train_config['epochs'] % 2 == 0, 'The number of epochs must be even'


    if save_results:
        output_path = root_path / f'OUTPUT/{dataset_name}/'
        if num_clients == 1:
            output_path = output_path.with_name(output_path.name + f'_single')
            if fedprox:
                print('WARNING: FedProx is not available for single client training')
                print('It\'s set to False')
                fedprox = False
        else:
            if block_assignment not in ['random', 'max3', 'ordered', 'all_in', 'min2_ordered']:
                raise ValueError(f'block_assignment must be "random", "max3", "ordered", "all_in" or "min2_ordered", not {block_assignment}')
            output_path = output_path.with_name(output_path.name + f'_{block_assignment}')
            output_path = output_path.with_name(output_path.name + \
                        f'_nC={num_clients}_mu={block_config["mu"]}_rounds={n_rounds}') / f'lr={block_config["lr"]}'
            if freeze:
                output_path = output_path.with_name(output_path.name + \
                                    f'_thresh={train_config["block_threshold"]}_patience={train_config["patience"]}')
            else:
                output_path = output_path.with_name(output_path.name + \
                                    f'_nofreeze')
            if niid:
                output_path = output_path.with_name(output_path.name + \
                                    f'_niid={niid}_beta={beta}')
            else:
                output_path = output_path.with_name(output_path.name + \
                                    f'_niid={niid}_balance={balance}')
    else:
        output_path = root_path / f'OUTPUT/Test_{dataset_name}/'

    os.makedirs(output_path, exist_ok=True)
    print('Save in ', output_path)

    transform = Compose([
            ToTensor(),
        ])
    
    if dataset_name == 'mnist':
        train = MNIST(root=root_path / 'Datasets/', train=True, download=True, transform=transform)
        x_train, y_train = train.data, train.targets
        test = MNIST(root=root_path / 'Datasets/', train=False, download=True, transform=transform)
        x_test, y_test = test.data, test.targets
    elif dataset_name == 'cifar10':
        train = CIFAR10(root=root_path / 'Datasets/', train=True, download=True, transform=transform)
        x_train, y_train = torch.tensor(train.data), torch.tensor(train.targets)
        test = CIFAR10(root=root_path / 'Datasets/', train=False, download=True, transform=transform)
        x_test, y_test = torch.tensor(test.data), torch.tensor(test.targets)
    elif dataset_name == 'fashion_mnist':
        train = FashionMNIST(root=root_path / 'Datasets/', train=True, download=True, transform=transform)
        x_train, y_train = train.data, train.targets
        test = FashionMNIST(root=root_path / 'Datasets/', train=False, download=True, transform=transform)
        x_test, y_test = test.data, test.targets
    else:
        raise NotImplementedError(f'Dataset {dataset_name} not implemented')

    x_train = x_train.squeeze()
    x_test = x_test.squeeze()
    y_train = y_train.squeeze()
    y_test = y_test.squeeze()
    if len(x_train.shape) == 3:
        x_train = x_train[:, None]
        x_test = x_test[:, None]
    else:
        assert len(x_train.shape) == 4, 'The images shape must have 4 dimensions: (n_samples, channels, height, width)'
        if x_train.shape[-1] <= 3:
            x_train = torch.moveaxis(x_train, -1, 1)
            x_test = torch.moveaxis(x_test, -1, 1)
    
    num_classes = len(np.unique(y_train)) # number of classes
    block_config['n_classes'] = num_classes

    # Normalize the dataset
    x_train = x_train / 255.
    x_test = x_test / 255.

    assert len(x_train.shape) == 4, 'The images shape must have 4 dimensions: (n_samples, channels, height, width)'
    channels = x_train.shape[1]+1
    assert x_train.shape[3] == x_train.shape[2], 'The images must be squared'
    dims = x_train.shape[2]

    model_config = model_configuration(channels, dims, block_config)
    
    # Split in batches
    test_dataset = TensorDataset(x_test, y_test)
    test_dataset = DataLoader(test_dataset, batch_size=global_batch_size, shuffle=False)

    train_dataset = TensorDataset(x_train, y_train)
    train_dataset = DataLoader(train_dataset, batch_size=global_batch_size, shuffle=False)
    
    X, y, statistic = separate_data((x_train.numpy(), y_train.numpy()), num_clients, num_classes, niid=niid, 
                    least_samples=least_samples, partition=partition, 
                    beta=beta, balance=balance, save_fig=output_path)
    
    # Convert to torch tensor
    X = {kX:torch.tensor(vX) for kX, vX in X.items()}
    y = {ky:torch.tensor(vy) for ky, vy in y.items()}

# %%

if __name__ == '__main__':
    clients = {}
    pd_block_assignment = {}
    multiprocess_dict = dict()
    min2_value = 2
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
            elif block_assignment == 'min2_ordered':
                clients[iclient]['data']['n_blocks'] = min2_value
                min2_value += 1
                if min2_value > model_config['n_blocks']:
                    min2_value = 2
        pd_block_assignment[iclient] = clients[iclient]['data']['n_blocks']
        clients[iclient]['data']['idx_blocks'] = list(range(clients[iclient]['data']['n_blocks']))
        clients[iclient]['device'] = device
        print(f'Client {iclient} has {clients[iclient]["data"]["idx_blocks"]} blocks')

    # Plot the block assignment
    pd_block_assignment = pd.DataFrame.from_dict(pd_block_assignment, orient='index', columns=['n_blocks'])

    plot_block_assignment(pd_block_assignment, output_path, 
                        num_blocks=model_config['n_blocks'], 
                        name=block_assignment)

    # Create the global model
    global_model = Network(model_config).to('cpu')
    train_config['global_state_dict'] = global_model.my_get_state_dict()

    global_parameters_len = len(train_config['global_state_dict'])
    global_parameters_keys = train_config['global_state_dict'].keys()

    accs = []; ends_clients = 0
    accs_glob = []
    reports = []; freeze_report = []
    local_freezed_blocks = {idx_fb : 0 for idx_fb in range(model_config['n_blocks'])}
    global_freezed_blocks = -1

    for i_round in trange(1, n_rounds+1, desc='Rounds'): 
        
        multiprocess_dict = run_train(num_clients, clients, train_config, model_config, multiprocess_dict, max_simultaneous_clients)

        # Compute the new global parameters with the weighted average of the parameters of each client
        if num_clients == 1:
            new_parameters = {}
            for name_block in global_parameters_keys:
                if name_block in multiprocess_dict[0]['parameters']:
                    new_parameters[name_block] = deepcopy(multiprocess_dict[0]['parameters'][name_block])
                else:
                    new_parameters[name_block] = deepcopy(train_config['global_state_dict'][name_block])
            global_model.my_set_state_dict(new_parameters)
            train_config['global_state_dict'] = global_model.my_get_state_dict()
        else:
            new_parameters = {}
            for name_block in global_parameters_keys:
                number_of_samples_in_block = 0
                for kmp, vmp in multiprocess_dict.items():
                    if name_block in vmp['parameters'].keys():
                        sample_in_client = clients[kmp]['data']['number_of_samples']
                        number_of_samples_in_block += sample_in_client

                if number_of_samples_in_block == 0:
                    new_parameters[name_block] = deepcopy(train_config['global_state_dict'][name_block])
                    continue

                for kmp, vmp in multiprocess_dict.items():
                    if name_block in vmp['parameters'].keys():
                        sample_in_client = clients[kmp]['data']['number_of_samples']
                        sample_factor = sample_in_client/number_of_samples_in_block
                        if name_block not in new_parameters:
                            new_parameters[name_block] = \
                                {k:f*sample_factor if k.split('.')[-1]!='num_batches_tracked' else torch.tensor(0) 
                                    for k, f in vmp['parameters'][name_block].items()}
                        else:
                            for k_np, v_np in new_parameters[name_block].items():
                                if k_np.split('.')[-1]!='num_batches_tracked':
                                    new_parameters[name_block][k_np] += \
                                        vmp['parameters'][name_block][k_np]*sample_factor
        
            # Update the global model
            global_model.my_set_state_dict(new_parameters)
            train_config['global_state_dict'] = global_model.my_get_state_dict()

        
        # Evaluate the model on the test set
        metrics_test = run_evaluate(global_model, test_dataset, device)

        display(pd.DataFrame(metrics_test['report']).T)
        acc = metrics_test['accuracy']
        accs.append(acc)
        acc_glob = metrics_test['global_accuracy']
        accs_glob.append(acc_glob)
        reports.append(metrics_test['report'])

        if i_round > 1 and len(accs) > 1:
            plt.figure()
            plt.title('Accuracy on the global model')
            plt.plot(accs)
            if len(freeze_report) > 0:
                for idx_fb, i_round_fb in freeze_report:
                    plt.axvline(x=i_round_fb, color='r', linestyle='--')
            os.makedirs(f'{output_path}/temp/', exist_ok=True)
            plt.savefig(f'{output_path}/temp/accuracy.png')
            myshow()

            plt.figure()
            plt.title('Global accuracy on the global model')
            plt.plot(accs_glob)
            if len(freeze_report) > 0:
                for idx_fb, i_round_fb in freeze_report:
                    plt.axvline(x=i_round_fb, color='r', linestyle='--')
            plt.savefig(f'{output_path}/temp/global_accuracy.png')
            myshow()

        print(f'\nRound {i_round}, Server Accuracy: {acc:.5f}, Accuracy on global blocks: {acc_glob:.5f}\n' )
        
        if freeze:
            n_client_want_freeze = 0
            for kmp, vmp in multiprocess_dict.items():
                n_client_want_freeze += vmp['freeze']
            print('Number of clients that want to freeze: ', n_client_want_freeze)
            if n_client_want_freeze >= np.ceil(2*num_clients/3).astype(int):
                global_freezed_blocks +=1
                freeze_report.append((global_freezed_blocks, i_round))
                print(f"Block {global_freezed_blocks} has been frozen globally")
                for k_client, v_client in clients.items():
                    clients[k_client]['data'].setdefault('freeze_blocks', []).append(global_freezed_blocks)
                    clients[k_client]['data']['idx_blocks'] = [idxb+1 for idxb in v_client['data']['idx_blocks'] if idxb+1 < model_config['n_blocks']]
                    multiprocess_dict[k_client]['freeze'] = False
                    multiprocess_dict[k_client]['freeze_wait'] = 0
            if global_freezed_blocks == model_config['n_blocks']-1:
                break
        
        # # Convert multiprocess_dict to multiprocess dict
        global_model = global_model.to('cpu')

# %%

if __name__ == '__main__':
    plt.figure(figsize=(10, 10))
    plt.title('Final accuracy on the global model')
    plt.plot(accs)
    if len(freeze_report) > 0:
        for idx_fb, i_round_fb in freeze_report:
            plt.axvline(x=i_round_fb, color='r', linestyle='--')
    plt.savefig(output_path / 'accuracy.png')
    myshow()

    plt.figure(figsize=(10, 10))
    plt.title('Final global accuracy on the global model')
    plt.plot(accs_glob)
    if len(freeze_report) > 0:
        for idx_fb, i_round_fb in freeze_report:
            plt.axvline(x=i_round_fb, color='r', linestyle='--')
    plt.savefig(output_path / 'global_accuracy.png')
    myshow()

    for kmp, vmp in multiprocess_dict.items(): 
        plt.figure(figsize=(10, 10))
        plt.plot(vmp['l2_norm'])
        plt.title(f'Client {kmp} l2 norm difference between pre and post training parameters on first block')
        os.makedirs(f'{output_path}/l2_norm/', exist_ok=True)
        if len(freeze_report) > 0:
            for idx_fb, i_round_fb in freeze_report:
                plt.axvline(x=i_round_fb, color='r', linestyle='--')
        (output_path / 'l2_norm').mkdir(exist_ok=True)
        plt.savefig(output_path / f'l2_norm/client_{kmp}.png')
        myshow()

    for kmp, vmp in multiprocess_dict.items():
        norm_l2 = np.array(vmp['l2_norm'])
        plt.figure(figsize=(10, 10))
        plt.plot(np.abs(norm_l2[1:] - norm_l2[:-1]))
        plt.title(f'Client {kmp} l2 variation norm between pre and post training parameters on first block')
        if len(freeze_report) > 0:
            for idx_fb, i_round_fb in freeze_report:
                plt.axvline(x=i_round_fb+1, color='r', linestyle='--')
        (output_path / 'l2_norm_variation').mkdir(exist_ok=True)
        plt.savefig(output_path / f'l2_norm_variation/client_{kmp}.png')
        myshow()

    # if save_results or True:
    with open(output_path / 'global_results.pkl', 'wb') as f:
        pickle.dump({'accs': accs, 'reports': reports, 'freeze_report': freeze_report}, f)
    with open(output_path / 'multiprocess_dict.pkl', 'wb') as f:
        pickle.dump(dict(multiprocess_dict), f)

    # Save pytorch model
    torch.save(global_model, output_path / 'global_model_weights.pt')

    with open(output_path / 'global_model_state_dict.pkl', 'wb') as f:
        pickle.dump(global_model.state_dict(), f)

    with open(output_path / 'train_config.pkl', 'wb') as f: 
        pickle.dump(train_config, f)

    with open(output_path / 'model_config.pkl', 'wb') as f: 
        pickle.dump(model_config, f)

    with open(f'{output_path}/net_configuration.yml', 'w') as f:
        yaml.dump(global_model.config, f)

    # display report
    idx = np.argmax(accs)
    df = pd.DataFrame(reports[idx]).transpose()
    display(df)

    # save report as csv
    df.to_csv(output_path / 'final_report.csv')
    df.to_excel(output_path / 'final_report.xlsx')


    df_parameters = pd.DataFrame()
    df_parameters['dataset'] = [dataset_name]
    df_parameters['num_clients'] = [num_clients]
    df_parameters['num_classes'] = [num_classes]
    df_parameters['block_assignment'] = [block_assignment]
    df_parameters['n_rounds'] = [n_rounds]
    df_parameters['batch_size'] = [batch_size]
    df_parameters['freeze'] = [freeze]
    df_parameters['niid'] = [niid]
    df_parameters['beta'] = [beta]
    df_parameters['balance'] = [balance]
    df_parameters['partition'] = [partition]
    df_parameters['lr'] = [block_config['lr']]
    df_parameters['scale'] = [block_config['scale']]
    df_parameters['alpha'] = [block_config['alpha']]
    df_parameters['tau'] = [block_config['tau']]
    df_parameters['mu'] = [block_config['mu']]
    df_parameters['bn'] = [block_config['bn']]
    df_parameters['olu'] = [block_config['olu']]
    df_parameters['epochs'] = [train_config['epochs']]
    df_parameters['fedprox'] = [train_config['fedprox']]
    df_parameters['block_threshold'] = [train_config['block_threshold']]
    df_parameters['patience'] = [train_config['patience']]
    df_parameters['max_simultaneous_clients'] = [max_simultaneous_clients]
    df_parameters['global_batch_size'] = [global_batch_size]
    df_parameters['seed'] = [seed]
    df_parameters['device'] = [device]
    df_parameters['output_path'] = [output_path]
    df_parameters['CVD'] = [CVD]
    df_parameters.to_csv(output_path / 'parameters.csv')
    df_parameters.to_excel(output_path / 'parameters.xlsx')

    print('Done')

# %%

