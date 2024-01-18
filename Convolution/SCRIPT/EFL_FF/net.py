#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Anonymous
"""

# %%

import torch
import numpy as np
from torch import nn
from copy import deepcopy
from torch.optim import Adam
from torchmetrics import Accuracy
from lightning import LightningModule
from torch.nn.functional import softplus
from sklearn.metrics import classification_report
from torchvision.transforms import RandomCrop, RandomRotation

# %%

def should_detach(olu, index, iteration):
    return (index % 2 == 0) ^ (iteration % 2 == 0) if olu else True


class Block(LightningModule):
    def __init__(self, in_channels, out_channels, index, prev, dims, lr, scale, alpha, tau, olu, 
                bn, maxpool=False, mu=0.):
        super().__init__()

        in_channels *= scale if in_channels != 4 else 1
        out_channels *= scale

        self.preprocessing = None
        if isinstance(prev, dict):
            self.preprocessing = nn.ModuleDict()
            params = []
            for key, value in prev.items():
                if key == 'Embedding':
                    self.preprocessing[key] = nn.Embedding(**value)
                
                elif key == 'Augmentation':
                    list_of_operation = []
                    for ka, va in value.items():
                        if ka == 'RandomCrop':
                            list_of_operation.append(RandomCrop(**va))
                        elif ka == 'RandomRotation':
                            list_of_operation.append(RandomRotation(**va))
                        else: raise NotImplementedError
                    self.preprocessing[key] = nn.Sequential(*list_of_operation)
                
                else: raise NotImplementedError
                params += list(self.preprocessing[key].parameters())
        else:
            params = list(prev.parameters()) if prev else list()
        
        self.inner = nn.Sequential(
            nn.BatchNorm2d(in_channels) if bn else nn.LayerNorm([in_channels, dims, dims]),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2) if maxpool else nn.Identity(),
        )

        self.params = params + list(self.inner.parameters())
        self.optimizer = Adam(params + list(self.inner.parameters()), lr=lr)

        self.index = index
        self.alpha = alpha
        self.tau = tau
        self.olu = olu
        self.dims_block = dims
        self.fixed_parameters = None
        self.mu = mu

    def forward(self, x, iteration):

        if self.preprocessing is not None:
            assert len(x)==2, "x should be a tuple of (x, y)"
            x, y = x
            if 'Augmentation' in self.preprocessing and self.training:
                x = self.preprocessing['Augmentation'](x)
            
            if 'Embedding' in self.preprocessing:
                embedding = self.preprocessing['Embedding'](y).view(-1, 1, self.dims_block,  self.dims_block)
                x = torch.cat([x, embedding], dim=1)

        out = self.inner(x)
        activations = out.pow(2).flatten(start_dim=1)

        if not self.training:
            return out, activations.mean(1), None, None, None, None

        elif should_detach(self.olu, self.index, iteration):
            pos, neg = activations.chunk(2)
            pos, neg = pos.mean(1), neg.mean(1)

            delta = pos - neg

            if self.alpha is None:
                loss = torch.cat([softplus(self.tau - pos), softplus(neg - self.tau)]).mean()
            else:
                loss = softplus(-self.alpha * delta).mean()

            if self.fixed_parameters is not None:
                l2_norm = torch.finfo(loss.dtype).eps

                for glob_param, param in zip(self.fixed_parameters, self.params):
                    glob_param = glob_param.to(param.device)
                    l2_norm += torch.square(glob_param - param).sum()
                
                l2_norm *= self.mu
                loss += l2_norm

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            return out.detach(), activations.mean(1), delta.mean(0), loss.item(), pos.mean(0), neg.mean(0)

        else:
            return out, None, None, None, None, None


class Network(LightningModule):
    def __init__(self, net_config):
        super().__init__()

        self.config = deepcopy(net_config['configuration'])
        
        self.layers = nn.ModuleDict()
        for name, layer in self.config.items():
            prev = layer.pop('prev', None)
            if isinstance(prev, (str, int)):
                if prev in self.layers.keys():
                    prev = self.layers[prev]
                else: prev = None
            elif not isinstance(prev, dict): raise ValueError(f'Invalid prev name: {type(prev)}')

            self.layers[name] = Block(**layer, prev=prev)
            
        self.n_classes = net_config['n_classes']
        self.num_layers = len(self.layers)

        self.accuracy = Accuracy("multiclass", num_classes=self.n_classes)
        
    def forward(self, x, iteration=None, return_output=False):

        metrics = []
        for layer in self.layers.values():
            x, *rest = layer(x, iteration)
            metrics.append(rest)

        if return_output:
            assert not self.training, "return_output should be False in training mode"
            return x

        if self.training:
            _, delta, loss, pos, neg = zip(*metrics)
            return delta, loss, pos, neg
        else:
            goodness, _, _, _, _ = zip(*metrics)
            return torch.stack(goodness)

    def my_get_weights(self, return_dict=True, return_list=False):
        weights = []
        weights_dict = {}
        for name, layer in self.layers.items():
            weights.append(list(layer.parameters()))
            weights_dict[name] = list(layer.parameters())
            
        if return_dict and return_list:
            return weights, weights_dict
        elif return_list:
            return weights
        elif return_dict:
            return weights_dict
        
    def my_get_state_dict(self):
        weights_dict = {}
        for name, layer in self.layers.items():
            weights_dict[name] = dict(layer.state_dict())
        return weights_dict
    
    def my_set_weights(self, weights):
        if isinstance(weights, dict):
            for name, layer in self.layers.items():
                for p, w in zip(layer.parameters(), weights[name]):
                    p.data = w.data
        elif isinstance(weights, list):
            for layer, w in zip(self.layers.values(), weights):
                for p, w in zip(layer.parameters(), w):
                    p.data = w.data
        else:
            raise ValueError('Invalid type for weights')
        
    def my_set_state_dict(self, weights_dict):
        for name, layer in self.layers.items():
            layer.load_state_dict(weights_dict[name])

    def training_step(self, batch, iteration, auxiliary_model=None):
        x, y_pos = batch
        x = x.to(self.device)
        y_pos = y_pos.to(self.device)

        y_random = torch.randint_like(y_pos, 0, self.n_classes - 1)
        y_same = torch.eq(y_random, y_pos)
        y_neg = ((self.n_classes - 1) * y_same) + (~y_same * y_random)

        if auxiliary_model:
            auxiliary_model.eval()
            auxiliary_model = auxiliary_model.to(self.device)
            auxiliary_input = [torch.cat([x, x]), torch.cat([y_pos, y_neg])]
            auxiliary_output = auxiliary_model(auxiliary_input, return_output=True)
            auxiliary_model = auxiliary_model.to('cpu')
            delta, loss, pos, neg = self(auxiliary_output.detach(), iteration)
        else:
            delta, loss, pos, neg = self([torch.cat([x, x]), torch.cat([y_pos, y_neg])], iteration)

        losses = {f"loss_{i}": l for i, l in enumerate(loss) if l is not None}
        losses = {**losses, "loss": torch.tensor([l for l in loss if l is not None]).mean()}
        deltas = {f"delta_{i}": a for i, a in enumerate(delta) if a is not None}
        pos = {f"pos_{i}": a for i, a in enumerate(pos) if a is not None}
        neg = {f"neg_{i}": a for i, a in enumerate(neg) if a is not None}

        metrics = {**losses, **deltas, **pos, **neg}
        return metrics
    
    def my_predict(self, x, y, auxiliary_model=None):
        x = x.to(self.device)
        y = y.cpu()

        if auxiliary_model:
            auxiliary_model.eval()
            auxiliary_model = auxiliary_model.to(self.device)
            auxiliary_output = [auxiliary_model([x, torch.full_like(y, label).to(self.device)], return_output=True
                                                ).detach() for label in range(self.n_classes)]
            auxiliary_model = auxiliary_model.to('cpu')
            per_label = [self(ao).cpu().detach() for ao in auxiliary_output]
        else:
            per_label = [self([x, torch.full_like(y, label).to(self.device)]).cpu().detach() for label in range(self.n_classes)]
        per_label = torch.stack(per_label)

        per_layer = [self.accuracy(per_label[:, layer].argmax(0), y) for layer in range(self.num_layers)]
        per_layer = torch.stack(per_layer)

        y_hat = per_label[:, -1:].mean(1).argmax(0)

        return y_hat, per_layer
    
    def batch_evaluate(self, x, y, compure_report=False):

        y_hat, per_layer = self.my_predict(x, y)
        accuracy = self.accuracy(y_hat, y)
        accuracies = {f"accuracy_{i}": a for i, a in enumerate(per_layer)}
        metrics = {**accuracies, "accuracy": accuracy}
        
        if compure_report:
            report = classification_report(y.numpy(), y_hat.numpy(), labels=self.n_classes, output_dict=True, zero_division=np.nan)
            metrics["report"] = report

        return metrics
    
    def evaluate(self, data_loader, device=None, multiprocess_dict=None, auxiliary_model=None):
        metrics = {}
        self.eval()
        if device: self.to(device)
        all_y, all_y_hat = [], []
        for batch in data_loader:
            x,y = batch
            y_hat, per_layer = self.my_predict(x, y, auxiliary_model=auxiliary_model)
            accuracy = self.accuracy(y_hat, y)
            accuracies = {f"accuracy_{i}": a for i, a in enumerate(per_layer)}
            metrics_batch = {**accuracies, "accuracy": accuracy, "global_accuracy": per_layer.mean()}
            
            all_y.append(y.numpy())
            all_y_hat.append(y_hat.numpy())

            for key, value in metrics_batch.items():
                metrics.setdefault(key, []).append(value)
        
        all_y = np.concatenate(all_y, axis=0)
        all_y_hat = np.concatenate(all_y_hat, axis=0)

        metrics = {key: torch.tensor(value).mean(0) for key, value in metrics.items()}
        report = classification_report(all_y, all_y_hat, output_dict=True, zero_division=np.nan)
        metrics["report"] = report

        if multiprocess_dict is not None:
            for km, vm in metrics.items():
                multiprocess_dict[km] = vm

        return metrics

    def test_step(self, batch, auxiliary_model=None):
        if isinstance(batch, torch.utils.data.DataLoader):
            return self.evaluate(batch, auxiliary_model=auxiliary_model) 
        else:
            raise NotImplementedError
    
    def compare_models(self, model_2):
        models_differ = 0
        for key_item_1, key_item_2 in zip(self.state_dict().items(), model_2.state_dict().items()):
            if key_item_1[1].device == key_item_2[1].device  and torch.equal(key_item_1[1], key_item_2[1]):
                pass
            else:
                models_differ += 1
                if (key_item_1[0] == key_item_2[0]):
                    _device = f'device {key_item_1[1].device}, {key_item_2[1].device}' if key_item_1[1].device != key_item_2[1].device else ''
                    print(f'Mismtach {_device} found at', key_item_1[0])
                else:
                    raise Exception
        if models_differ == 0:
            print('Models match perfectly! :)')

    def preprare_fedprox(self):
        for layer in self.layers.values():
            layer.fixed_parameters = [p.clone().detach() for p in layer.params]
        

class My_Trainer:
    def __init__(self, epochs, number_client, model, auxiliary_model=None, verbose=False):
        self.epochs = epochs
        self.number_client = number_client
        self.model = model
        self.auxiliary_model = auxiliary_model
        self.verbose = verbose

    def fit(self, data):
        for epoch in range(1, self.epochs+1):
            metrics_batch = {}
            self.model.train()
            for batch in data:
                metrics = self.model.training_step(batch, epoch, auxiliary_model=self.auxiliary_model)
                for key, value in metrics.items():
                    metrics_batch.setdefault(key, []).append(value)

        metrics_train = {key: torch.tensor(value).mean(0) for key, value in metrics_batch.items()}
        

        metrics_test = self.model.test_step(data, auxiliary_model=self.auxiliary_model)
        if self.verbose:
            print(f"Client {self.number_client} - Loss: {metrics_train['loss']} ", end=" ")
            print(f"Accuracy: {metrics_test['accuracy']}")
        

        intersection_key = set(metrics_train.keys()).intersection(set(metrics_test.keys()))
        assert len(intersection_key) == 0, f'The dictionary have in common {intersection_key}'
        return {**metrics_train, **metrics_test}
    
# %%

