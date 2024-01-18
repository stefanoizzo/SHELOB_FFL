#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Anonymous
"""

# %%
import time
import numpy as np
import tensorflow as tf
from copy import deepcopy
from sklearn.metrics import classification_report

# %%

def train(model, train_dataset, test_dataset, auxiliary_model=None, verbose=True):
    losses = []; ces = []; tec = []; accs = []; l2_losses = []

    for epoch in range(1, model.epochs+1):
        t0 = time.time()

        # Train the model
        model, running_loss, running_l2_loss, running_ce = \
            train_epoch(model, train_dataset, auxiliary_model) #, start_block)#, idxs, block_choice)

        t1 = time.time()-t0

        # Compute accuracy on the test set.
        # Split in batches
        if verbose:
            acc = []
            if test_dataset is None:
                for x_train_batch, y_train_batch in train_dataset.take(1):
                    acc.append(model.my_predict(x_train_batch, y_train_batch, return_acc=True, auxiliary_model=auxiliary_model)[1])
            else:
                for x_test_batch, y_test_batch in test_dataset:
                    acc.append(model.my_predict(x_test_batch, y_test_batch, return_acc=True, auxiliary_model=auxiliary_model)[1])
            acc = tf.reduce_mean(acc)

            t2 = time.time()-t0

            print(f"{t1:.2f} sec | {t2:.2f} sec -->",
            # print(f"{t1:.2f} sec -->",
                    f"Step {epoch:3d} | Loss: {running_loss:7.4f} | CE: {running_ce:7.4f} | ACCURACY: {acc.numpy()*100:3.2f}")
            accs.append(acc.numpy())

        losses.append(running_loss.numpy())
        l2_losses.append(running_l2_loss.numpy())
        ces.append(running_ce.numpy())
        tec.append(t1)

    accs = []
    preds_array = []
    y_array = []

    if test_dataset is None:
        for x_train_batch, y_train_batch in train_dataset:
            preds, acc, y_true = model.my_predict(x_train_batch, y_train_batch, 
                            return_acc=True, return_true=True, auxiliary_model=auxiliary_model)
            preds_array.append(preds)
            accs.append(acc)
            y_array.append(y_true)
    else:
        for x_test_batch, y_test_batch in test_dataset:
            preds, acc, y_true = model.my_predict(x_test_batch, y_test_batch, 
                        return_acc=True, return_true=True, auxiliary_model=auxiliary_model)
            preds_array.append(preds)
            accs.append(acc)
            y_array.append(y_true)

    accs = tf.reduce_mean(accs)
    y_array = np.concatenate(y_array)
    preds_array = np.concatenate(preds_array)
    n_label = list(range(model.n_classes))
    report = classification_report(y_array, preds_array, labels=n_label, output_dict=True, zero_division=np.nan)

    return model, losses, l2_losses, accs, ces, tec, report


def train_epoch(model, train_loader, auxiliary_model):
    running_loss = tf.constant(0.)
    running_ce = tf.constant(0.)
    running_l2_loss = tf.constant(0.)

    for x, y_pos in train_loader:
        # Check if the labels are one-hot encoded
        if len(y_pos.shape) == 2: 
            y_pos_one_hot = y_pos
            y_pos = tf.cast(tf.argmax(y_pos_one_hot, axis=1), dtype=tf.uint8)
        else:
            # y_pos_one_hot = tf.one_hot(y_pos, self.n_classes)
            y_pos_one_hot = tf.one_hot(y_pos, depth=model.n_classes)

        if model.training_mode == 'dense':
            x = tf.reshape(x, [x.shape[0], -1])
            
        elif model.training_mode in ['conv']:
            if len(x.shape)==3:
                x = tf.expand_dims(x, axis=-1)

        if model.training_mode == 'dense':
            x_pos = tf.concat([x, y_pos_one_hot], axis=1)
            # Create the uniform samples
            x_neg = tf.concat([x, tf.ones_like(y_pos_one_hot)*(1/model.n_classes)], axis=1)

        elif model.training_mode in ['conv']:
            x_pos = tf.Variable(x, trainable=False)
            x_pos = tf.constant(x_pos[:, :model.n_classes, 0, :].assign(tf.stack([y_pos_one_hot]*x.shape[-1], axis=-1)))
            # Create the uniform samples
            x_neg = tf.Variable(x, trainable=False)
            x_neg = tf.constant(x_neg[:, :model.n_classes, 0, :].assign(tf.stack([tf.ones_like(y_pos_one_hot)*(1/model.n_classes)]*x.shape[-1], axis=-1)))
            
        # Compute predictions of the network
        if auxiliary_model is not None:
            with tf.device('cpu:0'):
                x_neg = auxiliary_model(x_neg, last=True)
                x_pos = auxiliary_model(x_pos, last=True)
        
        ys = model(x_neg)
        ys = tf.concat([tf.reshape(k, [k.shape[0], -1]) for k in ys], axis=-1)
        
        # Compute cross-entropy loss
        with tf.GradientTape() as tape:
            logits = model.linear_cf(ys)
            ce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(y_pos, logits)
            running_ce += ce
        
        # Compute the gradient of the cross-entropy loss
        grads = tape.gradient(ce, model.linear_cf.trainable_variables)
        model.optimizer_cf.apply_gradients(zip(grads, model.linear_cf.trainable_variables))

        # Negative pairs from softmax layer
        probs = tf.nn.softmax(logits, axis=1)
        preds = tf.cast(tf.argmax(probs, axis=1), dtype=tf.uint8)
        idx = tf.where(preds != y_pos)

        # Negative pairs from random labels
        y_rand = tf.cast(tf.random.uniform([x.shape[0]], minval=0, 
                        maxval=model.n_classes, dtype=tf.int32), dtype=tf.uint8)
        # idx_rand = tf.where(y_rand != y_pos) # incorrect labels
        y_rand_one_hot = tf.one_hot(y_rand, model.n_classes)

        if model.training_mode == 'dense':
            x_rand = tf.concat([x, y_rand_one_hot], axis=1) #[idx_rand] # keeping positives seems to work better
            x_neg = x_rand

        elif model.training_mode in ['conv']:
            x_rand = tf.Variable(x, trainable=False)
            x_rand = tf.constant(
                x_rand[:, :model.n_classes, 0, :].assign(
                    tf.stack([y_rand_one_hot]*x.shape[-1], axis=-1)))
            # x_rand = tf.gather(x_rand, idx_rand[:,0]) # keeping positives seems to work better
            x_neg = x_rand

        if model.hard_negatives:
            y_hard_one_hot = tf.one_hot(preds, model.n_classes)
            if model.training_mode == 'dense':
                # taking the wrong predicted samples
                x_hard = tf.gather(tf.concat([x, y_hard_one_hot], axis=1), idx[:,0])

            elif model.training_mode in ['conv']:
                # taking the wrong predicted samples
                x_hard = tf.Variable(x, trainable=False)
                x_hard = tf.constant(x_hard[:, :model.n_classes, 0, :].assign(tf.stack([y_hard_one_hot]*x.shape[-1], axis=-1)))
                x_hard = tf.gather(x_hard, idx[:,0])

            x_neg = tf.concat([x_neg, x_hard], axis=0)       

        if auxiliary_model is not None:
            with tf.device('cpu:0'):
                x_neg = auxiliary_model(x_neg, last=True)
        
        # Perform the optimizations
        running_loss_partial, running_l2_loss_partial = model.train_step(x_pos, x_neg) # , start_block)#, block_choice, idxs)
        running_loss += running_loss_partial
        running_l2_loss += running_l2_loss_partial

    running_loss /= tf.cast(len(train_loader), dtype=tf.float32)
    running_ce /= tf.cast(len(train_loader), dtype=tf.float32)
    running_l2_loss /= tf.cast(len(train_loader), dtype=tf.float32)

    return model, running_loss, running_l2_loss, running_ce

# @tf.function
def compute_loss_norm(tensor, theta, s=1.):
    return tf.reduce_mean(tf.math.log(1 + tf.math.exp(s*(-tf.norm(tf.reshape(tensor, [tensor.shape[0], -1]), axis=-1) + theta))))

# @tf.function
def train_step_out(self, x_pos, x_neg):
    acc_loss = tf.constant(0.)
    acc_l2_loss = tf.constant(0.)

    # Evaluate the Gradients and perform the local BackPropagation
    with tf.GradientTape() as tape:
        z_pos = self(x_pos)
        z_neg = self(x_neg)

        trainable_variables = [b.trainable_variables for b in self.blocks]
        
        losses = []
        losses_model = []
        losses_l2_norm = []

        for zp, zn, tv, gp in zip(z_pos, z_neg, trainable_variables, self.glob_parameters):
            positive_loss = compute_loss_norm(zp, self.theta, 1.)
            negative_loss = compute_loss_norm(zn, self.theta, -1.)

            if self.fedprox and gp is not None: 
                l2_norm = tf.keras.backend.epsilon()
                
                for glob_param, param in zip(gp, tv):
                # for i_gp in range(len(gp)):
                    # l2_norm += tf.reduce_mean(tf.square(glob_param - param))
                    l2_norm += tf.reduce_sum(tf.square(glob_param - param))
                    # l2_norm += tf.norm(gp[i_gp] - tv[i_gp], ord=2)
                l2_norm *= self.mu/2
                # acc_l2_loss += self.mu/2 * l2_norm
                acc_l2_loss += l2_norm
                loss_model = (positive_loss + negative_loss) 
                loss_l2_norm = l2_norm
                loss = loss_model + loss_l2_norm   
            else:
                loss_l2_norm = tf.constant(0.)
                loss = positive_loss + negative_loss 
                loss_model = loss            

            acc_loss += loss
            losses.append(loss)
            losses_model.append(loss_model)
            losses_l2_norm.append(loss_l2_norm)
    
    # if self.glob_parameters is not None: 
    #     # params = self.my_get_weights(block_style=True)[0]
    #     for i_loss in range(len(losses)):
    #         l2_norm = 0; acc_l2_loss = 0
    #         for glob_param, param in zip(self.glob_parameters[i_loss], trainable_variables[i_loss]):
    #             l2_norm += tf.norm(glob_param - param, ord=2)
    #         acc_l2_loss += self.mu/2 * l2_norm
    #         losses[i_loss] += self.mu/2 * l2_norm

    # Compute the gradient of the loss
    grads = tape.gradient(losses, trainable_variables)
    # Check for debug
    # grads_model = tape.gradient(losses_model, trainable_variables)
    # grads_norm = tape.gradient(losses_l2_norm, trainable_variables)
    # grads = [grads_model[i] + grads_norm[i] for i in range(len(grads_model))]

    [opt.apply_gradients(zip(grad, var)) 
            for grad, var, opt in 
                    zip(grads, 
                    #    [b.trainable_variables for b in network.blocks],
                    trainable_variables,
                    self.optimizers)
    ]

    return acc_loss/len(losses), acc_l2_loss/len(losses)

def predict_out(self, x_test, y_test, return_acc=False, return_true=False, 
                on_first_block=False, auxiliary_model=None):
    # Check if the labels are one-hot encoded
    if len(y_test.shape) == 2: 
        y_test_one_hot = y_test
        y_test = tf.cast(tf.argmax(y_test_one_hot, axis=1), dtype=tf.uint8)

    if self.training_mode == 'dense':
        sample = tf.reshape(x_test, [x_test.shape[0], -1])
    elif self.training_mode in ['conv']:
        if len(x_test.shape)==3:
            sample = tf.expand_dims(x_test, axis=-1)
        else:
            sample = x_test
        
    acts_for_labels = []
    for label in range(self.n_classes):
        test_label = tf.ones(y_test.shape, dtype=tf.uint8) * label 
        test_label = tf.one_hot(test_label, depth=self.n_classes)
        
        if self.training_mode == 'dense':
            x_with_labels = tf.concat([sample, test_label], axis=1)

        elif self.training_mode in ['conv']:
            x_with_labels = tf.Variable(sample)
            x_with_labels = tf.constant(x_with_labels[:, :self.n_classes, 0, :].assign(tf.stack([test_label]*sample.shape[-1], axis=-1)))

        if auxiliary_model is not None:
            with tf.device('cpu:0'):
                x_with_labels = auxiliary_model(x_with_labels, last=True)
        
        acts = self(x_with_labels)
        if on_first_block:
            acts = tf.expand_dims(tf.norm(tf.reshape(
                acts[0], [acts[0].shape[0], -1]), ord='euclidean', axis=-1), axis=1)
        else:
            acts = tf.stack(
                [tf.norm(tf.reshape(tensor, [tensor.shape[0], -1]), ord='euclidean', axis=-1) for tensor in acts], axis=1
                )
        acts_for_labels.append(acts)

    acts_for_labels = tf.stack(acts_for_labels, axis=1)    
    preds = tf.argmax(tf.reduce_mean(acts_for_labels, axis=-1), axis=1)

    output = []
    output.append(preds.numpy())
    #Compute accuracy between preds and y_test
    if return_acc:
        preds = tf.cast(preds, dtype=tf.uint8)
        acc = tf.reduce_mean(tf.cast(tf.equal(preds, y_test), dtype=tf.float32))
        output.append(acc.numpy())
        output[0] = preds.numpy()

    if return_true:
        output.append(y_test.numpy())

    return output

def accuracy(output, target, topk=(1,)):
    # Computes the accuracy over the k top predictions for the specified values of k
    maxk = max(topk)
    batch_size = target.shape[0]

    _, pred = tf.math.top_k(output, maxk, sorted=True)
    pred = tf.cast(tf.transpose(pred), tf.uint8)
    correct = tf.math.equal(pred, tf.expand_dims(target,0))

    res = []
    for k in topk:
        correct_k = tf.cast(tf.reshape(correct[:k], -1), tf.float32)
        correct_k = tf.reduce_sum(correct_k)
        res.append(correct_k * (100.0 / batch_size))
    return res

# %%

class L2_normalization(tf.keras.layers.Layer):
    def __init__(self, axis=1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, x):
        x = tf.math.l2_normalize(x, axis=self.axis)
        return x

# class Block(tf.keras.models.Model):
class Block(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        # bias = layers_config.get('bias', False)
        self.model_layers = []
        for k_lc, v_lc in config.items():
            if k_lc.startswith('Input'):
                self.model_layers.append(tf.keras.layers.InputLayer(**v_lc))
                self.input_layer = v_lc['input_shape']
            elif k_lc.startswith('Conv2D'):
                self.model_layers.append(tf.keras.layers.Conv2D(**v_lc))
            elif k_lc.startswith('Dense'):
                self.model_layers.append(tf.keras.layers.Dense(**v_lc))
            elif k_lc.startswith('Reshape'):
                self.model_layers.append(tf.keras.layers.Reshape(**v_lc))
            elif k_lc.startswith('Flatten'):
                self.model_layers.append(tf.keras.layers.Flatten(**v_lc))
            elif k_lc.startswith('L2_normalization'):
                self.model_layers.append(L2_normalization(**v_lc))
            elif k_lc.startswith('MaxPooling2D'):
                self.model_layers.append(tf.keras.layers.MaxPooling2D(**v_lc))
            elif k_lc.startswith('AveragePooling2D'):
                self.model_layers.append(tf.keras.layers.AveragePooling2D(**v_lc))

        self.relu = tf.keras.layers.ReLU()

    def call(self, x):
        for l in self.model_layers:
            x = l(x)

        self.x = x
        return self.relu(x)

class Network(tf.keras.Model):
    def __init__(self, configuration, **kwargs):
    
        self.layers_in_block = kwargs.pop('layers_in_block', 3)
        self.n_classes = kwargs.pop('n_classes', None)
        self.hard_negatives = kwargs.pop('hard_negatives', True)
        self.theta = kwargs.pop('theta', 10.)
        self.lr = kwargs.pop('lr', 1e-4)
        self.epochs = kwargs.pop('epochs', 10)
        self.configuration = deepcopy(configuration)
        self.n_global_blocks = kwargs.pop('n_blocks', 5)
        self.idx_blocks = kwargs.pop('idx_blocks', list(range(self.n_global_blocks)))
        self.glob_parameters = kwargs.pop('global_parameters', None)
        self.mu = tf.constant(kwargs.pop('mu', 0.1))
        self.fedprox = kwargs.pop('fedprox', False)

        super().__init__(**kwargs)

        self.training_mode = configuration['train_mode']

        self.blocks = []
        for block in configuration.keys():
            if block.startswith('Block') and int(block.split('_')[-1]) in self.idx_blocks:
                self.blocks.append(tf.keras.Sequential(Block(configuration[block]), name=block))

        self.configuration['Input'] = self.blocks[0].layers[0].input_layer
        self.n_blocks = len(self.blocks)
        
        self.optimizers = [tf.optimizers.Adam(self.lr) for _ in range(self.n_blocks)]

        self.linear_cf = tf.keras.layers.Dense(self.n_classes)
        self.optimizer_cf = tf.optimizers.Adam(learning_rate=1e-4)

        self.my_predict = lambda x_test, y_test, return_acc=False, return_true=False, on_first_block=False, auxiliary_model=None: predict_out(self, x_test, y_test, return_acc, return_true, on_first_block, auxiliary_model)

    def call(self, x, y=None, last=False):
        # if y is not None:
        #     if reshape:
        #         x = tf.reshape(x, [x.shape[0], -1])
        #     x = tf.concat([x, y], axis=1)
        x = self.prepare_data(x, y)
        
        # x = self.input_layer(x)
        for b in self.blocks:
            x = b(x)
        
        if last:
            return x
        
        # xs = [b.x for b in self.blocks]
        xs = [b.layers[0].x for b in self.blocks]

        return xs

    # @tf.function
    def train_step(self, x_pos, x_neg): #, start_block):
        return train_step_out(self, x_pos, x_neg) #, start_block)
    
    def prepare_data(self, x, y=None):
        if x.shape[1:] != self.configuration['Input']:
            if len(self.configuration['Input'])==1:
                x = tf.reshape(x, [x.shape[0], -1])
            else:
                x = tf.reshape(x, [x.shape[0], *self.configuration['Input']])
        if y is not None:
            if len(self.configuration['Input'])==1:
                x = tf.concat([x, y], axis=1)
            elif len(self.configuration['Input']) > 1:
                if len(y.shape)==1:
                    y = tf.one_hot(y, self.n_classes)

                x = tf.Variable(x, trainable=False)
                stack_y = tf.stack([y]*x.shape[-1], axis=-1)
                # if len(x.shape) == 4:
                x = tf.constant(x[:, :self.n_classes, 0, :].assign(stack_y))
                # elif len(x.shape) == 5:
                #     x = tf.constant(x[:, :self.n_classes, 0, 0, 0:1].assign(stack_y))
                # print(x.shape)
        return x

    def my_build(self, example_data, parameters=None):
        t0 = time.time()
        if isinstance(example_data, tf.data.Dataset):
            x_e, y_e = next(iter(example_data))
        elif isinstance(example_data, tuple):
            x_e, y_e = example_data
        elif isinstance(example_data, tf.Tensor):
            x_e = example_data
            y_e = None
        else:
            raise ValueError('example_data must be a tf.data.Dataset or a tuple (x, y)')
        
        x_e = self.prepare_data(x_e, y_e)
        self(x_e)

        if parameters is not None:
            self.my_set_weights(parameters)

    def my_get_weights(self, block_style=False):
        weights_dict = {}
        for b in self.blocks:
            weights_dict[b.name] = b.get_weights()
        if block_style:
            weights_list = []
            for b in self.blocks:
                weights_list.append(b.get_weights())
        else:
            weights_list = []
            for b in self.blocks:
                weights_list += b.get_weights()
        
        return weights_list, weights_dict
    
    def my_set_weights(self, weights):
        if isinstance(weights, dict):
            for b in self.blocks:
                b.set_weights(weights[b.name])
        elif isinstance(weights, list):
            if len(weights) == len(self.blocks):
                for b, w in zip(self.blocks, weights):
                    b.set_weights(w)
            else:
                self.set_weights(weights)
        else:
            raise ValueError('weights must be a dict or a list')
        
    def my_get_config(self):
        return self.configuration

# %%