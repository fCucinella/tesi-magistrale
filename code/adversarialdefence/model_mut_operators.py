''' original code from https://github.com/tkuo-tkuo/DeepMutationOperators/blob/master/model_mut_operators.py '''


import tensorflow as tf
import numpy as np
import keras

import math, random

import adversarialdefence.utils as utils

class ModelMutationOperatorsUtils():

    def __init__(self):
        self.utils = utils.GeneralUtils()
        self.LD_mut_candidates = ['Dense']
        self.LAm_mut_candidates = ['Dense']

    def GF_on_list(self, lst, mutation_ratio, prob_distribution, STD, lower_bound, upper_bound, lam):
        copy_lst = lst.copy()
        number_of_data = len(copy_lst)
        permutation = utils.GeneralUtils().generate_permutation(number_of_data, mutation_ratio)

        if prob_distribution == 'normal':
            copy_lst[permutation] += np.random.normal(scale=STD, size=len(permutation))
        elif prob_distribution == 'uniform':
            copy_lst[permutation] += np.random.uniform(low=lower_bound, high=upper_bound, size=len(permutation))
        elif prob_distribution == 'exponential':
            assert lam != 0
            scale = 1 / lam 
            copy_lst[permutation] += np.random.exponential(scale=scale, size=len(permutation))
        else:
            pass

        return copy_lst

    def WS_on_Dense_list(self, lst, output_index):
        copy_lst = lst.copy()
        grabbed_lst = copy_lst[:, output_index]
        shuffle_grabbed_lst = utils.GeneralUtils().shuffle(grabbed_lst)
        copy_lst[:, output_index] = shuffle_grabbed_lst
        return copy_lst

    def WS_on_Conv2D_list(self, lst, output_channel_index):
        copy_lst = lst.copy()
        filter_width, filter_height, num_of_input_channels, num_of_output_channels = copy_lst.shape

        copy_lst = np.reshape(copy_lst, (filter_width * filter_height * num_of_input_channels, num_of_output_channels))
        grabbled_lst = copy_lst[:, output_channel_index]
        shuffle_grabbed_lst = utils.GeneralUtils().shuffle(grabbled_lst)
        copy_lst[:, output_channel_index] = shuffle_grabbed_lst
        copy_lst = np.reshape(copy_lst, (filter_width, filter_height, num_of_input_channels, num_of_output_channels))
        return copy_lst

class ModelMutationOperators():
    
    def __init__(self):
        self.utils = utils.GeneralUtils()
        self.model_utils = utils.ModelUtils()
        self.check = utils.ExaminationalUtils()
        self.MMO_utils = ModelMutationOperatorsUtils()

    def GF_mut(model, mutation_ratio, prob_distribution='normal', STD=0.1, lower_bound=None, upper_bound=None, lam=None, mutated_layer_indices=None):
        utils.ExaminationalUtils().mutation_ratio_range_check(mutation_ratio)  
        
        valid_prob_distribution_types = ['normal', 'uniform', 'exponential']
        assert prob_distribution in valid_prob_distribution_types, 'The probability distribution type ' + prob_distribution + ' is not implemented in GF mutation operator'
        if prob_distribution == 'uniform' and ((lower_bound is None) or (upper_bound is None)):
            raise ValueError('In uniform distribution, users are required to specify the lower bound and upper bound of noises')
        if prob_distribution == 'exponential' and (lam is None):
            raise ValueError('In exponential distribution, users are required to specify the lambda value')

        GF_model = utils.ModelUtils().model_copy(model, 'GF')
        layers = [l for l in GF_model.layers]

        num_of_layers = len(layers)
        utils.ExaminationalUtils().valid_indices_of_mutated_layers_check(num_of_layers, mutated_layer_indices)
        layers_should_be_mutated = utils.ModelUtils().get_booleans_of_layers_should_be_mutated(num_of_layers, mutated_layer_indices)

        for index, layer in enumerate(layers):
            weights = layer.get_weights()
            new_weights = []
            if not (len(weights) == 0) and layers_should_be_mutated[index]:
                for val in weights:
                    val_shape = val.shape
                    flat_val = val.flatten()
                    GF_flat_val = ModelMutationOperatorsUtils().GF_on_list(flat_val, mutation_ratio, prob_distribution, STD, lower_bound, upper_bound, lam)
                    GF_val = GF_flat_val.reshape(val_shape)
                    new_weights.append(GF_val)
                layer.set_weights(new_weights) 

        return GF_model

    def WS_mut(model, mutation_ratio, mutated_layer_indices=None):
        utils.ExaminationalUtils().mutation_ratio_range_check(mutation_ratio)

        WS_model = utils.ModelUtils().model_copy(model, 'WS')
        layers = [l for l in WS_model.layers]

        num_of_layers = len(layers)
        utils.ExaminationalUtils().valid_indices_of_mutated_layers_check(num_of_layers, mutated_layer_indices)
        layers_should_be_mutated = utils.ModelUtils().get_booleans_of_layers_should_be_mutated(num_of_layers, mutated_layer_indices)

        for index, layer in enumerate(layers):
            weights = layer.get_weights()
            layer_name = type(layer).__name__
            new_weights = []
            if not (len(weights) == 0):
                for val in weights:
                    val_shape = val.shape
                    if (len(val.shape) != 1) and layers_should_be_mutated[index]:
                        if layer_name == 'Conv2D':
                            filter_width, filter_height, num_of_input_channels, num_of_output_channels = val_shape
                            permutation = utils.GeneralUtils().generate_permutation(num_of_output_channels, mutation_ratio)
                            for output_channel_index in permutation:
                                val = ModelMutationOperatorsUtils().WS_on_Conv2D_list(val, output_channel_index)
                        elif layer_name == 'Dense':
                            input_dim, output_dim = val_shape
                            permutation = utils.GeneralUtils().generate_permutation(output_dim, mutation_ratio)
                            for output_dim_index in permutation:
                                val = ModelMutationOperatorsUtils().WS_on_Dense_list(val, output_dim_index)
                        else:
                            pass
                    new_weights.append(val)
                layer.set_weights(new_weights)

        return WS_model

    def NAI_mut(model, mutation_ratio, mutated_layer_indices=None):
        utils.ExaminationalUtils().mutation_ratio_range_check(mutation_ratio)

        NAI_model = utils.ModelUtils().model_copy(model, 'NAI')
        layers = [l for l in NAI_model.layers]

        num_of_layers = len(layers)
        utils.ExaminationalUtils().valid_indices_of_mutated_layers_check(num_of_layers, mutated_layer_indices)
        layers_should_be_mutated = utils.ModelUtils().get_booleans_of_layers_should_be_mutated(num_of_layers, mutated_layer_indices)

        for index, layer in enumerate(layers):
            weights = layer.get_weights()
            layer_name = type(layer).__name__
            new_weights = []
            if not (len(weights) == 0):
                for val in weights:
                    val_shape = val.shape
                    if (len(val.shape) != 1) and layers_should_be_mutated[index]:
                        if layer_name == 'Conv2D':
                            filter_width, filter_height, num_of_input_channels, num_of_output_channels = val_shape
                            permutation = utils.GeneralUtils().generate_permutation(num_of_output_channels, mutation_ratio)
                            for output_channel_index in permutation:
                                val[:, :, :, output_channel_index] *= -1
                        elif layer_name == 'Dense':
                            input_dim, output_dim = val_shape
                            permutation = utils.GeneralUtils().generate_permutation(output_dim, mutation_ratio)
                            for output_dim_index in permutation:
                                val[:, output_dim_index] *= -1
                        else:
                            pass 
                    new_weights.append(val)
                layer.set_weights(new_weights)

        return NAI_model