import numpy as np
import pandas as pd
import keras

import random, math 
import unittest

class GeneralUtils():

    def __init__(self):
        pass

    ''' 
    Return True with prob
    Input: probability within [0, 1]
    Ouput: True or False 
    '''
    def decision(self, prob):
        assert prob >= 0, 'Probability should in the range of [0, 1]'
        assert prob <= 1, 'Probability should in the range of [0, 1]'
        return random.random() < prob

    def generate_permutation(self, size_of_permutation, extract_portion):
        assert extract_portion <= 1
        num_of_extraction = math.floor(size_of_permutation * extract_portion)
        permutation = np.random.permutation(size_of_permutation)
        permutation = permutation[:num_of_extraction]
        return permutation

    def shuffle(self, a):
        shuffled_a = np.empty(a.shape, dtype=a.dtype)
        length = len(a)
        permutation = np.random.permutation(length)
        index_permutation = np.arange(length)
        shuffled_a[permutation] = a[index_permutation]
        return shuffled_a

    def get_data_with_advs(advs_path):
        df_advs = pd.read_csv(advs_path).assign(Label=1)
        df_advs.drop(df_advs.columns[0], axis=1, inplace=True)
        
        X = df_advs.copy()
        y = X.pop('Label')
        
        return df_advs, X, y

    def get_advs_samples(preds, df):
        df = df.assign(Pred=preds)
        df_advs = df[df['Pred'] != df['Label']]    
        df_advs = df_advs.drop(['Pred'], axis=1)
        
        return df_advs


class ModelUtils():

    def __init__(self):
        pass

    def model_copy(self, model, mode=''):
        original_layers = [l for l in model.layers]
        suffix = '_copy_' + mode 
        new_model = keras.models.clone_model(model)
        for index, layer in enumerate(new_model.layers):
            original_layer = original_layers[index]
            original_weights = original_layer.get_weights()
            layer.set_weights(original_weights)
        return new_model

    def get_booleans_of_layers_should_be_mutated(self, num_of_layers, indices):
        if indices == None:
            booleans_for_layers = np.full(num_of_layers, True)
        else:
            booleans_for_layers = np.full(num_of_layers, False)
            for index in indices:
                booleans_for_layers[index] = True
        return booleans_for_layers 
    
    def binary_preds_supervised(model, X, threshold=0.5):
        try:
            preds = model.predict(np.array(X), verbose=False)
        except:
            preds = model.predict(np.array([X, ]), verbose=False)

        preds[preds > threshold] = 1
        preds[preds <= threshold] = 0
        
        return preds[:, 1]

    def binary_preds_unsupervised(model, X, threshold=0.05):
        try:
            preds = model.predict(np.array(X), verbose=False)
        except:
            preds = model.predict(np.array([X, ]), verbose=False)

        mse = np.mean(np.power(X - preds, 2), axis=1)

        recons_df = pd.DataFrame({
            'error': mse,
        }).reset_index(drop=True)

        recons_df['y_pred'] = recons_df['error'] > threshold
        
        return recons_df['y_pred']


class ExaminationalUtils():

    def __init__(self):
        pass

    def mutation_ratio_range_check(self, mutation_ratio):
        assert mutation_ratio >= 0, 'Mutation ratio attribute should in the range [0, 1]'
        assert mutation_ratio <= 1, 'Mutation ratio attribute should in the range [0, 1]'
        pass 

    def valid_indices_of_mutated_layers_check(self, num_of_layers, indices):
        if indices is not None:
            for index in indices:
                assert index >= 0, 'Index should be positive'
                assert index < num_of_layers, 'Index should not be out of range, where index should be smaller than ' + str(num_of_layers)
                pass 

    def in_suitable_indices_check(self, suitable_indices, indices):
        if indices is not None:
            for index in indices:
                assert index in suitable_indices, 'Index ' + str(index) + ' is an invalid index for this mutation'
                pass 