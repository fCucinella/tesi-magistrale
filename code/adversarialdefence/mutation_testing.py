import os
import logging
import itertools
import shutil
import random
import numpy as np

import adversarialdefence.model_mut_operators as model_mut_operators
import adversarialdefence.utils as utils

from keras.models import load_model

class AdversarialDetectorThroughMutation():

    def __init__(self, orig_model, model_name, model_type: "SUPERVISED|UNSUPERVISED", binary_preds_threshold, dest_path):
        self.orig_model = orig_model
        self.model_name = model_name

        if not model_type == 'SUPERVISED' and not model_type == 'UNSUPERVISED':
            raise Exception('Model type non valid. Choose between [SUPERVISED, UNSUPERVISED]')

        self.model_type = model_type
        self.binary_preds_threshold = binary_preds_threshold
        if not dest_path.endswith('/'):
            dest_path += '/'
        self.dest_path = dest_path + model_name

    def __get_mutated_models(self, num_mutation, mutation_ratio, mutation_operator):
        mutated_models = []
        for i in range(0, num_mutation):
            if mutation_operator == 'NAI':
                mutated_model = model_mut_operators.ModelMutationOperators.NAI_mut(self.orig_model, mutation_ratio)
            elif mutation_operator == 'GF':
                mutated_model = model_mut_operators.ModelMutationOperators.GF_mut(self.orig_model, mutation_ratio)
            elif mutation_operator == 'WS':
                mutated_model = model_mut_operators.ModelMutationOperators.WS_mut(self.orig_model, mutation_ratio)
            else:
                raise Exception('Mutation operator not valid. Choose between [NAI, GF, WS]')
            mutated_models.append(mutated_model)

        return mutated_models

    def __get_LCR_metrics(self, X, mutated_models):
        LCR_list = [None] * X.shape[0]
        num_deflected_models = np.zeros(X.shape[0])

        for mutated_model in mutated_models:
            if self.model_type == 'SUPERVISED':
                orig_labels = utils.ModelUtils.binary_preds_supervised(self.orig_model, X, self.binary_preds_threshold)
                mutated_labels = utils.ModelUtils.binary_preds_supervised(mutated_model, X, self.binary_preds_threshold)
            else:
                orig_labels = utils.ModelUtils.binary_preds_unsupervised(self.orig_model, X, self.binary_preds_threshold)
                mutated_labels = utils.ModelUtils.binary_preds_unsupervised(mutated_model, X, self.binary_preds_threshold)

            for i in range(0, len(orig_labels)):
                if orig_labels[i] != mutated_labels[i]:
                    num_deflected_models[i] = num_deflected_models[i] + 1
        
            for i in range(0, len(LCR_list)):
                LCR_list[i] = num_deflected_models[i] / len(mutated_models)
        
        return sum(LCR_list) / len(LCR_list), max(LCR_list)

    def __generate_mutated_models(self, num_mutation, mutation_ratio):
        mutated_models = []
        logging.getLogger('tensorflow').setLevel(logging.ERROR)

        mutation_op_list = ['NAI', 'GF', 'WS']

        num_mutated_models = round(num_mutation / len(mutation_op_list))
        diff_mutation = num_mutation - (num_mutated_models * len(mutation_op_list))

        for op in mutation_op_list:
            mutated_models.append(self.__get_mutated_models(num_mutated_models, mutation_ratio, op))
        if diff_mutation != 0:
            mutated_models.append(self.__get_mutated_models(diff_mutation, mutation_ratio, 'NAI'))
        
        mutated_models = list(itertools.chain.from_iterable(mutated_models))

        if os.path.exists(self.dest_path):
            shutil.rmtree(self.dest_path)

        for i in range(0, len(mutated_models)):
            mutated_models[i].save(self.dest_path + '/mutated_' + self.model_name + '_' + str(i) + '.hdf5')
        
        return mutated_models

    def __fetch_mutated_model(self):
        i = random.choice(range(0, len(os.listdir(self.dest_path)) - 1))
        model_path = self.dest_path + '/mutated_' + self.model_name + '_' + str(i) + '.hdf5'
        return load_model(model_path, compile=False)
    
    def fit(self, X_norm, num_mutation, mutation_ratio):
        num_attempts = 0
        LCR_up = 0.5
        while LCR_up > 0.4 and num_attempts < 5:
            mutated_models = self.__generate_mutated_models(num_mutation, mutation_ratio)
            _LCR_avg, LCR_up = self.__get_LCR_metrics(X_norm, mutated_models)
            num_attempts += 1
        
        if num_attempts == 5 and LCR_up > 0.4:
            raise Exception('Adversarial detector fitting failed') 
        
        with open(self.dest_path + '/LCR_th.txt', 'w') as f:
            f.write(str(LCR_up))

        return LCR_up

    def __calculate_sprt_ratio(self, LCR_th, c, n, sigma):
        p1 = LCR_th + sigma
        p0 = LCR_th - sigma

        return c * np.log(p1 / p0) + (n - c) * np.log((1 - p1) / (1 - p0))

    def detect(self, X, max_iter, detection_sensibility):
        stop = False
        deflected_mutated_model_count = np.zeros(X.shape[0])
        total_mutated_model_count = 0
        detect_status = [None] * X.shape[0]
        alpha, beta = 0.05, 0.05

        accept_pr = np.log((1 - beta)/alpha)
        deny_pr = np.log(beta / (1 - alpha))

        with open(self.dest_path + '/LCR_th.txt', 'r') as f:
            LCR_th = float(f.readline())

        if LCR_th is None:
            raise Exception('Adversarial detector fitting is required before detection')

        LCR_th = detection_sensibility * LCR_th
        sigma = LCR_th * 0.99

        while (not stop):
            total_mutated_model_count += 1
            
            if total_mutated_model_count > max_iter:
                detect_status = [False if d is None else d for d in detect_status]
                return detect_status
            
            mutated_model = self.__fetch_mutated_model()

            if self.model_type == 'SUPERVISED':
                orig_labels = utils.ModelUtils.binary_preds_supervised(self.orig_model, X, self.binary_preds_threshold)
                new_lables = utils.ModelUtils.binary_preds_supervised(mutated_model, X, self.binary_preds_threshold)
            else:
                orig_labels = utils.ModelUtils.binary_preds_unsupervised(self.orig_model, X, self.binary_preds_threshold)
                new_lables = utils.ModelUtils.binary_preds_unsupervised(mutated_model, X, self.binary_preds_threshold)

            for i in range(0, len(orig_labels)):
                pr = self.__calculate_sprt_ratio(LCR_th, deflected_mutated_model_count[i], total_mutated_model_count, sigma)
                if(orig_labels[i] != new_lables[i]):
                    deflected_mutated_model_count[i] += 1
                    if pr >= accept_pr:
                        detect_status[i] = True
                    if pr <= deny_pr:
                        detect_status[i] = False
            
            if not None in detect_status:
                return detect_status
            
        return