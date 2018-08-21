import torch
import torch.nn.functional as F
import numpy as np
import scipy.stats as st
from tqdm import tqdm


def recall_at_k_np(scores, ks=[1, 2, 3, 4, 5]):
    """
    Evaluation recalll
    :param scores:  sigmoid scores
    :param ks:
    :return:
    """
    #sort the scores
    sorted_idxs = np.argsort(-scores, axis=1)
    ranks = (sorted_idxs == 0).argmax(1)
    recalls = [np.mean(ranks+1 <= k) for k in ks]
    return recalls


def eval_model(model, dataset, mode='valid', gpu=False, no_tqdm=False):
    """
    evaluation for DKE-GRU and AddGRU
    :param model:
    :param dataset:
    :param mode:
    :param gpu:
    :param no_tqdm:
    :return:
    """
    model.eval()
    scores = []

    assert mode in ['valid', 'test']

    data_iter = dataset.get_iter(mode)

    if not no_tqdm:
        data_iter = tqdm(data_iter)
        data_iter.set_description_str('Evaluation')
        n_data = dataset.n_valid if mode == 'valid' else dataset.n_test
        data_iter.total = n_data // dataset.batch_size

    for mb in data_iter:
        context, response, y, cm, rm, key_r, key_mask_r = mb

        # Get scores
        scores_mb = F.sigmoid(model(context, response, cm, rm, key_r, key_mask_r)) #Appropritate this line while running different models.
        scores_mb = scores_mb.cpu() if gpu else scores_mb
        scores.append(scores_mb.data.numpy())

    scores = np.concatenate(scores)
    
    # Handle the case when numb. of data not divisible by 10
    mod = scores.shape[0] % 10
    scores = scores[:-mod if mod != 0 else None]

    scores = scores.reshape(-1, 10)  # 1 in 10
    recall_at_ks = [r for r in recall_at_k_np(scores)]

    return recall_at_ks
