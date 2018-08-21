import os
import torch


def save_model(model, name):
    if not os.path.exists('models/'):
        os.makedirs('models/')

    torch.save(model.state_dict(), 'models/{}.bin'.format(name))


def load_model(model, name, gpu=True):
    if gpu:
        model.load_state_dict(torch.load('models/{}.bin'.format(name)))
    else:
        model.load_state_dict(torch.load('models/{}.bin'.format(name), map_location=lambda storage, loc: storage))

    return model


def clip_gradient_threshold(model, min, max):
    for p in model.parameters():
        if p.grad is not None:
            p.grad.data.clamp_(min, max)
