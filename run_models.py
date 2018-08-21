import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from data import UDCv1
from evaluation import eval_model
from util import save_model, clip_gradient_threshold, load_model
from models import biGRU, A_DE_bigRU, AK_DKE_biGRU, Add_GRU
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(
    description='UDC Experiment Runner'
)

parser.add_argument('--gpu', default=False, action='store_true',
                    help='whether to run in the GPU')
parser.add_argument('--h_dim', type=int, default=100, metavar='',
                    help='hidden dimension (default: 100)')
parser.add_argument('--lr', type=float, default=1e-3, metavar='',
                    help='learning rate (default: 1e-3)')
parser.add_argument('--emb_drop', type=float, default=0.3, metavar='',
                    help='embedding dropout (default: 0.3)')
parser.add_argument('--mb_size', type=int, default=128, metavar='',
                    help='size of minibatch (default: 128)')
parser.add_argument('--n_epoch', type=int, default=500, metavar='',
                    help='number of iterations (default: 500)')
parser.add_argument('--randseed', type=int, default=123, metavar='',
                    help='random seed (default: 123)')
parser.add_argument('--no_tqdm', default=False, action='store_true',
                    help='disable tqdm progress bar')
parser.add_argument('--early_stop', type=int, default=3,
                    help='early stopping')
args = parser.parse_args()

# Set random seed
np.random.seed(args.randseed)
torch.manual_seed(args.randseed)

if args.gpu:
    torch.cuda.manual_seed(args.randseed)

max_seq_len = 320
model_name = 'AK_DKE_biGRU'
#dataset
udc = UDCv1('ubuntu_data', batch_size=args.mb_size, use_mask=True,
            max_seq_len=max_seq_len, gpu=args.gpu, use_fasttext=True)
#model definition
model = AK_DKE_biGRU(
    udc.emb_dim, udc.vocab_size, args.h_dim, udc.vectors, 0, args.gpu
)
#optimizer
solver = optim.Adam(model.parameters(), lr=args.lr)

if args.gpu:
    model.cuda()


def run_model():
    """
    Training method
    :return:
    """
    best_val = 0.0
    recall1s = []
    for epoch in range(args.n_epoch):
        print('\n\n-------------------------------------------')
        print('Epoch-{}'.format(epoch))
        print('-------------------------------------------')

        model.train()

        train_iter = enumerate(udc.get_iter('train'))

        if not args.no_tqdm:
            train_iter = tqdm(train_iter)
            train_iter.set_description_str('Training')
            train_iter.total = udc.n_train // udc.batch_size

        for it, mb in train_iter:
            context, response, y, cm, rm, key_r, key_mask_r = mb
            output = model(context, response, cm, rm, key_r, key_mask_r)  # Appropriate this line while running different models
            loss = F.binary_cross_entropy_with_logits(output, y)

            loss.backward()
            solver.step()
            solver.zero_grad()

        # Validation
        recall_at_ks = eval_model(
            model, udc, 'valid', gpu=args.gpu, no_tqdm=args.no_tqdm
        )

        print('Loss: {:.3f}; recall@1: {:.3f}; recall@2: {:.3f}; recall@5: {:.3f}'
              .format(loss.data[0], recall_at_ks[0], recall_at_ks[1], recall_at_ks[4]))
        recall_1 = recall_at_ks[0]
        # if epoch > 10:
        #     eval_test()

        if best_val == 0.0:
            save_model(model, model_name)
            best_val = recall_1
            recall1s.append(recall_1)
        else:
            if recall_1 > best_val:
                best_val = recall_1
                print ("Saving model for recall@1:" + str(recall_1))
                save_model(model, model_name)
            else:
                print ("Not saving, best accuracy so far:" + str(best_val))
        #Early stopping
        if recall_1 < np.max(recall1s[-args.early_stop:]):
            break


def eval_test(model):
    '''
    Evaluation
    :param model:
    :return:
    '''
    print('\n\nEvaluating on test set...')
    print('-------------------------------')
    print('Loading the best model........')
    model = load_model(model, model_name)
    model.eval()
    recall_at_ks = eval_model(
        model, udc, 'test', gpu=args.gpu, no_tqdm=args.no_tqdm
    )

    print('Recall@1: {:.3f}; recall@2: {:.3f}; recall@5: {:.3f}'
          .format(recall_at_ks[0], recall_at_ks[1], recall_at_ks[4]))


if __name__ == '__main__':
    #run the models
    try:
        run_model()
        eval_test(model)
    except KeyboardInterrupt:
        eval_test(model)
        exit(0)