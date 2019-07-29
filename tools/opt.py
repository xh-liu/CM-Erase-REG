from pprint import pprint
import argparse

def parse_opt():

    parser = argparse.ArgumentParser()
    # Data input settings
    parser.add_argument('--dataset', type=str, default='refcoco', help='name of dataset')
    parser.add_argument('--splitBy', type=str, default='unc', help='who splits this dataset')
    parser.add_argument('--start_from', type=str, default=None, help='continuing training from saved model')
    # FRCN setting
    parser.add_argument('--imdb_name', default='coco_minus_refer', help='image databased trained on.')
    parser.add_argument('--net_name', default='res101', help='net_name: res101 or vgg16')
    parser.add_argument('--iters', default=1250000, type=int, help='iterations we trained for faster R-CNN')
    parser.add_argument('--tag', default='notime', help='on default tf, don\'t change this!')
    # Visual Encoder Setting
    parser.add_argument('--visual_sample_ratio', type=float, default=0.3, help='ratio of same-type objects over different-type objects')
    parser.add_argument('--visual_fuse_mode', type=str, default='concat', help='concat or mul')
    parser.add_argument('--visual_init_norm', type=float, default=20, help='norm of each visual representation')
    parser.add_argument('--visual_use_bn', type=int, default=-1, help='>0: use bn, -1: do not use bn in visual layer')    
    parser.add_argument('--visual_use_cxt', type=int, default=1, help='if we use contxt')
    parser.add_argument('--visual_cxt_type', type=str, default='frcn', help='frcn or res101')
    parser.add_argument('--visual_drop_out', type=float, default=0.2, help='dropout on visual encoder')
    parser.add_argument('--window_scale', type=float, default=2.5, help='visual context type')
    # Visual Feats Setting
    parser.add_argument('--with_st', type=int, default=1, help='if incorporating same-type objects as contexts')
    parser.add_argument('--num_cxt', type=int, default=5, help='how many surrounding objects do we use') # 68
    # Language Encoder Setting
    parser.add_argument('--glove_size', type=int, default=300, help='the size of glove embedding')
    parser.add_argument('--word_embedding_size', type=int, default=512, help='the encoding size of each token')
    parser.add_argument('--word_vec_size', type=int, default=512, help='further non-linear of word embedding')
    parser.add_argument('--word_drop_out', type=float, default=0.5, help='word drop out after embedding')
    parser.add_argument('--bidirectional', type=int, default=1, help='bi-rnn')
    parser.add_argument('--rnn_hidden_size', type=int, default=512, help='hidden size of LSTM')
    parser.add_argument('--rnn_type', type=str, default='lstm', help='rnn, gru or lstm')
    parser.add_argument('--rnn_drop_out', type=float, default=0.2, help='dropout between stacked rnn layers')
    parser.add_argument('--rnn_num_layers', type=int, default=1, help='number of layers in lang_encoder')
    parser.add_argument('--variable_lengths', type=int, default=1, help='use variable length to encode') 
    # Joint Embedding setting
    parser.add_argument('--jemb_drop_out', type=float, default=0.1, help='dropout in the joint embedding')
    parser.add_argument('--jemb_dim', type=int, default=512, help='joint embedding layer dimension')
    # Loss Setting
    parser.add_argument('--att_weight', type=float, default=1.0, help='weight on attribute prediction')
    parser.add_argument('--visual_rank_weight', type=float, default=1.0, help='weight on paired (ref, sent) over unpaired (neg_ref, sent)')
    parser.add_argument('--lang_rank_weight', type=float, default=1.0, help='weight on paired (ref, sent) over unpaired (ref, neg_sent)')
    parser.add_argument('--margin', type=float, default=0.1, help='margin for ranking loss')
    parser.add_argument('--erase_margin', type=float, default=0.1, help='margin for erase ranking loss')
    parser.add_argument('--erase_loss', type=str, default='ranking', help='loss function for erased sents')
    # Optimization: General
    parser.add_argument('--max_epochs', type=int, default=30, help='max number of epochs to run') 
    parser.add_argument('--sample_ratio', type=float, default=0.3, help='ratio of same-type objects over different-type objects')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size in number of images per batch')
    parser.add_argument('--grad_clip', type=float, default=0.1, help='clip gradients at this value')
    parser.add_argument('--seq_per_ref', type=int, default=3, help='number of expressions per object during training')
    parser.add_argument('--learning_rate_decay_start', type=int, default=6, help='at what epoch to start decaying learning rate')
    parser.add_argument('--learning_rate_decay_every', type=int, default=6, help='every how many epochs thereafter to drop LR by half')
    parser.add_argument('--optim_epsilon', type=float, default=1e-8, help='epsilon that goes into denominator for smoothing')
    parser.add_argument('--learning_rate', type=float, default=4e-4, help='learning rate')
    parser.add_argument('--optim_alpha', type=float, default=0.8, help='alpha for adam')
    parser.add_argument('--optim_beta', type=float, default=0.999, help='beta used for adam')
    # Evaluation/Checkpointing
    parser.add_argument('--num_sents', type=int, default=-1, help='how many images to use when periodically evaluating the validation loss? (-1 = all)')
    parser.add_argument('--save_checkpoint_every', type=int, default=1, help='how often to save a model checkpoint? (unit: epoch)')
    parser.add_argument('--checkpoint_path', type=str, default='output', help='directory to save models')   
    parser.add_argument('--language_eval', type=int, default=0, help='Evaluate language as well (1 = yes, 0 = no)?')
    parser.add_argument('--losses_log_every', type=int, default=25, help='How often do we snapshot losses, for inclusion in the progress dump? (0 = disable)')
    parser.add_argument('--load_best_score', type=int, default=0, help='Do we load previous best score when resuming training.')      
    # misc
    parser.add_argument('--id', type=str, default='0', help='an id identifying this run/job.')
    parser.add_argument('--seed', type=int, default=24, help='random number generator seed to use')
    parser.add_argument('--gpuid', type=int, default=0, help='which gpu to use, -1 = use CPU')
    # testing
    parser.add_argument('--split', type=str, default='testA', help='split: testAB or val, etc')
    parser.add_argument('--verbose', type=int, default=1, help='if we want to print the testing progress')
    # Language Erasing
    parser.add_argument('--resume', type=int, default=0, help='resume on previous model or not')
    parser.add_argument('--erase_train', type=int, default=0, help='whether to erase or not in training')
    parser.add_argument('--erase_test', type=int, default=0, help='whether to erase or not in testing')
    parser.add_argument('--erase_lang_weight', type=float, default=0, help='weight on erased paired (ref, sent) over erased unpaired (neg_ref, sent)')
    parser.add_argument('--erase_allvisual_weight', type=float, default=0, help='erase all visual part')
    parser.add_argument('--visual_erase_rule', type=str, default='stochastic')
    parser.add_argument('--erase_size_visual', type=int, default=3, help='visual erase size')

    # loss functions
    parser.add_argument('--loss_type', type=str, default='MaxMargin', help='loss function. original is MaxMargin, other options like BCE, ...')
    parser.add_argument('--loss_pos_weight', type=float, default=1.0, help='positive sample loss weight, used for non-triplet loss')
    parser.add_argument('--loss_neg_weight', type=float, default=1.0, help='negative sample loss weight, used for non-triplet loss')
    parser.add_argument('--att_loss', type=float, default=1, help='loss weight for att')
    # select features
    parser.add_argument('--shuffle', type=int, default=1, help='shuffle in training')
    parser.add_argument('--eval_mode', type=int, default=0, help='evaluation mode fix bn and dropout')
    
    # parse 
    args = parser.parse_args()
    opt = vars(args)
    pprint('parsed input parameters:')
    pprint(opt)
    return args

if __name__ == '__main__':

    opt = parse_opt()
    print('opt[\'id\'] is ', opt['id'])
