from __future__ import absolute_import 
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import sys
import numpy as np
import json
import h5py
import time
import random
from pprint import pprint

# model
import _init_paths
from loaders.gt_mrcn_loader import GtMRCNLoader
from layers.joint_match import JointMatching
import models.utils as model_utils
import models.eval_easy_utils as eval_utils
from crits.max_margin_crit import MaxMarginCriterion
from crits.max_margin_crit import MaxMarginEraseCriterion
from opt import parse_opt

# torch
import torch 
import torch.nn as nn
from torch.autograd import Variable

# tensorboard
import tensorboard_logger as tb_logger

# train one iter
def lossFun(loader, optimizer, model, mm_crit, att_crit, opt, iter, erase_model = None):
  # set mode
  if opt['eval_mode']:
    model.eval()
  else:
    model.train()

  # zero gradient
  optimizer.zero_grad()

  # time
  T = {}

  # load one batch of data
  tic = time.time()
  data = loader.getBatch('train', opt)
  Feats = data['Feats']
  labels = data['labels']

  # add [neg_vis, neg_lang]
  if opt['visual_rank_weight'] > 0:
    Feats = loader.combine_feats(Feats, data['neg_Feats'])
    labels = torch.cat([labels, data['labels']])
  if opt['lang_rank_weight'] > 0:
    Feats = loader.combine_feats(Feats, data['Feats'])
    labels = torch.cat([labels, data['neg_labels']])

  att_labels, select_ixs = data['att_labels'], data['select_ixs']

  T['data'] = time.time()-tic

  # change label (erase)
  if opt['erase_train']:
    if opt['erase_lang_weight'] > 0:
      tmp_img_pool5 = data['Feats']['img_pool5']
      tmp_img_pool5.detach_()
      tmp_img_pool5.volatile = True
      labels_new = data['labels'].clone()
      labels_new.detach_()
      labels_new.volatile = True
      labels_new_lengths = (labels_new != 0).sum(1)
      labels_new_lengths_list = labels_new_lengths.data.cpu().numpy().tolist()
      if max(labels_new_lengths_list) == labels_new.size(1):
        labels_new = model.manipulate_labels(labels_new, tmp_img_pool5)
      else:
        labels_new_clip = labels_new[:,:max(labels_new_lengths_list)].clone()
        labels_new_remain = labels_new[:,max(labels_new_lengths_list):].clone()
        labels_new_clip = model.manipulate_labels(labels_new_clip, tmp_img_pool5)
        labels_new = torch.cat((labels_new_clip, labels_new_remain), 1)
      
      labels = torch.cat([labels, labels_new, labels_new], 0)
      Feats = loader.combine_feats(Feats, data['Feats'])
      Feats = loader.combine_feats(Feats, data['neg_Feats'])

    if opt['erase_allvisual_weight'] > 0:
      feats_new = {}
      feats_new = {}
      for key, value in data['Feats'].items():
        feats_new[key] = data['Feats'][key].clone()
        feats_new[key].detach_()
        feats_new[key].volatile = True
      labels_tmp = data['labels']
      labels_tmp.detach_()
      labels_tmp.volatile = True
      labels_tmp_lengths = (labels_tmp != 0).sum(1)
      labels_tmp_lengths_list = labels_tmp_lengths.data.cpu().numpy().tolist()
      if max(labels_tmp_lengths_list) != labels_tmp.size(1):
        labels_tmp = labels_tmp[:,:max(labels_tmp_lengths_list)]
      feats_new = model.erase_allvisual(labels_tmp, feats_new, erase_size=opt['erase_size_visual'], erase_rule=opt['visual_erase_rule'])
      labels = torch.cat([labels, data['labels'], data['neg_labels']])
      Feats = loader.combine_feats(Feats, feats_new)
      Feats = loader.combine_feats(Feats, feats_new)
    
    labels.detach_()
    labels.volatile = False
    for key, _ in Feats.items():
      Feats[key].detach_()
      Feats[key].volatile = False
  # forward
  tic = time.time()

  scores, _, _, _, _, _, _, att_scores, _ = model(Feats['pool5'], Feats['fc7'], Feats['lfeats'], Feats['dif_lfeats'], Feats['cxt_fc7'], Feats['cxt_lfeats'], labels, img_pool5=Feats['img_pool5'])
 
  loss1 = mm_crit(scores)
  if select_ixs.numel() > 0 and opt['att_loss'] > 0:
    loss2 = opt['att_weight'] * att_crit(att_scores.index_select(0, select_ixs),
                                         att_labels.index_select(0, select_ixs))
    loss = loss1 + loss2
  else:
    loss = loss1

  loss.backward()
  model_utils.clip_gradient(optimizer, opt['grad_clip'])
  optimizer.step()
  T['model'] = time.time()-tic
  del scores
  del att_scores

  # return 
  if select_ixs.numel() > 0 and opt['att_loss'] > 0:
    lossdata = loss.data[0]
    loss1data = loss1.data[0]
    loss2data = loss2.data[0]
    del loss
    del loss1
    del loss2
    return lossdata, loss1data, loss2data, T, data['bounds']['wrapped']
  else:
    lossdata = loss.data[0]
    loss1data = loss1.data[0]
    del loss
    del loss1
    return lossdata, loss1data, 0, T, data['bounds']['wrapped']

def main(args):

  opt = vars(args)
  tb_logger.configure('tb_logs/'+opt['id'], flush_secs=2)

  # initialize
  opt['dataset_splitBy'] = opt['dataset'] + '_' + opt['splitBy']
  if opt['dataset'] == 'refcocog':
    opt['unk_token'] = 3346
  elif opt['dataset'] == 'refcoco':
    opt['unk_token'] = 1996 
  elif opt['dataset'] == 'refcoco+':
    opt['unk_token'] = 2629
  checkpoint_dir = osp.join(opt['checkpoint_path'], opt['dataset_splitBy'])
  if not osp.isdir(checkpoint_dir): os.makedirs(checkpoint_dir)

  # set random seed
  torch.manual_seed(opt['seed'])
  random.seed(opt['seed'])

  # set up loader
  data_json = osp.join('cache/prepro', opt['dataset_splitBy'], 'data.json')
  data_h5 = osp.join('cache/prepro', opt['dataset_splitBy'], 'data.h5')
  loader = GtMRCNLoader(data_h5=data_h5, data_json=data_json)
  # prepare feats
  feats_dir = '%s_%s_%s' % (args.net_name, args.imdb_name, args.tag)
  head_feats_dir=osp.join('cache/feats/', opt['dataset_splitBy'], 'mrcn', feats_dir)
  loader.prepare_mrcn(head_feats_dir, args)
  ann_feats = osp.join('cache/feats', opt['dataset_splitBy'], 'mrcn', 
                       '%s_%s_%s_ann_feats.h5' % (opt['net_name'], opt['imdb_name'], opt['tag']))
  loader.loadFeats({'ann': ann_feats})

  # set up model
  opt['vocab_size']= loader.vocab_size
  opt['fc7_dim']   = loader.fc7_dim
  opt['pool5_dim'] = loader.pool5_dim
  opt['num_atts']  = loader.num_atts
  model = JointMatching(opt)

  # resume from previous checkpoint
  infos = {}
  if opt['start_from'] is not None:
    checkpoint = torch.load(os.path.join('output',opt['dataset_splitBy'],opt['start_from']+'.pth'))
    model.load_state_dict(checkpoint['model'].state_dict())
    infos = json.load(open(os.path.join('output',opt['dataset_splitBy'],opt['start_from']+'.json') ,'r'))
    print('start from model %s, best val score %.2f%%\n' % (opt['start_from'], infos['best_val_score']*100))

  if opt['resume']:
    iter = infos.get('iter', 0)
    epoch = infos.get('epoch', 0)
    val_accuracies = infos.get('val_accuracies', [])
    val_loss_history = infos.get('val_loss_history', {})
    val_result_history = infos.get('val_result_history', {})
    loss_history = infos.get('loss_history', {})
    loader.iterators = infos.get('iterators', loader.iterators)
    best_val_score = infos.get('best_val_score', None)
  else:
    iter = 0
    epoch = 0
    val_accuracies = []
    val_loss_history = {}
    val_result_history = {}
    loss_history = {}
    best_val_score = None


  # set up criterion
  if opt['erase_lang_weight'] > 0 or opt['erase_allvisual_weight'] > 0:
    if opt['erase_allvisual_weight'] > 0:
      mm_crit = MaxMarginEraseCriterion(opt['visual_rank_weight'], opt['lang_rank_weight'],
        opt['erase_lang_weight'], opt['erase_allvisual_weight'], opt['margin'], opt['erase_margin'])
    elif opt['erase_lang_weight'] > 0:
      mm_crit = MaxMarginEraseCriterion(opt['visual_rank_weight'], opt['lang_rank_weight'],
        opt['erase_lang_weight'], opt['erase_allvisual_weight'], opt['margin'], opt['erase_margin'])
  else:
    mm_crit = MaxMarginCriterion(opt['visual_rank_weight'], opt['lang_rank_weight'], opt['margin'])

  att_crit = nn.BCEWithLogitsLoss(loader.get_attribute_weights())

  # move to GPU
  if opt['gpuid'] >= 0:
    model.cuda()
    mm_crit.cuda()
    att_crit.cuda()

  # set up optimizer
  optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                               lr=opt['learning_rate'],
                               betas=(opt['optim_alpha'], opt['optim_beta']),
                               eps=opt['optim_epsilon'])

  # start training
  data_time, model_time = 0, 0
  lr = opt['learning_rate']
  best_predictions, best_overall = None, None
  if opt['shuffle']:
    loader.shuffle('train')

  while True:
    # run one iteration
    loss, loss1, loss2, T, wrapped = lossFun(loader, optimizer, model, mm_crit, att_crit, opt, iter)
    data_time += T['data']
    model_time += T['model']

    # write the training loss summary
    if iter % opt['losses_log_every'] == 0:
      loss_history[iter] = loss
      # print stats
      log_toc = time.time()
      print('iter[%s](epoch[%s]), train_loss=%.3f, lr=%.2E, data:%.2fs/iter, model:%.2fs/iter' \
            % (iter, epoch, loss, lr, data_time/opt['losses_log_every'], model_time/opt['losses_log_every']))
      # write tensorboard logger
      tb_logger.log_value('epoch', epoch, step=iter)
      tb_logger.log_value('iter', iter, step=iter)
      tb_logger.log_value('training_loss', loss, step=iter)
      tb_logger.log_value('training_loss1', loss1, step=iter)
      tb_logger.log_value('training_loss2', loss2, step=iter)
      tb_logger.log_value('learning_rate', lr, step=iter)
      data_time, model_time = 0, 0

    # decay the learning rates
    if opt['learning_rate_decay_start'] > 0 and epoch > opt['learning_rate_decay_start']:
      frac = (epoch - opt['learning_rate_decay_start']) / opt['learning_rate_decay_every']
      decay_factor =  0.1 ** frac
      lr = opt['learning_rate'] * decay_factor
      # update optimizer's learning rate
      model_utils.set_lr(optimizer, lr)

              
    # update iter and epoch
    iter += 1
    #wrapped = True # for debugging validation phase
    if wrapped:
      if opt['shuffle']:
        loader.shuffle('train')
      epoch += 1
      # eval loss and save checkpoint
      val_loss, acc, predictions, overall = eval_utils.eval_split(loader, model, None, 'val', opt)
      val_loss_history[iter] = val_loss
      val_result_history[iter] = {'loss': val_loss, 'accuracy': acc}
      val_accuracies += [(iter, acc)]
      print('val loss: %.2f' % val_loss)
      print('val acc : %.2f%%\n' % (acc*100.0))
      print('val precision : %.2f%%' % (overall['precision']*100.0))
      print('val recall    : %.2f%%' % (overall['recall']*100.0))
      print('val f1        : %.2f%%' % (overall['f1']*100.0))
      # write tensorboard logger
      tb_logger.log_value('val_loss', val_loss, step=iter)
      tb_logger.log_value('val_acc', acc, step=iter)
      tb_logger.log_value('val precision', overall['precision']*100.0, step=iter)
      tb_logger.log_value('val recall', overall['recall']*100.0, step=iter)
      tb_logger.log_value('val f1', overall['f1']*100.0, step=iter)

      # save model if best
      current_score = acc
      if best_val_score is None or current_score > best_val_score:
        best_val_score = current_score
        best_predictions = predictions
        best_overall = overall
        checkpoint_path = osp.join(checkpoint_dir, opt['id'] + '.pth')
        checkpoint = {}
        checkpoint['model'] = model
        checkpoint['opt'] = opt
        torch.save(checkpoint, checkpoint_path) 
        print('model saved to %s' % checkpoint_path) 

      # write json report 
      infos['iter'] = iter
      infos['epoch'] = epoch
      infos['iterators'] = loader.iterators
      infos['loss_history'] = loss_history
      infos['val_accuracies'] = val_accuracies
      infos['val_loss_history'] = val_loss_history
      infos['best_val_score'] = best_val_score
      infos['best_predictions'] = predictions if best_predictions is None else best_predictions
      infos['best_overall'] = overall if best_overall is None else best_overall
      infos['opt'] = opt
      infos['val_result_history'] = val_result_history
      infos['word_to_ix'] = loader.word_to_ix
      infos['att_to_ix'] = loader.att_to_ix
      with open(osp.join(checkpoint_dir, opt['id']+'.json'), 'wb') as io:
        json.dump(infos, io)


      if epoch >= opt['max_epochs'] and opt['max_epochs'] > 0:
        break


if __name__ == '__main__':

  args = parse_opt()
  main(args)

