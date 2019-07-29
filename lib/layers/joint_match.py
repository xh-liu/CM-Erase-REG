from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


from layers.lang_encoder import RNNEncoderImgGlove, PhraseAttention
from layers.visual_encoder import SubjectEncoder, RelationEncoderAtt, LocationEncoderAtt

import numpy as np
import os
import pdb

"""
Simple Matching function for
- visual_input (n, vis_dim)  
- lang_input (n, vis_dim)
forward them through several mlp layers and finally inner-product, get cossim
"""
class Matching(nn.Module):

  def __init__(self, vis_dim, lang_dim, jemb_dim, jemb_drop_out):
    super(Matching, self).__init__()
    self.vis_emb_fc  = nn.Sequential(nn.Linear(vis_dim, jemb_dim),
                                     nn.BatchNorm1d(jemb_dim),
                                     nn.ReLU(),
                                     nn.Dropout(jemb_drop_out),
                                     nn.Linear(jemb_dim, jemb_dim),
                                     nn.BatchNorm1d(jemb_dim),
                                     )
    self.lang_emb_fc = nn.Sequential(nn.Linear(lang_dim, jemb_dim),
                                     nn.BatchNorm1d(jemb_dim),
                                     nn.ReLU(),
                                     nn.Dropout(jemb_drop_out),
                                     nn.Linear(jemb_dim, jemb_dim),
                                     nn.BatchNorm1d(jemb_dim)
                                     ) 

  def forward(self, visual_input, lang_input):
    """
    Inputs:
    - visual_input float32 (n, vis_dim)
    - lang_input   float32 (n, lang_dim)
    Output:
    - cossim       float32 (n, 1), which is inner-product of two views
    """
    # forward two views
    visual_emb = self.vis_emb_fc(visual_input)
    lang_emb = self.lang_emb_fc(lang_input)

    # l2-normalize
    visual_emb_normalized = nn.functional.normalize(visual_emb, p=2, dim=1) # (n, jemb_dim)
    lang_emb_normalized = nn.functional.normalize(lang_emb, p=2, dim=1)     # (n, jemb_dim)

    # compute cossim
    cossim = torch.sum(visual_emb_normalized * lang_emb_normalized, 1)  # (n, )
    return cossim.view(-1, 1)

class JointMatching(nn.Module):

  def __init__(self, opt):
    super(JointMatching, self).__init__()
    num_layers = opt['rnn_num_layers']
    hidden_size = opt['rnn_hidden_size']
    num_dirs = 2 if opt['bidirectional'] > 0 else 1
    jemb_dim = opt['jemb_dim']
    self.unk_token = opt['unk_token']

    # language rnn encoder
    word_emb_path = os.path.join(os.getcwd(),'glove_emb',opt['dataset']+'.npy')
    dict_emb = np.load(word_emb_path)
    
    self.rnn_encoder = RNNEncoderImgGlove(dict_emb, vocab_size=opt['vocab_size'],
                                  word_embedding_size=opt['word_embedding_size'],
                                  word_vec_size=opt['word_vec_size'],
                                  hidden_size=opt['rnn_hidden_size'],
                                  bidirectional=opt['bidirectional']>0,
                                  input_dropout_p=opt['word_drop_out'],
                                  dropout_p=opt['rnn_drop_out'],
                                  n_layers=opt['rnn_num_layers'],
                                  rnn_type=opt['rnn_type'],
                                  variable_lengths=opt['variable_lengths']>0)

    # [vis; loc] weighter
    self.weight_fc = nn.Linear(num_layers * num_dirs * hidden_size, 3)

    # phrase attender
    self.sub_attn = PhraseAttention(hidden_size * num_dirs)
    self.loc_attn = PhraseAttention(hidden_size * num_dirs)
    self.rel_attn = PhraseAttention(hidden_size * num_dirs)

    # visual matching 
    self.sub_encoder = SubjectEncoder(opt)
    self.sub_matching = Matching(opt['fc7_dim']+opt['jemb_dim'], opt['word_vec_size'], 
                                 opt['jemb_dim'], opt['jemb_drop_out'])

    # location matching
    self.loc_encoder = LocationEncoderAtt(opt)
    self.loc_matching = Matching(opt['jemb_dim'], opt['word_vec_size'], 
                                 opt['jemb_dim'], opt['jemb_drop_out'])


    # relation matching
    self.rel_encoder  = RelationEncoderAtt(opt)
    self.rel_matching = Matching(opt['jemb_dim'], opt['word_vec_size'],
                                 opt['jemb_dim'], opt['jemb_drop_out']) 

    self.fc7_dim = opt['fc7_dim']
    self.process_pool5 = nn.Sequential(nn.Conv2d(opt['pool5_dim'], 256, 1), nn.BatchNorm2d(256), nn.ReLU(),
		    nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
		    nn.Conv2d(256, opt['pool5_dim'], 1), nn.BatchNorm2d(opt['pool5_dim']))
    self.proj_img = nn.Linear(opt['pool5_dim'], opt['jemb_dim'])


  def forward(self, pool5, fc7, lfeats, dif_lfeats, cxt_fc7, cxt_lfeats, labels, img_pool5):

    img_res = self.process_pool5(img_pool5)
    img_res = img_res + img_pool5
    img_res = F.avg_pool2d(img_res, 7).squeeze(3).squeeze(2)
    img_res = self.proj_img(img_res) # (n, 512)
    context, hidden, embedded = self.rnn_encoder(labels, img_res)
    weights = F.softmax(self.weight_fc(hidden)) # (n, 3)

    # subject matching
    sub_attn, sub_phrase_emb = self.sub_attn(context, embedded, labels)
    sub_feats, sub_grid_attn, att_scores = self.sub_encoder(pool5, fc7, sub_phrase_emb) # (n, fc7_dim+att_dim), (n, 49), (n, num_atts)
    sub_matching_scores = self.sub_matching(sub_feats, sub_phrase_emb) # (n, 1)

    # location matching
    loc_attn, loc_phrase_emb = self.loc_attn(context, embedded, labels)
    loc_feats = self.loc_encoder(lfeats, dif_lfeats, loc_phrase_emb)

    loc_matching_scores = self.loc_matching(loc_feats, loc_phrase_emb)    # (n, 1)

    # rel matching
    rel_attn, rel_phrase_emb = self.rel_attn(context, embedded, labels)
    rel_feats, rel_ixs = self.rel_encoder(cxt_fc7, cxt_lfeats, rel_phrase_emb)
    rel_matching_scores = self.rel_matching(rel_feats, rel_phrase_emb)

    # final scores
    scores = (weights * torch.cat([sub_matching_scores, 
                                     loc_matching_scores, 
                                     rel_matching_scores], 1)).sum(1) # (n, )
    return scores, sub_grid_attn, sub_attn, loc_attn, rel_attn, rel_ixs, weights, att_scores, sub_matching_scores.view(-1)
    

  def get_token_weights(self, labels, img_pool5=None):
    """
    Inputs:
    - labels      : (n, seq_len)
    Output:
    - sub_attn      : (n, seq_len) attn on subjective words of expression 
    - loc_attn      : (n, seq_len) attn on location words of expression
    - rel_attn      : (n, seq_len) attn on relation words of expression
    - weights       : (n, 3) attn on modules
    """
    # expression encoding
    img_res = self.process_pool5(img_pool5)
    img_res = img_res + img_pool5
    img_res = F.avg_pool2d(img_res, 7).squeeze(3).squeeze(2)
    img_res = self.proj_img(img_res) # (n, 512)
    context, hidden, embedded = self.rnn_encoder(labels, img_res)

    weights = F.softmax(self.weight_fc(hidden)) # (n, 3)

    # subject matching
    sub_attn, sub_phrase_emb = self.sub_attn(context, embedded, labels)

    # location matching
    loc_attn, loc_phrase_emb = self.loc_attn(context, embedded, labels)

    # rel matching
    rel_attn, rel_phrase_emb = self.rel_attn(context, embedded, labels)

    # final scores
    weights = weights.unsqueeze_(1)
    attention = (weights.expand(-1, sub_attn.size(1), -1) * torch.cat([sub_attn.unsqueeze_(2), 
                                   loc_attn.unsqueeze_(2), 
                                   rel_attn.unsqueeze_(2)], 2)).sum(2) # (n, )

    return attention, sub_phrase_emb, loc_phrase_emb, rel_phrase_emb, weights

  def manipulate_labels(self, labels, img_pool5):
    if labels.size(1) <= 2:
      return labels
    all_idx = np.asarray(range(labels.size(0)))
    attention, _, _, _, _ = self.get_token_weights(labels, img_pool5)

    select_index = torch.multinomial(attention, 1)
    select_index = np.asarray(select_index.cpu().data)
    if len((labels[:,2]==0).nonzero().size()) != 0:
      one_word_idx = np.asarray((labels[:,2]==0).nonzero().cpu().data.squeeze(1))
      all_idx = np.delete(all_idx, one_word_idx)
      select_index = np.delete(select_index, one_word_idx)
    labels[all_idx, select_index] = self.unk_token
    return labels

  def erase_subj(self, labels, pool5, fc7, sub_phrase_emb, erase_size, erase_rule):

    batch_size = labels.size(0)

    # get attention weights
    visual_attn = self.sub_encoder.extract_visual_attn(pool5, fc7, sub_phrase_emb) # (n, 49)
    visual_attn = visual_attn.view(batch_size, 7, 7) # (n, 7, 7)

    # select region to erase
    pool5 = pool5.data.cpu().numpy()
    fc7 = fc7.data.cpu().numpy()
    all_index = np.asarray(range(batch_size))
    if erase_rule == 'largest_pixels':
      _, sorted_idx = torch.sort(visual_attn.view(batch_size,-1), dim=1, descending=True)
      sorted_idx = np.asarray(sorted_idx.cpu().data)
      select_index = sorted_idx[:,:erase_size]
      select_index_row = select_index // 7
      select_index_col = select_index % 7
      for cnt in range(erase_size):
        pool5[all_index, :, select_index_row[:,cnt],select_index_col[:,cnt]] = 0
        fc7[all_index, :, select_index_row[:,cnt], select_index_col[:,cnt]] = 0
      pool5 = Variable(torch.from_numpy(pool5), requires_grad=False).cuda()
      fc7 = Variable(torch.from_numpy(fc7), requires_grad=False).cuda()
      return pool5, fc7
    elif erase_rule == 'stochastic':
      region_scores = F.avg_pool2d(visual_attn, erase_size, stride=1) # (n, 7-self.erase_size_visual+1, 7-self.erase_size_visual+1)
      select_index = torch.multinomial(region_scores.view(batch_size, -1), 1) # (n, 1), ranging from 0 to 24  
    elif erase_rule == 'largest':
      region_scores = F.avg_pool2d(visual_attn, erase_size, stride=1) # (n, 7-self.erase_size_visual+1, 7-self.erase_size_visual+1)
      _, select_index = torch.max(region_scores.view(batch_size, -1), 1)

    # erase corresponding region on pool5 and fc7 features
    select_index = np.asarray(select_index.view(-1).cpu().data)
    select_index_row = select_index // (7-erase_size+1)
    select_index_col = select_index % (7-erase_size+1)
    for erase_row in range(erase_size):
      for erase_col in range(erase_size):
        pool5[all_index, :, select_index_row+erase_row, select_index_col+erase_col] = 0
        fc7[all_index, :, select_index_row+erase_row, select_index_col+erase_col] = 0

    pool5 = Variable(torch.from_numpy(pool5), requires_grad=False).cuda()
    fc7 = Variable(torch.from_numpy(fc7), requires_grad=False).cuda()

    return pool5, fc7

  def erase_loc(self, labels, lfeats, dif_lfeats, phrase_emb):
    batch_size = lfeats.size(0)
    loc_att = self.loc_encoder.extract_loc_attn(lfeats, dif_lfeats, phrase_emb)
    select_idx = torch.multinomial(loc_att, 1)
    select_idx = np.asarray(select_idx.view(-1).cpu().data)
    all_idx = np.asarray(range(batch_size))
    all_feats = torch.cat([lfeats, dif_lfeats], 1).view(batch_size, 6, 5) #.contiguous() # (n, 6, 5)
    all_feats[all_idx, select_idx, :] = 0
    lfeats = all_feats[:, 0, :].contiguous().view(batch_size, 5)
    dif_lfeats = all_feats[:, 1:, :].contiguous().view(batch_size, 25)
    return lfeats, dif_lfeats

  def erase_rel(self, labels, cxt_feats, cxt_lfeats, phrase_emb):
    batch_size = cxt_feats.size(0)
    rel_att = self.rel_encoder.extract_rel_attn(cxt_feats, cxt_lfeats, phrase_emb)
    select_idx = torch.multinomial(rel_att, 1)
    select_idx = np.asarray(select_idx.view(-1).cpu().data)
    all_idx = np.asarray(range(batch_size))
    cxt_feats[all_idx, select_idx, :] = 0
    cxt_lfeats[all_idx, select_idx, :] = 0
    return cxt_feats, cxt_lfeats

  def erase_allvisual(self, labels, feats, erase_size, erase_rule):
    batch_size = labels.size(0)
    if labels.size(1) == 1:
      return labels
    
    attention, sub_phrase_emb, loc_phrase_emb, rel_phrase_emb, weights = self.get_token_weights(labels, feats['img_pool5'])
    # decide which module to erase
    module = torch.multinomial(weights.view(batch_size,-1), 1).view(-1)
    erase_subj_idx = (module==0).nonzero().view(-1).cpu().data.numpy() # to be checked
    erase_loc_idx = (module==1).nonzero().view(-1).cpu().data.numpy()
    erase_rel_idx = (module==2).nonzero().view(-1).cpu().data.numpy()
    # devide into three sub-batches based on which module to erase
    if len(erase_subj_idx) > 0:
      feats['pool5'][erase_subj_idx], feats['fc7'][erase_subj_idx] = self.erase_subj(labels[erase_subj_idx], 
          feats['pool5'][erase_subj_idx], feats['fc7'][erase_subj_idx], sub_phrase_emb[erase_subj_idx], erase_size, erase_rule)
    if len(erase_loc_idx) > 0:
      feats['lfeats'][erase_loc_idx], feats['dif_lfeats'][erase_loc_idx] = self.erase_loc(labels[erase_loc_idx],
          feats['lfeats'][erase_loc_idx], feats['dif_lfeats'][erase_loc_idx], loc_phrase_emb[erase_loc_idx])
    if len(erase_rel_idx) > 0:
      feats['cxt_fc7'][erase_rel_idx], feats['cxt_lfeats'][erase_rel_idx] = self.erase_rel(labels[erase_rel_idx],
          feats['cxt_fc7'][erase_rel_idx], feats['cxt_lfeats'][erase_rel_idx], rel_phrase_emb[erase_rel_idx])

    return feats

