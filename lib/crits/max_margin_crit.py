from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import pdb

class MaxMarginCriterion(nn.Module):

    def __init__(self, visual_rank_weight, lang_rank_weight, margin):
        super(MaxMarginCriterion, self).__init__()
        self.visual_rank = visual_rank_weight > 0 
        self.lang_rank = lang_rank_weight > 0
        self.visual_rank_weight = visual_rank_weight
        self.lang_rank_weight = lang_rank_weight
        self.margin = margin

    def forward(self, cossim):
        N = cossim.size(0)
        batch_size = 0
        if self.visual_rank and not self.lang_rank:
            batch_size = N//2
            assert isinstance(batch_size, int)
            paired = cossim[:batch_size]
            unpaired = cossim[batch_size:]
            visual_rank_loss = self.visual_rank_weight * torch.clamp(self.margin + unpaired - paired, min=0)
            lang_rank_loss = 0.
            
        elif not self.visual_rank and self.lang_rank:
            batch_size = N//2
            assert isinstance(batch_size, int)
            paird = cossim[:batch_size]
            unpaired = cossim[batch_size:]
            lang_rank_loss = self.lang_rank_weight * torch.clamp(self.margin + unpaired - paired, min=0)
            visual_rank_loss = 0.

        elif self.visual_rank and self.lang_rank:
            batch_size = N//3
            assert isinstance(batch_size, int)
            paired = cossim[:batch_size]
            visual_unpaired = cossim[batch_size: batch_size*2]
            lang_unpaired = cossim[batch_size*2:]
            visual_rank_loss = self.visual_rank_weight * torch.clamp(self.margin + visual_unpaired - paired, 0)
            lang_rank_loss = self.lang_rank_weight * torch.clamp(self.margin + lang_unpaired - paired, 0)

        else:
            raise NotImplementedError

        loss = (visual_rank_loss + lang_rank_loss).sum() / batch_size
        return loss

# max margin criterion with erasing.
class MaxMarginEraseCriterion(nn.Module):

    def __init__(self, visual_rank_weight, lang_rank_weight, erase_lang_weight, erase_visual_weight, margin, erase_margin):
        super(MaxMarginEraseCriterion, self).__init__()
        self.visual_rank = visual_rank_weight > 0 
        self.lang_rank = lang_rank_weight > 0
        self.erase_lang = erase_lang_weight > 0
        self.erase_visual = erase_visual_weight > 0
        self.erase = self.erase_lang or self.erase_visual
        self.visual_rank_weight = visual_rank_weight
        self.lang_rank_weight = lang_rank_weight
        self.erase_lang_weight = erase_lang_weight
        self.erase_visual_weight = erase_visual_weight
        self.margin = margin
        self.erase_margin = erase_margin

    def forward(self, cossim):
        N = cossim.size(0)
        batch_size = 0
        if self.visual_rank and self.lang_rank and self.erase_lang and self.erase_visual:
            batch_size = N//7
            assert isinstance(batch_size, int)
            paired = cossim[:batch_size]
            visual_unpaired = cossim[batch_size: batch_size*2]
            lang_unpaired = cossim[batch_size*2: batch_size*3]
            lang_erase_paired = cossim[batch_size*3: batch_size*4]
            lang_erase_unpaired = cossim[batch_size*4:batch_size*5]
            visual_erase_paired = cossim[batch_size*5:batch_size*6]
            visual_erase_unpaired = cossim[batch_size*6:]
            visual_rank_loss = self.visual_rank_weight * torch.clamp(self.margin + visual_unpaired - paired, 0)
            lang_rank_loss = self.lang_rank_weight * torch.clamp(self.margin + lang_unpaired - paired, 0)
            lang_erase_rank_loss = self.erase_lang_weight * torch.clamp(self.erase_margin + lang_erase_unpaired - lang_erase_paired, 0)
            visual_erase_rank_loss = self.erase_visual_weight * torch.clamp(self.erase_margin + visual_erase_unpaired - visual_erase_paired, 0)
            loss = (visual_rank_loss + lang_rank_loss + lang_erase_rank_loss + visual_erase_rank_loss).sum() / batch_size

        elif self.visual_rank and self.lang_rank and self.erase:
            batch_size = N//5
            assert isinstance(batch_size, int)
            paired = cossim[:batch_size]
            visual_unpaired = cossim[batch_size: batch_size*2]
            lang_unpaired = cossim[batch_size*2: batch_size*3]
            paired_erase = cossim[batch_size*3: batch_size*4]
            unpaired_erase = cossim[batch_size*4:]
            visual_rank_loss = self.visual_rank_weight * torch.clamp(self.margin + visual_unpaired - paired, 0)
            lang_rank_loss = self.lang_rank_weight * torch.clamp(self.margin + lang_unpaired - paired, 0)
            if self.erase_visual:
                erase_rank_loss = self.erase_visual_weight * torch.clamp(self.erase_margin + unpaired_erase - paired_erase, 0)
            elif self.erase_lang:
                erase_rank_loss = self.erase_lang_weight * torch.clamp(self.erase_margin + unpaired_erase - paired_erase, 0)
            loss = (visual_rank_loss + lang_rank_loss + erase_rank_loss).sum() / batch_size

        else:
            raise NotImplementedError

        return loss

