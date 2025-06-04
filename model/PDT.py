import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
# from model.word_embedding import load_word_embeddings
from model.common import MLP, Delta_MLP
from model.losses import ReconLoss

from copy import deepcopy


class PDT(nn.Module):
    def __init__(self, dset, args):
        super(PDT, self).__init__()
        self.dset = dset
        self.args = args
        self.feat_dim = dset.feat_dim
        self.epoch = None
        self.alpha = args.alpha
        self.loss_info = None
        def get_all_ids(relevant_pairs):
            attrs, objs = zip(*relevant_pairs)
            attrs = [dset.attr2idx[attr] for attr in attrs]
            objs = [dset.obj2idx[obj] for obj in objs]
            pairs = [a for a in range(len(relevant_pairs))]
            attrs = torch.LongTensor(attrs).to(args.device)
            objs = torch.LongTensor(objs).to(args.device)
            pairs = torch.LongTensor(pairs).to(args.device)
            return attrs, objs, pairs
        self.temp_negatives = None
        self.temp_anchor = None

        # Validation - Use all pairs to validate
        self.val_attrs, self.val_objs, self.val_pairs = get_all_ids(self.dset.pairs)

        # All attrs and objs without repetition
        self.uniq_attrs, self.uniq_objs = torch.arange(len(self.dset.attrs)).long().to(args.device), \
                                          torch.arange(len(self.dset.objs)).long().to(args.device)
        if args.train_only:
            self.train_attrs, self.train_objs, self.train_pairs = get_all_ids(self.dset.train_pairs)
        else:
            self.train_attrs, self.train_objs, self.train_pairs = self.val_attrs, self.val_objs, self.val_pairs
        
        pairs = [(dset.attr2idx[attr], dset.obj2idx[obj]) for attr, obj in dset.pairs]
        self.pairs = torch.LongTensor(pairs)
        
        attr_word_emb_file = '{}_{}_attr.save'.format(args.dataset, args.emb_type)
        attr_word_emb_file = os.path.join(args.main_root, 'word embedding', attr_word_emb_file)
        obj_word_emb_file = '{}_{}_obj.save'.format(args.dataset, args.emb_type)
        obj_word_emb_file = os.path.join(args.main_root, 'word embedding', obj_word_emb_file)

        print('  Load attribute word embeddings--')
        # pretrained_weight_attr = load_word_embeddings(dset.attrs, args)
        # emb_dim = pretrained_weight_attr.shape[1]
        emb_dim = 300
        self.attr_embedder = nn.Embedding(len(dset.attrs), emb_dim).to(args.device)
        # self.attr_embedder.weight.data.copy_(pretrained_weight_attr)

        print('  Load object word embeddings--')
        # pretrained_weight_obj = load_word_embeddings(dset.objs, args)
        self.obj_embedder = nn.Embedding(len(dset.objs), emb_dim).to(args.device)
        # self.obj_embedder.weight.data.copy_(pretrained_weight_obj)

        if not args.update_word_features:
            for param in self.attr_embedder.parameters():
                param.requires_grad = False
            for param in self.obj_embedder.parameters():
                param.requires_grad = False
        
        '''======================= Visual Encoders ======================='''
        self.image_embedder_comp = MLP(dset.feat_dim+emb_dim, emb_dim, num_layers=2, relu=False, bias=True,
                                       dropout=True, norm=True, layers=args.mlp_layers)
        self.image_embedder_attr = MLP(dset.feat_dim, emb_dim, num_layers=2, relu=False, bias=True, dropout=True, norm=True, layers=args.mlp_layers)
        self.image_embedder_obj = MLP(dset.feat_dim, emb_dim, num_layers=2, relu=False, bias=True,
                                       dropout=True, norm=True, layers=args.mlp_layers)
        self.projection = MLP(emb_dim*2, emb_dim, num_layers=2, relu=False, bias=True,
                                       dropout=False, norm=True, layers=[], p=0)
        '''====================  Cosine Classifier ======================='''
        self.classifier = CosineClassifier(temp=args.cosine_scale)
        self.d_classifier = UCosineClassifier()
        self.guid_classifier = CosineClassifier(temp=args.guid_cosine_scale)

        self.beta = self.args.beta
        '''====================  Delta Encoder ==========================='''
        if self.args.tau>0:
            latent_dim = args.latent_dim1
            self.delta_encoder = Delta_MLP(dset.feat_dim*2, latent_dim, num_layers=args.delta_encoder_layers, relu=False, bias=True, dropout=True, norm=True, layers=args.delta_encoder, p = 0.5)
            self.delta_decoder = Delta_MLP(latent_dim+dset.feat_dim, dset.feat_dim, num_layers=args.delta_decoder_layers, relu=False, bias=True, dropout=False, norm=False, layers=args.delta_decoder)
            self.image_embedder_obj_guid =  MLP(dset.feat_dim, len(dset.objs), num_layers=3, relu=False, bias=True,
                                            dropout=False, norm=False, layers=args.mlp_layers)
            self.keep_image_embedder_obj_guid = copy.deepcopy(self.image_embedder_obj_guid)
            self.recon_loss = ReconLoss()

    def compose(self, attrs, objs, dist = False):
        attrs, objs = self.attr_embedder(attrs), self.obj_embedder(objs)
        inputs = torch.cat([attrs, objs], -1)
        if dist:
            output = self.projection(inputs.detach()) 
        else:
            output = self.projection(inputs) 
        return output
    
    def update_re_image_embedder_obj(self):
        self.keep_image_embedder_obj_guid = copy.deepcopy(self.image_embedder_obj_guid)
        
    def train_forward(self, input_batch, dist = False):
        x, a, o, c = input_batch[0], input_batch[1], input_batch[2], input_batch[3]
        x_o_c, a_neg, c_o_c, x_a_c, o_neg, c_a_c = input_batch[4], input_batch[5], input_batch[6], input_batch[7], input_batch[8], input_batch[9]
        del input_batch

        x_cls = x[:, 0]
        x_a_c_cls = x_a_c[:, 0]
        if self.args.only_cls:
            x = x[:, 0]
            x_a_c = x_a_c[:, 0]
            x_o_c = x_o_c[:, 0]
        else:
            x = x[:, 1]
            x_a_c = x_a_c[:, 1]
            x_o_c = x_o_c[:, 1]

        s_a = self.attr_embedder(self.uniq_attrs)
        s_o = self.obj_embedder(self.uniq_objs)
        
        s_c = self.compose(self.train_attrs, self.train_objs)

        v_o = self.image_embedder_obj(x_cls)
        v_a = self.image_embedder_attr(x)
        # v_ao = self.image_embedder_comp(x)
        v_ao = self.image_embedder_comp(torch.cat([x, v_o], dim = -1))
        
        o_pred = self.classifier(v_o, s_o)
        a_pred = self.classifier(v_a, s_a)
        c_pred = self.classifier(v_ao, s_c)

        loss_o_cls = F.cross_entropy(o_pred, o)
        loss_a_cls = F.cross_entropy(a_pred, a)
        loss_c_cls = F.cross_entropy(c_pred, c)

        loss_cls = loss_c_cls + self.args.w_a*loss_a_cls + self.args.w_o*loss_o_cls

        if self.args.tau>0:
            self.update_re_image_embedder_obj()
            delta_a2an = self.delta_encoder(torch.cat([x, x_o_c], dim = -1))
            t_x_o_c = self.delta_decoder(torch.cat([delta_a2an, x], dim = -1))
            loss_recon = self.recon_loss(t_x_o_c, x_o_c.detach())
            v_o_neg = self.image_embedder_obj(x_a_c_cls)

            # generation
            x_anon = self.delta_decoder(torch.cat([delta_a2an, x_a_c], dim = -1))
            
            v_og = self.image_embedder_obj_guid(x_a_c)
            loss_guid_o = F.cross_entropy(v_og, o_neg)
            
            # object guid
            v_on_re = self.keep_image_embedder_obj_guid(x_anon)
            loss_o_cls_re = F.cross_entropy(v_on_re, o_neg)

            v_anon = self.image_embedder_comp(torch.cat([x_anon.detach(), v_o_neg.detach()], dim = -1))  ###333
            s_c2 = self.compose(self.train_attrs, self.train_objs, dist=True)
            s_c_2 = torch.cat([s_c2[c_a_c], s_c2[c_o_c]], dim = 1).view(c.shape[0], 2, -1)
            pred_ori = self.d_classifier(v_ao.unsqueeze(1), s_c_2, scale= False).squeeze().transpose(0, 1)
            pred = self.d_classifier(v_anon.unsqueeze(1), s_c_2, scale = False).squeeze().transpose(0, 1).flip(-1)
            loss_u = F.smooth_l1_loss(pred, pred_ori.detach(), beta = 1.0)
        else:
            loss_u, loss_recon, loss_o_cls_re, loss_guid_o = 0.0, 0.0, 0.0, 0.0
            
        totloss = loss_cls + self.args.w_r2*loss_recon
        
        if self.args.w_r1 > 0:
            totloss += self.args.w_o*loss_guid_o
        else:
            loss_guid_o = 0.0
            loss_o_cls_re = 0.0
        if self.args.after < self.epoch:
            totloss += self.args.w_r1*loss_o_cls_re
            totloss += self.args.tau*loss_u

        self.loss_info['loss_c_cls'] += float(loss_c_cls)
        self.loss_info['loss_a_cls'] += float(loss_a_cls) 
        self.loss_info['loss_o_cls'] += float(loss_o_cls)
        self.loss_info['loss_o_cls_re'] += float(loss_o_cls_re)
        self.loss_info['loss_recon'] += float(loss_recon)
        self.loss_info['loss_guid_o'] += float(loss_guid_o)
        self.loss_info['loss_u'] += float(loss_u)

        return totloss.mean(), None

    def val_forward(self, input_batch):
        x = input_batch[0]
        del input_batch
        x_cls = x[:, 0]
        if self.args.only_cls:
            x = x[:, 0]
        else:
            x = x[:, 1]
        v_o = self.image_embedder_obj(x_cls)
        v_a = self.image_embedder_attr(x)
        v_c = self.image_embedder_comp(torch.cat([x, v_o], dim = -1))
        # v_c = self.image_embedder_comp(x)
        s_c = self.compose(self.val_attrs, self.val_objs)
        s_o =self.obj_embedder(self.uniq_objs)
        s_a =self.attr_embedder(self.uniq_attrs)
        
        obj_pred = self.classifier(v_o, s_o, scale=False)
        obj_pred = (1+obj_pred)/2
        attr_pred = self.classifier(v_a, s_a, scale=False)
        attr_pred = (1+attr_pred)/2
        c_pred = self.classifier(v_c, s_c, scale=False)
        c_pred = (1+c_pred)/2
        
        scores = {}
        for _, pair in enumerate(self.dset.pairs):
            attr, obj = pair
            scores[pair] = (1-self.alpha)*c_pred[:, self.dset.all_pair2idx[pair]]\
                + self.alpha*attr_pred[:, self.dset.attr2idx[attr]] * obj_pred[:, self.dset.obj2idx[obj]]
        return None, scores
    
    def forward(self, x, dist = False):
        if self.training:
            loss, pred = self.train_forward(x, dist = dist)
            return loss, pred
        else:
            with torch.no_grad():
                loss, pred = self.val_forward(x)
            return loss, pred
        
    def demo_forward(self, x):
        x_cls = x[:, 0]
        if self.args.only_cls:
            x = x[:, 0]
        else:
            x = x[:, 1]
        v_o = self.image_embedder_obj(x_cls)
        v_a = self.image_embedder_attr(x)
        v_c = self.image_embedder_comp(torch.cat([x, v_o], dim = -1))
        # v_c = self.image_embedder_comp(x)
        s_c = self.compose(self.val_attrs, self.val_objs)
        s_o =self.obj_embedder(self.uniq_objs)
        s_a =self.attr_embedder(self.uniq_attrs)
        
        obj_pred = self.classifier(v_o, s_o, scale=False)
        obj_pred = (1+obj_pred)/2
        attr_pred = self.classifier(v_a, s_a, scale=False)
        attr_pred = (1+attr_pred)/2
        c_pred = self.classifier(v_c, s_c, scale=False)
        c_pred = (1+c_pred)/2
        
        scores = {}
        for _, pair in enumerate(self.dset.pairs):
            attr, obj = pair
            scores[pair] = (1-self.alpha)*c_pred[:, self.dset.all_pair2idx[pair]]\
                + self.alpha*attr_pred[:, self.dset.attr2idx[attr]] * obj_pred[:, self.dset.obj2idx[obj]]
        return None, scores

class CosineClassifier(nn.Module):
    def __init__(self, temp=0.05):
        super(CosineClassifier, self).__init__()
        self.temp = temp

    def forward(self, img, concept, scale=True):
        img_norm = F.normalize(img, dim=-1)
        concept_norm = F.normalize(concept, dim=-1)
        pred = torch.matmul(img_norm, concept_norm.transpose(0, 1))
        if scale:
            pred = pred / self.temp
        return pred

class UCosineClassifier(nn.Module):
    def __init__(self, temp=0.05):
        super(UCosineClassifier, self).__init__()
        self.temp = temp

    def forward(self, img, concept, scale=True):
        img_norm = F.normalize(img, dim=-1)
        concept_norm = F.normalize(concept, dim=-2)
        pred = torch.matmul(img_norm, concept_norm.unsqueeze(-1).transpose(0, 1))
        if scale:
            pred = pred / self.temp
        return pred

