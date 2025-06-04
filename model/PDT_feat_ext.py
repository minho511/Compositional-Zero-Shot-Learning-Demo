import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from model.word_embedding import load_word_embeddings
from model.common import MLP, Delta_MLP
from model.losses import ReconLoss

from copy import deepcopy


class PDT_EXT(nn.Module):
    def __init__(self, dset, args):
        super(PDT_EXT, self).__init__()
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
        pretrained_weight_attr = load_word_embeddings(dset.attrs, args)
        emb_dim = pretrained_weight_attr.shape[1]
        self.attr_embedder = nn.Embedding(len(dset.attrs), emb_dim).to(args.device)
        self.attr_embedder.weight.data.copy_(pretrained_weight_attr)

        print('  Load object word embeddings--')
        pretrained_weight_obj = load_word_embeddings(dset.objs, args)
        self.obj_embedder = nn.Embedding(len(dset.objs), emb_dim).to(args.device)
        self.obj_embedder.weight.data.copy_(pretrained_weight_obj)

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
        self.re_image_embedder_obj = copy.deepcopy(self.image_embedder_obj)
        for param in self.re_image_embedder_obj.parameters():
            param.detach_()
        
    def train_forward(self, input_batch, f_type):
        x, a, o, c = input_batch[0], input_batch[1], input_batch[2], input_batch[3]
        x_o_c, a_neg, c_o_c, x_a_c, o_neg, c_a_c = input_batch[4], input_batch[5], input_batch[6], input_batch[7], input_batch[8], input_batch[9]
        del input_batch
        self.update_re_image_embedder_obj()
        beta = self.beta
        x_cls = x[:, 0]
        x = x[:, 1]
        x_a_c_cls = x_a_c[:, 0]
        x_a_c = beta*x_a_c[:, 0]+(1-beta)*x_a_c[:, 1]
        x_o_c = beta*x_o_c[:, 0]+(1-beta)*x_o_c[:, 1]

        # s_a = self.attr_embedder(self.uniq_attrs)
        # s_o = self.obj_embedder(self.uniq_objs)
        # s_c = self.compose(self.train_attrs, self.train_objs)

        v_o = self.image_embedder_obj(x_cls)
        # v_a = self.image_embedder_attr(x)
        v_o_neg = self.image_embedder_obj(x_a_c_cls)

        delta_a2an = self.delta_encoder(torch.cat([x, x_o_c], dim = -1))

        v_c = self.image_embedder_comp(torch.cat([x, v_o], dim = -1))

        x_anon = self.delta_decoder(torch.cat([delta_a2an, x_a_c], dim = -1))

        v_anon = self.image_embedder_comp(torch.cat([x_anon.detach(), v_o_neg.detach()], dim = -1))
        if f_type == 'v':
            return v_c, v_anon    
        elif f_type =='x':
            return x, x_anon
        else:
            print("type \"f_type\" correctly")
            exit()
        
    

    def val_forward(self, input_batch, f_type):
        x = input_batch[0]
        del input_batch
        x_cls = x[:, 0]
        x = x[:, 1]
        v_o = self.image_embedder_obj(x_cls)
        v_c = self.image_embedder_comp(torch.cat([x, v_o], dim = -1))

        if f_type=='v':
            return v_c
        elif f_type=='x':    
            return x
        else:
            print("type \"f_type\" correctly")
            exit()
        
    
    def forward(self, x, is_train = False, f_type='v'):
        if is_train:
            v_c, v_anon = self.train_forward(x, f_type)
        else:
            v_anon = None
            v_c = self.val_forward(x, f_type)
        return v_c, v_anon
    

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
