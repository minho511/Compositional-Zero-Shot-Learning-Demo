import torch
import torch.nn as nn
import torch.nn.functional as F
from model.common import MLP

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
        
        print('  Load attribute word embeddings--')
        emb_dim = 300
        self.attr_embedder = nn.Embedding(len(dset.attrs), emb_dim).to(args.device)

        print('  Load object word embeddings--')
        self.obj_embedder = nn.Embedding(len(dset.objs), emb_dim).to(args.device)

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

    def compose(self, attrs, objs, dist = False):
        attrs, objs = self.attr_embedder(attrs), self.obj_embedder(objs)
        inputs = torch.cat([attrs, objs], -1)
        if dist:
            output = self.projection(inputs.detach()) 
        else:
            output = self.projection(inputs) 
        return output
    
    def demo_forward(self, x):
        x_cls = x[:, 0]
        x = x[:, 1]
        v_o = self.image_embedder_obj(x_cls)
        v_a = self.image_embedder_attr(x)
        v_c = self.image_embedder_comp(torch.cat([x, v_o], dim = -1))
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

