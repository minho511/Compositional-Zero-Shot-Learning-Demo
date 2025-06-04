'''
Image feature extractor
'''
from model.build_vit_backbone import build_vit_sup_models
from model.configs_vpt.config import get_cfg
from model.dino.vision_transformer import vit_base
import torch

def get_image_extractor(args, arch='vit-frozen', pretrained=True, feature_dim=None):
    '''
    Inputs
        arch: Base architecture
        pretrained: Bool, Imagenet weights
        feature_dim: Int, output feature dimension
        checkpoint: String, not implemented
    Returns
        Pytorch model
    '''
    print("image_extractor >> ", arch)
    if arch == 'vit-frozen':
        prompt_cfg = None
        model, _ = build_vit_sup_models(model_type="sup_vitb16", crop_size=224, prompt_cfg=prompt_cfg)
        for k, p in model.named_parameters():
            p.requires_grad = False
        model.eval()

    elif arch == 'vit-prompt':
        cfg = get_cfg()
        prompt_cfg = cfg.MODEL.PROMPT
        cfg.MODEL.PROMPT.DEEP = True
        model, _ = build_vit_sup_models(model_type="sup_vitb16", crop_size=224, prompt_cfg=prompt_cfg)
        for k, p in model.named_parameters():
            if "prompt" not in k:
                p.requires_grad = False
        model.train()

    elif arch == 'vit-finetuning':
        cfg = get_cfg()
        prompt_cfg = None
        model, _ = build_vit_sup_models(model_type="sup_vitb16", crop_size=224, prompt_cfg=prompt_cfg)
        model.train()

    elif arch == 'vit-dino':
        model = vit_base()
        state_dict = torch.load('/data/NeurIPS2024/pretrained/dino_vitbase16_pretrain.pth')
        model.load_state_dict(state_dict)
        for k, p in model.named_parameters():
            p.requires_grad = False
        model.eval()
    return model