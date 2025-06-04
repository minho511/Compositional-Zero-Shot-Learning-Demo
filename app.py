import gradio as gr
import torch
from PIL import Image
import os

from model.PDT import PDT
from flags import parser
from utils.utils import load_args
from data import dataset as dset
from model.dino.vision_transformer import vit_base
import torchvision.transforms as transforms
from utils.utils import get_norm_values, chunks
import torch.nn as nn
from huggingface_hub import hf_hub_download

args = parser.parse_args()
args.main_root = os.path.dirname(__file__)
args.data_root = 'dataset'
args.device = 'cpu'
config_path = 'config/pdt.yml'
mean, std = get_norm_values(norm_family='imagenet')
transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

load_args(config_path, args)
testset = dset.CompositionDataset(
        args=args,
        root=os.path.join(args.data_root, args.dataset),
        phase='test',
        split=args.splitname,
        model =args.image_extractor,
        update_image_features = args.update_image_features,
    )
backbone = vit_base()
ckpt_path = hf_hub_download(repo_id="Minho511/pdt-cgqa-czsl", filename='dino_vitbase16_pretrain.pth')
state_dict = torch.load(ckpt_path, map_location='cpu')
backbone.load_state_dict(state_dict)
for k, p in backbone.named_parameters():
    p.requires_grad = False
# backbone = nn.Sequential(*list(backbone.children())[:-1])
backbone.eval()

model = PDT(testset, args).to(args.device)

ckpt_path = hf_hub_download(repo_id="Minho511/pdt-cgqa-czsl", filename='ckpt_best_auc_cgqa.t7')
checkpoint = torch.load(ckpt_path, map_location='cpu')
model.load_state_dict(checkpoint['net'], strict=False)
model.eval()

def predict(image, alpha):
    image = Image.fromarray(image).convert("RGB")
    image = transform(image).unsqueeze(0)
    image = backbone(image)
    model.alpha = alpha
    _, pred = model.demo_forward(image.to(args.device)) # top-1 or top-k 예측
    sorted_preds = sorted(pred.items(), key=lambda x: x[1].item(), reverse=True)
    return {f"{k[0]} {k[1]}": float(v) for k, v in sorted_preds}  # dict for gr.Label


demo = gr.Interface(fn=predict,
                    inputs=[
                        gr.Image(type="numpy"),
                        gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.01, label="Alpha (weight)")
                    ],
                    outputs=gr.Label(num_top_classes=5),
                    examples=[
                        ["examples/ex1.jpeg", 0.5],
                        ["examples/ex2.jpeg", 0.5],
                        ["examples/ex3.jpeg", 0.5],
                    ],
                    title="Compositional Zero-Shot Inference Demo")

demo.launch()
