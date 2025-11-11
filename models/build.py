from .backbones import resnet, hrnet, vit
from .decoders import deconv_head, duc_head, dcn_head, transformer_head

def build_backbone(cfg):
    if cfg['type'] == 'resnet':
        return resnet.ResNet(cfg['depth'])
    elif cfg['type'] == 'hrnet':
        return hrnet.HRNet(cfg['width'])
    elif cfg['type'] == 'vit':
        return vit.ViT(cfg['name'])
    else:
        raise ValueError(f"Unknown backbone {cfg['type']}")

def build_decoder(cfg, num_keypoints):
    if cfg['type'] == 'deconv':
        return deconv_head.DeconvHead(cfg, num_keypoints)
    elif cfg['type'] == 'duc':
        return duc_head.DUCHead(cfg, num_keypoints)
    elif cfg['type'] == 'dcn':
        return dcn_head.DCNHead(cfg, num_keypoints)
    elif cfg['type'] == 'transformer':
        return transformer_head.TransformerHead(cfg, num_keypoints)
    else:
        raise ValueError(f"Unknown decoder {cfg['type']}")
