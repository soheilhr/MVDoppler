import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor

from transformer import PatchEmbedding, GPT

class FusionResNet(torch.nn.Module):
    def __init__(self, args):
        super(FusionResNet, self).__init__()  
        num_classes = args.train.num_classes      
        self.pretrained = args.train.use_pretrain
        self.fusion_level = args.fusion.fusion_level
        self.fusion_mode = args.fusion.fusion_mode

        return_nodes = {
                        'layer1': 'layer1',
                        'layer2': 'layer2',
                        'layer3': 'layer3',
                        'layer4': 'layer4',
                        }
        self.n_extractor = len(return_nodes)
        ResNet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=self.pretrained)
        self.extractor = create_feature_extractor(ResNet, return_nodes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) 
        self.classifier =  nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes),
        )
        if 'late' in self.fusion_level:
            if 'transformer' in self.fusion_mode:
                self.transformer_late = GPT(n_embd=512,
                                n_head=8,               
                                block_exp=4,
                                n_layer=1,
                                T_anchor_dim=4*4,      # (TODO): must be modified to deal with variable input size
                                seq_len=1,
                                embd_pdrop = 0.1,
                                resid_pdrop = 0.1,
                                attn_pdrop = 0.1)
        elif 'multi' in self.fusion_level:
            channel_list = [64, 128, 256, 512]
            # channel_list = 256
            if 'transformer' in self.fusion_mode:
                PatchEmbedding_list = [PatchEmbedding(n_channel=channel_list[i], patch_size=2**(3-i), emb_size=512) for i in range(self.n_extractor)]
                # PatchEmbedding_list = [PatchEmbedding(n_channel=channel_list[i], patch_size=patchsize_list[i], emb_size=256) for i in range(self.n_extractor)]
                transformer_list = [GPT(n_embd=512,
                                n_head=8,               
                                block_exp=4,
                                n_layer=1,
                                T_anchor_dim=4*4,      # (TODO): must be modified to deal with variable input size
                                seq_len=1,
                                embd_pdrop = 0.1,
                                resid_pdrop = 0.1,
                                attn_pdrop = 0.1)
                                for i in range(self.n_extractor)]
                self.PatchEmbedding_list = nn.ModuleList(PatchEmbedding_list)
                self.transformer_list = nn.ModuleList(transformer_list)
            else:
                conv1x1 = [nn.Conv2d(channel_list[i], 512, kernel_size=1) for i in range(self.n_extractor)]
                self.conv1x1 = nn.ModuleList(conv1x1)

    def forward(self, x):
        if 'single' in self.fusion_level:
            if '1' in self.fusion_level:
                x_input = x[:,0,:,:,:]
            elif '2' in self.fusion_level:
                x_input = x[:,1,:,:,:]
            elif 'all' in self.fusion_level:
                x_input = x
            x_list = self.extractor(x_input)
            x = x_list[f'layer{self.n_extractor}']
            x = self.avgpool(x)
        elif 'input' in self.fusion_level:
            x_input = x.mean(dim=1)
            x_input[:,0,:,:] = x[:,0,0,:,:]
            x_input[:,1,:,:] = x[:,0,0,:,:]
            x_input[:,2,:,:] = x[:,1,0,:,:]
            x_list = self.extractor(x_input)
            x = x_list[f'layer{self.n_extractor}']
            x = self.avgpool(x)
        else:
            x1_list = self.extractor(x[:,0,:,:,:])
            x2_list = self.extractor(x[:,1,:,:,:])
            if 'late' in self.fusion_level:
                x1 = x1_list[f'layer{self.n_extractor}']
                x2 = x2_list[f'layer{self.n_extractor}']
                if 'average' in self.fusion_mode:
                    x1 = self.avgpool(x1)
                    x2 = self.avgpool(x2)
                    x = (x1+x2)/2
                if 'transformer' in self.fusion_mode:
                    x1 = x1.view(x1.shape[0], x1.shape[1], -1)
                    x2 = x2.view(x2.shape[0], x2.shape[1], -1)
                    x1, x2 = self.transformer_late(x1, x2)
                    x1 = self.avgpool(x1.unsqueeze(-1))
                    x2 = self.avgpool(x2.unsqueeze(-1))
                    x = (x1+x2)/2
            elif 'multi' in self.fusion_level:
                x1_multi = [x1_list[f'layer{i+1}'] for i in range(self.n_extractor)]
                x2_multi = [x2_list[f'layer{i+1}'] for i in range(self.n_extractor)]
                if 'transformer' in self.fusion_mode:       # Multi-level transformer-based fusion
                    for i in range(len(x1_multi)):
                        x1 = self.PatchEmbedding_list[i](x1_multi[i])
                        x2 = self.PatchEmbedding_list[i](x2_multi[i])
                        x1, x2 = self.transformer_list[i](x1, x2)
                        x1_l = self.avgpool(x1.unsqueeze(-1))
                        x2_l = self.avgpool(x2.unsqueeze(-1))
                        x_l = (x1_l+x2_l)/2
                        if i==0:
                            x = x_l
                        else:
                            x += x_l
                            # x = torch.concat((x,x_l),dim=1)
                    x = x/len(x1_multi)
                elif 'average' in self.fusion_mode:              # Multi-level average-based fusion
                    for i in range(len(x1_multi)):
                        x1 = self.conv1x1[i](x1_multi[i])
                        x2 = self.conv1x1[i](x2_multi[i])
                        x1_l = self.avgpool(x1)
                        x2_l = self.avgpool(x2)
                        x_l = (x1_l+x2_l)/2
                        if i==0:
                            x = x_l
                        else:
                            x += x_l
                    x = x/len(x1_multi)  

        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
