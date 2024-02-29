import torch
import timm
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from transformers import ViTModel, ViTConfig
from efficientnet_pytorch import EfficientNet


class CustomViT(nn.Module):
    def __init__(self, num_classes=9):
        super(CustomViT, self).__init__()
        config = ViTConfig.from_pretrained('model/config.json')
        self.vit = ViTModel.from_pretrained('model/pytorch_model.bin', config=config)

        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(768, num_classes)

    def forward(self, x):
        outputs = self.vit(x)
        x = outputs.last_hidden_state

        x = self.dropout(x[:, 0])  
        out = self.fc(x)
        return out
    



class CustomEfficientNet(nn.Module):
    def __init__(self, num_classes=9, model_name='efficientnet-b0', pretrained=True, dropout_rate=0.3):
        super(CustomEfficientNet, self).__init__()
        
        if pretrained:
            self.model = EfficientNet.from_pretrained(model_name)
        else:
            self.model = EfficientNet.from_name(model_name)
        
        
        num_features = self.model._fc.in_features 
        self.dropout = nn.Dropout(p=dropout_rate)  
        self.model._fc = nn.Linear(num_features, num_classes)  

    def forward(self, x):
        x = self.model.extract_features(x)  
        x = self.model._avg_pooling(x)  
        x = x.flatten(start_dim=1)  
        x = self.dropout(x)  
        outputs = self.model._fc(x)  
        return outputs



if __name__ == '__main__':
    model = CustomEfficientNet(num_classes=9, pretrained=True)
    x = torch.randn(30, 3, 224, 224)
    output = model(x)

    import ipdb;ipdb.set_trace()