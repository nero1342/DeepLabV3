import torch 
import torch.nn as nn 
from torchvision import models 

class DeeplabV3(nn.Module):
    def __init__(self, numclasses):
        super().__init__() 
        
        model = models.segmentation.deeplabv3_resnet101(pretrained=True)
        model.classifier[-1] = torch.nn.Conv2d(256, numclasses, kernel_size=1)
        model.aux_classifier[-1] = torch.nn.Conv2d(256, numclasses, kernel_size=1)

        self.model = model 
    def forward(self, x):
        return self.model(x)['out']
    
if __name__ == "__main__":
    model = DeeplabV3(numclasses=40)
    from torchsummary import summary 

    summary(model)
    img = torch.rand(2, 3, 480, 480)

    x = model(img)
    y = x.argmax(1)
    print(x.shape, y.shape)
