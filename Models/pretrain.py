import torchvision.models as models
import torch
from Models.HAR_model import DCNN
from Models.speech_model import vgg11
from transformers import BertForSequenceClassification

def ResNet18_pretrained(n_classes):
    classifier = models.resnet18(pretrained=True)
    classifier.fc = torch.nn.Linear(512, n_classes)
    for k, v in classifier.named_parameters():
        if 'fc' not in k:
            v.requires_grad = False
    return classifier

def ResNet50_pretrained(n_classes):
    classifier = models.resnet50(pretrained=True)
    classifier.fc = torch.nn.Linear(1024, n_classes)
    for k, v in classifier.named_parameters():
        if 'fc' not in k:
            v.requires_grad = False
    return classifier

def ResNet101_pretrained(n_classes):
    classifier = models.resnet101(pretrained=True)
    classifier.fc = torch.nn.Linear(2048, n_classes)
    for k, v in classifier.named_parameters():
        if 'fc' not in k:
            v.requires_grad = False
    return classifier

def DCNN_pretrained(n_classes=6, path=None):
    model = DCNN(output=n_classes)
    model.load_state_dict(torch.load(path))
    model.lin1 = torch.nn.Linear(6, 400)
    model.lin2 = torch.nn.Linear(400, n_classes)
    model.to('cpu')
    for k, v in model.named_parameters():
        if 'lin' not in k:
            v.requires_grad = False
    return model

def VGG11_pretrained(n_classes=10, path='/root/Experiments/Delta/results/pretrain/speechcommand/VGG11.pth'):
    model = vgg11()
    model.load_state_dict(torch.load(path))
    model.classifier = torch.nn.Sequential(
            torch.nn.Linear(512 * 1 * 1, 512),
            torch.nn.ReLU(True),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(True),
            torch.nn.Linear(512, n_classes),
        )
    for k, v in model.named_parameters():
        if 'classifier' not in k:
            v.requires_grad = False
    return model
    
def Bert_pretrained(n_classes=10, root_path=None):
    path_bert1 = root_path + "Models/Pretrain/textclassification/bert_pretrain"
    path_bert2 = root_path = "Models/Pretrain/textclassification/Bert.pth"
    model = BertForSequenceClassification.from_pretrained(path_bert1, 
                                                          num_labels=10,
                                                          output_attentions=False,
                                                          output_hidden_states = True)
    model.load_state_dict(torch.load(path_bert2))
    model.classifier = torch.nn.Linear(768, n_classes)
    for k, v in model.named_parameters():
        if 'classifier' not in k:
            v.requires_grad = False
    return model