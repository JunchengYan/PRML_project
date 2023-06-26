import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.resnet import resnet50
from model import ResNet18
from data import StairDataset, load_and_split_data
from torch.utils.data import DataLoader
from torchvision import transforms
from main import _init_, test_and_eval, train
from utils import IOStream
import argparse
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
from sklearn import metrics
from torchstat import stat

class KD_loss(nn.Module):
    def __init__(self, T):
        super(KD_loss, self).__init__()
        self.T = T

    def forward(self, out_s, out_t):
        loss = F.kl_div(F.log_softmax(out_s / self.T, dim=1),
                        F.softmax(out_t / self.T, dim=1),
                        reduction='batchmean') * self.T * self.T

        return loss

# """
def train_distill(args, io, teacher_model, student_model, train_loader, valid_loader, optimizer, scheduler, device, epochs):
    alpha = 0.9
    train_losses = torch.ones(epochs)
    train_accs = torch.ones(epochs)
    valid_losses = torch.ones(epochs)
    valid_accs = torch.ones(epochs)

    best_acc = 0.0
    cnt = 0
    student_model.to(device)
    teacher_model.to(device)
    teacher_model.eval()
    for epoch in range(epochs):
        # train
        student_model.train()
        io.cprint("\nEpoch %d | Learning Rate is %.6f" % (epoch+1, optimizer.param_groups[0]['lr']))
        count = 0
        train_loss = 0.0
        train_acc = 0.0
        train_pred = []
        train_true = []
        for data, label in tqdm(train_loader):
            data = data.float().to(device)
            label = label.float().to(device)
            batch_size = data.size()[0]
            
            optimizer.zero_grad()
            
            student_output = student_model(data)
            
            
            teacher_output = teacher_model(data)
            teacher_output = teacher_output.detach()
            
            # teacher_output = F.softmax(teacher_output, dim=-1)
            loss_base = F.cross_entropy(student_output, label.long())
            loss_kd = KD_loss(T=8)(student_output, teacher_output)  # 通过老师的 teacher_output训练学生的output
            loss = (1-alpha)*loss_base + alpha*loss_kd
            loss.backward()
            
            optimizer.step()
            
            student_output = F.softmax(student_output, dim=-1)
            preds = student_output.max(dim=1)[1] # 预测标签
            
            count += batch_size
            train_loss += loss.item() * batch_size
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())
        
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        
        
        train_acc = metrics.accuracy_score(train_true, train_pred)
        train_loss = train_loss*1.0/count
        train_losses[epoch] = train_loss
        train_accs[epoch] = train_acc
        
        io.cprint("Epoch %d | Mean train loss is %.6f, Train acc is %.3f" % 
              (epoch+1, train_loss, train_acc*100.00))  

        scheduler.step()
        
        # valid
        student_model.eval()
        count = 0
        valid_loss = 0.0
        valid_acc = 0.0
        valid_true = []
        valid_pred = []
        for data, label in tqdm(valid_loader):
            data = data.float().to(device)
            label = label.float().to(device)
            batch_size = data.size()[0]
            
            student_output = student_model(data)
            
            loss_base = F.cross_entropy(student_output, label.long())
            # loss_kd = KD_loss(T=10)(student_output, teacher_output)  # 通过老师的 teacher_output训练学生的output
            loss = loss_base
            
            student_output = F.softmax(student_output, dim=-1)
            preds = student_output.max(dim=1)[1]
            
            count += batch_size
            
            valid_loss += loss.item() * batch_size
            valid_true.append(label.cpu().numpy())
            valid_pred.append(preds.detach().cpu().numpy())
                
        
        valid_true = np.concatenate(valid_true)
        valid_pred = np.concatenate(valid_pred)
        valid_acc = metrics.accuracy_score(valid_true, valid_pred)
        valid_loss = valid_loss*1.0/count
        
        valid_losses[epoch] = valid_loss
        valid_accs[epoch] = valid_acc
        
        if valid_acc >= best_acc:
            best_acc = valid_acc
            torch.save(student_model.state_dict(), './logs/%s/models/model.t7' % args.exp_name)
            cnt=0
        else:
            cnt+=1
        
        io.cprint("Epoch %d | Mean valid loss is %.6f, valid acc is %.3f" % 
                (epoch+1, valid_loss, valid_acc*100.00))
        io.cprint("Best valid acc is %.3f" % (best_acc*100.00))
        if cnt >= 7:
            train_accs = train_accs[:epoch]
            train_losses = train_losses[:epoch]
            valid_accs = valid_accs[:epoch]
            valid_losses = valid_losses[:epoch]
            break
        
        
    # 画图
    l = len(train_losses)
    plt.figure()
    plt.plot(torch.arange(0, l, step=1), train_accs, 'b', label='Training accuracy')
    plt.plot(torch.arange(0, l, step=1), valid_accs, 'r', label='validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend(loc='lower right')
    plt.savefig(("./logs/%s/acc.jpg" % args.exp_name), dpi=300)
    plt.figure()
    plt.plot(torch.arange(0, l, step=1), train_losses, 'b', label='Training loss')
    plt.plot(torch.arange(0, l, step=1), valid_losses, 'r', label='validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig(("./logs/%s/loss.jpg" % args.exp_name), dpi=300)
    return student_model
# """

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Images classification')
    parser.add_argument('--exp_name', type=str, default='exp', help='Name of the experiment')
    args = parser.parse_args()
    _init_(args)
    io = IOStream('./logs/' + args.exp_name + '/run.log')
    
    
    
    train_transform = transforms.Compose([
        # transforms.RandomRotation(30, center=(0, 0), expand=True),
        # transforms.RandomHorizontalFlip(0.5),
        # transforms.RandomRotation(90),
        transforms.Resize([96, 96]),
        transforms.ToTensor(),
        # transforms.Normalize(mean, std)
    ]) 
    test_transform = transforms.Compose([
        # transforms.RandomHorizontalFlip(0.5),
        # transforms.RandomRotation(90),
        transforms.Resize([96, 96]),
        transforms.ToTensor(),
        # transforms.Normalize(mean, std)
    ])
    trainset, validset, testset = load_and_split_data('./stair/public/')
    train_set = StairDataset(trainset, 'train', train_transform)
    valid_set = StairDataset(validset, 'valid', test_transform)
    test_set = StairDataset(testset, 'test', test_transform)
    train_loader = DataLoader(train_set, batch_size=8, num_workers=16, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_set, batch_size=1, num_workers=16, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=1, num_workers=16, shuffle=False, drop_last=False)
    
    teacher_model = resnet50(pretrained=True)
    num_ftrs = teacher_model.fc.in_features
    teacher_model.fc = nn.Linear(num_ftrs, 2)
    teacher_model.load_state_dict(torch.load(('./logs/res50/models/model.t7')))
    
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # # lr_adam = 1e-4
    # # epochs = 30
    # # optimizer = torch.optim.Adam(teacher_model.parameters(), lr=lr_adam, weight_decay=1e-4)
    # # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=0, last_epoch=-1)
    # # teacher_model = train(args, io, teacher_model, train_loader, valid_loader, optimizer, scheduler, device, epochs)
    # test_and_eval(args, io, teacher_model, test_loader, device, eval=False, best_path=('./logs/res50/models/model.t7'))
    
    
    
    
    student_model = ResNet18(num_classes=2, use_dropout=False, use_init=False)
    lr_adam = 1e-4
    epochs = 5
    optimizer = torch.optim.Adam(student_model.parameters(), lr=lr_adam, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=0, last_epoch=-1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # student_model = train(args, io, student_model, train_loader, valid_loader, optimizer, scheduler, device, epochs)
    # test_and_eval(args, io, student_model, test_loader, device, eval=False, best_path=('./logs/%s/models/model.t7' % args.exp_name))
    
    # student_model.load_state_dict(torch.load(('./logs/distill_exp4/models/model.t7')))
    # stat(student_model, (3,96,96))
    
    student_model.load_state_dict(torch.load(('./logs/distill_exp4/models/model.t7')))
    test_and_eval(args, io, student_model, test_loader, device, eval=True, best_path=('./logs/distill_exp4/models/model.t7'))
    # student_model = train_distill(args, io, teacher_model, student_model, train_loader, valid_loader, optimizer, scheduler, device, epochs)
    # test_and_eval(args, io, student_model, test_loader, device, eval=False, best_path=('./logs/%s/models/model.t7' % args.exp_name))
    

    