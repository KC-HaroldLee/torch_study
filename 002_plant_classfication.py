#%%

import os
import shutil
import math

#%%

datasets_dir = './datasets/plant/'
classes_list = os.listdir(datasets_dir)

base_dir = './datasets/plant_splitted/'

train_dir = base_dir + 'train'
val_dir = base_dir + 'val'
test_dir = base_dir + 'test'
print(os.path.isdir(train_dir))
print(os.path.isdir(val_dir))
print(os.path.isdir(test_dir))

#%%

# for class_name in classes_list :
#     os.mkdir(os.path.join(train_dir, class_name))
#     os.mkdir(os.path.join(val_dir, class_name))
#     os.mkdir(os.path.join(test_dir, class_name))
# %%

# for class_name in classes_list :
#     path = os.path.join(datasets_dir, class_name)
#     file_name_list = os.listdir(path)

#     train_size = math.floor(len(file_name_list) * 0.6)
#     val_size = math.floor(len(file_name_list) * 0.2)
#     test_size = math.floor(len(file_name_list) * 0.2)

#     # print(train_size, val_size, test_size)

#     # 이렇게 구분하는 방법도 있군
#     train_file_name_list = file_name_list[:train_size]
#     train_class_folder_path = os.path.join(train_dir, class_name)
#     # print('train size-[{}]:{}'.format(class_name, len(train_file_name_list)))
#     # print(len(train_file_name_list))
#     for train_file_name in train_file_name_list :
#         src_file = os.path.join(path, train_file_name)
#         dst_file = os.path.join(train_class_folder_path, train_file_name)
#         shutil.copyfile(src_file, dst_file)


#     val_file_name_list = file_name_list[train_size:train_size+val_size] # 신박하네 ㅋㅋ
#     val_class_folder_path = os.path.join(val_dir, class_name)
#     # print('val size-[{}]:{}'.format(class_name, len(val_file_name_list)))
#     # print(len(val_file_name_list))
#     for val_file_name in val_file_name_list :
#         src_file = os.path.join(path, val_file_name)
#         dst_file = os.path.join(val_class_folder_path, val_file_name)
#         shutil.copyfile(src_file, dst_file)


#     test_file_name_list = file_name_list[train_size+val_size:train_size+val_size+test_size] # 파일 개수가 많지 않겠군
#     test_class_folder_path = os.path.join(test_dir, class_name)
#     # print('test size-[{}]:{}'.format(class_name, len(test_file_name_list)))
#     # print(len(test_file_name_list))
#     for test_file_name in test_file_name_list :
#         src_file = os.path.join(path, test_file_name)
#         dst_file = os.path.join(test_class_folder_path, test_file_name)
#         shutil.copyfile(src_file, dst_file)
# %%

import torch

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

BATCH_SIZE = 1024
EPOCH = 30

#%%
import torch

torch.__version__
#%%
# 베이스라인 모델 학습을 위한 준비
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder # 무슨 모듈?

# compose는 이미지 전처리, 증강등에 이용되는 메소드이다. 
# 이미지 크기, 그리고 데이터형식(Tensor형태)로 변환해 준다.
transform_base = transforms.Compose([transforms.Resize((64,64)), transforms.ToTensor()]) # 형식을 지정하는 듯 하다

# 폴더 이름을 클래스이름으로 사용할 때 ImageFolder를 사용한다. transform은 데이터를 불러온 후 전처리, 증강을 위한 방법을 지정한다. 앞에서 정의한 것을 지정한다
train_ds = ImageFolder(root='datasets/plant_splitted/train/', transform=transform_base)
val_ds = ImageFolder(root='datasets/plant_splitted/val/', transform=transform_base)

# %%
from torch.utils.data import DataLoader

# train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4) # 오 워커까지

# val_loader = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4) # 오 워커까지
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)


# %%

# 모델 설계
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(4096, 512) #Dense!?
        self.fc2 = nn.Linear(512, len(classes_list)) #ㅋㅋㅋㅋㅋㅋ

    def forward(self, x):

        x = self.conv1(x)
        x = F.relu(x) # 액티베이션 레이어구먼 스위시가 없네
        x = self.pool(x)
        x = F.dropout(x, p=0.25, training=self.training) # 여기서의 self.training은 상속 받았음(nn.Module)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = F.dropout(x, p=0.25, training=self.training)
    
        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool(x)
        x = F.dropout(x, p=0.25, training=self.training)

        x = x.view(-1, 4096) # 플랫튼 같은데...
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)

base_model = Net().to(DEVICE)
optimizer = optim.Adam(base_model.parameters(), lr=0.001) 
# %%
# # 트레이닝... 근데 val이없음
# def train(model, train_loader, optimizer):
#     model.train() # nn.Module 에서 상속 받음
#     print('ㄷㄷㄷ', end='')
#     for batch_idx, (data, target) in enumerate(train_loader):
#         data, target = data.to(DEVICE), target.to(DEVICE)
#         optimizer.zero_grad()
#         output = model(data)
#         loss = F.cross_entropy(output, target)
#         loss.backward()
#         optimizer.step() # 파라미터에 할당된 Gradient값을 이용해 모델의 파라미터를 업데이트한다.

#     print('KKKKKKKKKK')

def train(model, train_loader, optimizer):
    model.train()  
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE) 
        optimizer.zero_grad() 
        output = model(data)  
        loss = F.cross_entropy(output, target) 
        loss.backward()  
        optimizer.step()  
# %%
def evaluate(model, test_loader):
    model.eval()  
    test_loss = 0 
    correct = 0   
    
    with torch.no_grad(): 
        for data, target in test_loader:  
            data, target = data.to(DEVICE), target.to(DEVICE)  
            output = model(data) 
            
            test_loss += F.cross_entropy(output,target, reduction='sum').item() 
 
            
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item() 
   
    test_loss /= len(test_loader.dataset) 
    test_accuracy = 100. * correct / len(test_loader.dataset) 
    return test_loss, test_accuracy  
# %%
# 모델학습 실행하기
import time
import copy

# def train_baseline(model, train_loader, val_loader, # 여기 있군
#                 optimizer, num_epoch = 30):
    
#     best_acc = 0.0
#     best_model_weights = copy.deepcopy(model.state_dict()) # 가중치는 여기 있는 듯하다

#     for epoch in range(1, num_epoch + 1): # 그냥 1 시프트네 프린트를 위한것이다.
#         since = time.time()
#         train(model, train_loader, optimizer)
#         train_loss, train_acc = evaluate(model, train_loader)
#         val_loss, val_acc = evaluate(model, val_loader)

#         if  val_acc > best_acc:
#             best_acc = val_acc
#             best_model_weights = copy.deepcopy(model.state_dict()) # 가중치는 여기 있는 듯하다

#         time_elapsed = time.time() - since

#         print('----------------- epoch {} -----------------'.format(epoch))
#         print('train Loss : {:.4f}, Acc: {:.2f}%'.format(train_loss, train_acc))
#         print('val Loss : {:.4f}, Acc: {:.2f}%'.format(val_loss, val_acc))
#         print('Completeed in {:.0f}m, {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

#     model.load_state_dict(best_model_weights) # 가중치를 가져오는...
#     return model

def train_baseline(model ,train_loader, val_loader, optimizer, num_epochs = 30): # 여기 있군 로더
    best_acc = 0.0  
    best_model_wts = copy.deepcopy(model.state_dict()) # 가중치는 여기잇다.
  
    for epoch in range(1, num_epochs + 1): #프린트를 위한 시프트
        since = time.time()  
        train(model, train_loader, optimizer)
        train_loss, train_acc = evaluate(model, train_loader) 
        val_loss, val_acc = evaluate(model, val_loader)
        
        if val_acc > best_acc: 
            best_acc = val_acc 
            best_model_wts = copy.deepcopy(model.state_dict()) 
        
        time_elapsed = time.time() - since 
        print('-------------- epoch {} ----------------'.format(epoch))
        print('train Loss: {:.4f}, Accuracy: {:.2f}%'.format(train_loss, train_acc))   
        print('val Loss: {:.4f}, Accuracy: {:.2f}%'.format(val_loss, val_acc))
        print('Completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)) 
    model.load_state_dict(best_model_wts)  # 가중치 가져오기
    return model
 

base = train_baseline(base_model, train_loader, val_loader, optimizer, EPOCH)  	 #(16)
torch.save(base,'baseline.pt')

#%%
base = train_baseline(base_model, train_loader, val_loader, optimizer, EPOCH)

#%%
torch.save(base, 'baseline.pt')


# %%

# %%

# %%
