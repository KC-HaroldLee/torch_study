#%%
from pickletools import optimize
import torch
import torch.nn as nn #기본 구성요소
import torch.nn.functional as F # 자주사용되는 모듈
import torch.optim as optim # 가중치 추정
from torchvision import datasets, transforms #

from matplotlib import pyplot as plt

#%%
is_cuda = torch.cuda.is_available()
device = torch.device('cuda' if is_cuda else 'cpu')

print('set 완료', device)

#%%
batch_size = 50
epoch_num = 15
lr = 0.0001

train_data = datasets.MNIST('./datasets', train=True, download=True,
                            transform= transforms.ToTensor()) # 저장과 동시에 전치리를 한다.
test_data = datasets.MNIST('./datasets', train=False, download=True,
                            transform= transforms.ToTensor())

#%%
image, label = train_data[0]

print(type(image)) # <class 'torch.Tensor'>
print(type(label)) # <class 'int'>
                
# %%
# print(dir(image))
print(image.numpy()) # 와우!

# %%
plt.imshow(image.squeeze().numpy(), cmap='gray') # 스퀴즈는 차원 줄이기
plt.title('label is :'+str(label))
plt.show()
# %%

## 미니 배치 구성하기

train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                           batch_size=batch_size,
                                           shuffle=True)

first_batch = train_loader.__iter__().__next__()
print('{:15s} | {:<25s} | {}'.format('name', 'type', 'size'))
print('{:15s} | {:<25s} | {}'.format('Num of Batch', '',len(train_loader)))
print('{:15s} | {:<25s} | {}'.format('first_batch', str(type(first_batch)), len(first_batch)))
print('{:15s} | {:<25s} | {}'.format('first_batch[0]', str(type(first_batch[0])), first_batch[0].shape))
print('{:15s} | {:<25s} | {}'.format('first_batch[1]', str(type(first_batch[1])), first_batch[1].shape))



# %%
# CNN layer만들기
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__() # nn.Module을 상속받는다.
        self.conv1 = nn.Conv2d(1,32,3,1, padding=0) # 0이 기본이긴하다.
        self.conv2 = nn.Conv2d(32,64,3,1, padding=0)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128) # 이빨이
        self.fc2 = nn.Linear(128, 10) # 이빨이

    def forward(self, x): # 어쩌면 이름을 맞춰야할 수도. 'x'가 이제 이해된다.
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x ,2) # 물론 이렇게 바로 생성도 된다.
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim = 1)
        return output
# %%
# 이제 모델과 옵티마이저 세팅
model = CNN().to(device=device)
optimizer = optim.Adam(model.parameters(), lr = lr) # 장착한다
criterion = nn.CrossEntropyLoss() # 이름이 왜 이럴까 표준이다. 아직 장착 ㄴ

# %%

print(model) # 써머리다
# %%
model.train()
i = 0

# 신기하게 학습을 반복문으로 넣는다.
for epoch in range(epoch_num) :
    for data, label in train_loader :
        data = data.to(device)
        label = label.to(device)
        optimizer.zero_grad() # 이미 장착되어 있으니까 - 초기화
        output = model(data)

        loss = criterion(output, label) # 검사한다.
        loss.backward()
        optimizer.step()

        if i % 1000 == 0 :
            print ('Train Step:{}\tLoss: {:.3f}'.format(i, loss.item()))
        i += 1
# %%
# 평가한다.
model.eval()
correct = 0
for data, label in test_loader:
    data = data.to(device)
    label = label.to(device)
    output = model(data)
    prediction = output.data.max(1)[1]
    correct += prediction.eq(label.data).sum()

print('테스트 결과 :', (100*correct / len(test_loader.dataset)))
# %%
