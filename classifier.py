import torch
from torchvision import datasets, transforms, models
import time
from PIL import Image
import myModels

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, root, datatxt, transform=None, target_transform=None):
        super(MyDataset, self).__init__()
        fh = open(root + datatxt, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], int(words[1])))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = Image.open(self.root + fn).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)


# load data
transform = transforms.Compose([transforms.RandomCrop(150), transforms.ToTensor(),
                                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

train_set = MyDataset(root='./', datatxt='train_label.txt', transform=transform)
test_set = MyDataset(root='./', datatxt='test_label.txt', transform=transform)

batch_size = 64

train_loader = torch.utils.data.DataLoader(
                 dataset=train_set,
                 batch_size=batch_size,
                 shuffle=True)

test_loader = torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=batch_size,
                shuffle=False)

# preparation
model = myModels.SPPNet(7)
model_name = "SPPNet"
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

for epoch in range(3):
    # use GPU to accelerates
    model.cuda()

    # train the model
    start = time.process_time()

    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = torch.nn.CrossEntropyLoss()
        loss = loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 16 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

    elapsed = (time.process_time() - start)
    print('Train Epoch: {} [FINISH]\t Time used: {}'.format(epoch, elapsed))

    # test the model
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            test_loss += torch.nn.functional.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    # save the model
    torch.save(model.state_dict(), "model_parameters" + model_name + ".pt")