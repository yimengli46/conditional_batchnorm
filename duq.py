#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F

import numpy as np
import sklearn.datasets

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

class Model_bilinear(nn.Module):
    def __init__(self, features, num_classes): # num_embeddings equals num_classes
        super().__init__()
        
        self.gamma = 0.99
        self.sigma = 0.3
        
        embedding_size = 10
        
        self.fc1 = nn.Linear(2, features)
        self.fc2 = nn.Linear(features, features)
        self.fc3 = nn.Linear(features, features)
        
        # embedding_size is # of centroids
        # W.shape = num_centroids x num_classes x feature_size
        self.W = nn.Parameter(torch.normal(torch.zeros(embedding_size, num_classes, features), 1)) 
        
        self.register_buffer('N', torch.ones(num_classes) * 20) # self.N.shape = torch.Size([2])
        self.register_buffer('m', torch.normal(torch.zeros(embedding_size, num_classes), 1)) # self.m.shape = torch.Size([10, 2])
        
        self.m = self.m * self.N.unsqueeze(0) # self.m.shape = torch.Size([10, 2])

    def embed(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x) # x.shape = batch_size x feature_size
        #print('x.shape = {}'.format(x.shape))
        #assert 1==2

        # i is batch, m is embedding_size, n is num_classes (classes)
        x = torch.einsum('ij,mnj->imn', (x, self.W))
        
        return x

    def bilinear(self, z):
        embeddings = self.m / self.N.unsqueeze(0) #embeddings.shape = torch.Size([10, 2])
        #print('embeddings.shape = {}'.format(embeddings.shape))
        
        # implement Eq (1) in the paper
        diff = z - embeddings.unsqueeze(0)            
        y_pred = (- diff**2).mean(1).div(2 * self.sigma**2).exp()

        return y_pred

    def forward(self, x):
        z = self.embed(x) # z: batch_size x num_centroids x num_classes, z.shape = torch.Size([64, 10, 2])

        y_pred = self.bilinear(z) # y_pred.shape = torch.Size([64, 2])

        return z, y_pred

    def update_embeddings(self, x, y):
        z = self.embed(x)
        
        # normalizing value per class, assumes y is one_hot encoded
        # implement Eq (4)
        self.N = torch.max(self.gamma * self.N + (1 - self.gamma) * y.sum(0), torch.ones_like(self.N))
        
        # compute sum of embeddings on class by class basis
        features_sum = torch.einsum('ijk,ik->jk', (z, y))
        
        # implement Eq (5)
        self.m = self.gamma * self.m + (1 - self.gamma) * features_sum


np.random.seed(0)
torch.manual_seed(0)

l_gradient_penalty = 1.0

# Moons
noise = 0.1
#X_train, y_train = sklearn.datasets.make_moons(n_samples=1500, noise=noise)
#X_test, y_test = sklearn.datasets.make_moons(n_samples=200, noise=noise)

n_train_samples = 20000
centers = [(-5, -5), (5, 5)]
X_train, y_train = make_blobs(n_samples=n_train_samples, centers=centers, shuffle=False, random_state=40)
n_test_samples = 10000
centers = [(-5, -5), (5, 5)]
X_test, y_test = make_blobs(n_samples=n_test_samples, centers=centers, shuffle=False, random_state=20)

num_classes = 2
batch_size = 63

model = Model_bilinear(20, num_classes)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

# implement Eq (7)
def calc_gradient_penalty(x, y_pred):
    gradients = torch.autograd.grad(
            outputs=y_pred,
            inputs=x,
            grad_outputs=torch.ones_like(y_pred),
            create_graph=True,
        )[0]

    #print('x.shape = {}'.format(x.shape))
    #print('y_pred.shape = {}'.format(y_pred.shape))
    #print('A gradients.shape = {}'.format(gradients.shape))
    gradients = gradients.flatten(start_dim=1)
    #print('B gradients.shape = {}'.format(gradients.shape))
    # L2 norm
    grad_norm = gradients.norm(2, dim=1)
    #print('C grad_norm.shape = {}'.format(grad_norm.shape))
    # Two sided penalty
    gradient_penalty = ((grad_norm - 1) ** 2).mean()
    # One sided penalty
    #gradient_penalty = (torch.max((grad_norm - 1), torch.zeros_like(grad_norm))).mean()
    #print('D gradient_penalty.shape = {}'.format(gradient_penalty.shape))
    # One sided penalty - down
#     gradient_penalty = F.relu(grad_norm - 1).mean()

    return gradient_penalty


def output_transform_acc(output):
    y_pred, y, x, z = output
    
    y = torch.argmax(y, dim=1)
        
    return y_pred, y


def output_transform_bce(output):
    y_pred, y, x, z = output

    return y_pred, y


def output_transform_gp(output):
    y_pred, y, x, z = output

    return x, y_pred


def step(batch):
    model.train()
    optimizer.zero_grad()
    
    x, y = batch
    x.requires_grad_(True)
    
    z, y_pred = model(x)
    
    loss1 =  F.binary_cross_entropy(y_pred, y)
    #loss2 = l_gradient_penalty * calc_gradient_penalty(x, y_pred)
    
    #loss = loss1 + loss2
    loss = loss1
    
    loss.backward()
    optimizer.step()
    
    with torch.no_grad():
        model.update_embeddings(x, y)
    
    return loss.item()


def eval_step(batch):
    model.eval()

    x, y = batch

    x.requires_grad_(True)

    z, y_pred = model(x)

    return y_pred, y, x, z
    
y_train_targets = np.zeros((y_train.shape[0], 2))
y_train_targets[y_train==0, 0] = 1
y_train_targets[y_train==1, 1] = 1
ds_train = torch.utils.data.TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train_targets).float())
dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=True)

y_test_targets = np.zeros((y_test.shape[0], 2))
y_test_targets[y_test==0, 0] = 1
y_test_targets[y_test==1, 1] = 1
ds_test = torch.utils.data.TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test_targets).float())
dl_test = torch.utils.data.DataLoader(ds_test, batch_size=200, shuffle=False)



num_epochs = 30
for epoch_num in range(num_epochs):
    epoch_loss = []

    for iter_num, (images, targets) in enumerate(dl_train):
        #if iter_num > 10:
        #    break

        #images, targets = images.to(device), targets.to(device)
        #print('targets = {}'.format(targets))
        #assert 1==2
        
        #================================ compute loss ============================
        loss = step((images, targets))

        #============================ print loss =====================================
        epoch_loss.append(loss)

        if iter_num % 10 == 0:
            print('Epoch: {} | Iteration: {} | Running loss: {:1.5f}'.format(
                epoch_num, iter_num, np.mean(epoch_loss)))



# So this experiment computes the confidence for all the points in the X_grid points.
# Then visualize the X_vis as sampled points on the curvature image.
domain = 10
x_lin = np.linspace(-domain+0.5, domain+0.5, 100)
y_lin = np.linspace(-domain, domain, 100)

xx, yy = np.meshgrid(x_lin, y_lin) ## xx.shape: 100x100, yy.shape: 100 x 100

X_grid = np.column_stack([xx.flatten(), yy.flatten()]) # X_grid: 10000 x 2

# X_vis is the generated samples, y_vis is the integer labels (0 or 1) for class membership of each sample.
#X_vis, y_vis = sklearn.datasets.make_moons(n_samples=1000, noise=noise) # X_vis: 1000 x 2, y_vis: 1000
centers = [(-5, -5), (5, 5)]
X_vis, y_vis = make_blobs(n_samples=n_test_samples, centers=centers, shuffle=False, random_state=20)
mask = y_vis.astype(np.bool)


with torch.no_grad():
    output = model(torch.from_numpy(X_grid).float())[1] # output.shape: [10000, 2]
    confidence = output.max(1)[0].numpy() # confidence: 10000,

z = confidence.reshape(xx.shape)

#'''
fig = plt.figure()
plt.contourf(x_lin, y_lin, z, cmap='cividis')

# draw the two moon dataset points
plt.scatter(X_vis[mask,0], X_vis[mask,1])
plt.scatter(X_vis[~mask,0], X_vis[~mask,1])
plt.show()
#fig.tight_layout()
#fig.savefig('temp.jpg')
#plt.close()
#'''


