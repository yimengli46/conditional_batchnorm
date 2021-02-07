import numpy as np 
import matplotlib.pyplot as plt 
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import make_blobs
import random

exp_id = 14

FT_SIZE = 16
NUM_CLASSES = 3

def vis_points(X, Y, name='temp'):
	plt.scatter(X[:, 0], X[:, 1], c=Y, s=20, cmap='viridis')
	plt.grid()
	xmin, xmax = np.min(X[:, 0])-2, np.max(X[:, 0])+2
	ymin, ymax = np.min(X[:, 1])-2, np.max(X[:, 1])+2
	plt.axis([xmin, xmax, ymin, ymax], 'equal')
	plt.gca().set_aspect("equal")
	#plt.show()
	plt.savefig('{}.jpg'.format(name))
	plt.close()

class perception(nn.Module):
	def __init__(self, input_size=2, hidden_size=16, ft_size=FT_SIZE, output_size=NUM_CLASSES):
		super(perception, self).__init__()
		self.linear1 = nn.Linear(input_size, hidden_size)
		self.linear2 = nn.Linear(hidden_size, ft_size)
		self.linear3 = nn.Linear(ft_size, output_size)
		self.bn1 = nn.BatchNorm1d(hidden_size)
		self.bn2 = nn.BatchNorm1d(ft_size)
		#self.bn1_2 = nn.BatchNorm1d(hidden_size)
		#self.bn2_2 = nn.BatchNorm1d(ft_size)

	def forward_class_0(self, x):
		x = F.relu(self.bn1(self.linear1(x)))
		x = F.relu(self.bn2(self.linear2(x)))
		out = self.linear3(x)
		return out, x


def set_bn_eval(m):
	m.bn1.eval()
	m.bn2.eval()

def set_bn_train(m):
	m.bn1.train()
	m.bn2.train()


n_train_samples = 30000
centers = [(-5, -5), (5, 5), (7.5, -7.5)]
train_X, train_Y = make_blobs(n_samples=n_train_samples, centers=centers, shuffle=False, random_state=40)
n_test_samples = 10000
centers = [(-5, -5), (5, 5), (7.5, -7.5)]#, (-7.5, 7.5)]
test_X, test_Y = make_blobs(n_samples=n_test_samples, centers=centers, shuffle=False, random_state=40)

#vis_points(train_X, train_Y)

device = torch.device('cuda:0')
model = perception().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.00001, momentum=0.9, weight_decay=1e-4)
criterion = torch.nn.CrossEntropyLoss()

num_epochs = 8
batch_size = 64
idx_list = [x for x in range(n_train_samples)]
model.train()
for epoch_num in range(num_epochs):
	print('epoch_num = {}'.format(epoch_num))
	epoch_loss = []

	for iter_num in range(n_train_samples//batch_size):
		list_idxs = random.choices(idx_list, k=batch_size)
		#print('list_idxs = {}'.format(list_idxs))
		x, y = train_X[list_idxs], train_Y[list_idxs]
		x = torch.tensor(x, dtype=torch.float).to(device)
		y = torch.tensor(y, dtype=torch.long).to(device)
		#print('y = {}'.format(y))

		'''
		x_0 = x[y==0]
		y_0 = y[y==0]
		#if x_0.shape[0] > 0:
		#set_bn_train(model)
		out_0, _ = model.forward_class_0(x_0)
		loss_0 = criterion(out_0, y_0)
		#loss_0.backward()
		#optimizer.step()
		#assert 1==2
		x_1 = x[y>0]
		y_1 = y[y>0]
		#set_bn_eval(model)
		out_1, _ = model.forward_class_0(x_1)
		loss_1 = criterion(out_1, y_1)
		loss_0.backward()
		loss_1.backward()
		optimizer.step()
		#assert 1==2
		loss = loss_0 + loss_1
		'''

		out, _ = model.forward_class_0(x)
		loss = criterion(out, y)
		loss.backward()
		optimizer.step()

		epoch_loss.append(loss.item())
		if iter_num % 100 == 0:
			print('iter = {}, Loss: {:.4f}'.format(iter_num, sum(epoch_loss) / len(epoch_loss)))
			#print('out = {}'.format(out))
		#assert 1==2
print('-----------------start testing -----------------------------------')

#--------------------------------- eval off ---------------------------
flag_eval = False
model.train()
pred_Y = np.ones((test_Y.shape)) * -1
test_ft = np.zeros((n_test_samples, FT_SIZE))
batch_size = 10
with torch.no_grad():
	for iter_num in range(int(n_test_samples/batch_size)):
		x = test_X[iter_num*batch_size:(iter_num+1)*batch_size]
		x = torch.tensor(x, dtype=torch.float).to(device)
		y = test_Y[iter_num*batch_size:(iter_num+1)*batch_size]
		y = torch.tensor(y, dtype=torch.long).to(device)
		out, z = model.forward_class_0(x)
		'''
		loss = criterion(out, y)
		if iter_num % 100 == 0:
			print('iter = {}, Loss: {:.4f}'.format(iter_num, loss.item()))
		'''
		pred = out.cpu().numpy()
		pred = np.argmax(pred, axis=1)
		pred_Y[iter_num*batch_size:(iter_num+1)*batch_size] = pred

		z = z.cpu().numpy()
		test_ft[iter_num*batch_size:(iter_num+1)*batch_size] = z

		#assert 1==2

vis_points(test_X, pred_Y, 'exp{}, eval={}, X_pred_Y'.format(exp_id, flag_eval))
vis_points(test_X, test_Y, 'exp{}, eval={}, X_gt_Y'.format(exp_id, flag_eval))
#vis_points(test_ft, pred_Y, 'exp{}, eval={}, ft_pred_Y'.format(exp_id, flag_eval))
#vis_points(test_ft, test_Y, 'exp{}, eval={}, ft_gt_Y'.format(exp_id, flag_eval))

#-------------------------------------- eval on ---------------------------------
flag_eval = True
model.eval()
pred_Y = np.ones((test_Y.shape)) * -1
test_ft = np.zeros((n_test_samples, FT_SIZE))
batch_size = 10
with torch.no_grad():
	for iter_num in range(int(n_test_samples/batch_size)):
		x = test_X[iter_num*batch_size:(iter_num+1)*batch_size]
		x = torch.tensor(x, dtype=torch.float).to(device)
		y = test_Y[iter_num*batch_size:(iter_num+1)*batch_size]
		y = torch.tensor(y, dtype=torch.long).to(device)
		out, z = model.forward_class_0(x)
		'''
		loss = criterion(out, y)
		if iter_num % 100 == 0:
			print('iter = {}, Loss: {:.4f}'.format(iter_num, loss.item()))
		'''
		pred = out.cpu().numpy()
		pred = np.argmax(pred, axis=1)
		pred_Y[iter_num*batch_size:(iter_num+1)*batch_size] = pred

		z = z.cpu().numpy()
		test_ft[iter_num*batch_size:(iter_num+1)*batch_size] = z

		#assert 1==2

vis_points(test_X, pred_Y, 'exp{}, eval={}, X_pred_Y'.format(exp_id, flag_eval))
vis_points(test_X, test_Y, 'exp{}, eval={}, X_gt_Y'.format(exp_id, flag_eval))
#vis_points(test_ft, pred_Y, 'exp{}, eval={}, ft_pred_Y'.format(exp_id, flag_eval))
#vis_points(test_ft, test_Y, 'exp{}, eval={}, ft_gt_Y'.format(exp_id, flag_eval))
