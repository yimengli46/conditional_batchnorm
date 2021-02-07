import numpy as np 
import matplotlib.pyplot as plt 
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import make_blobs
import random
import matplotlib.cm as cmx
import matplotlib.colors as colors

exp_id = 16

FT_SIZE = 4

def vis_points(X, Y, name='temp', cats=[0,1,2]):
	# define the colormap
	cmap = plt.get_cmap('viridis')
	cNorm = colors.Normalize(vmin=cats[0], vmax=cats[-1]+1)
	scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
	for c in cats[::-1]:
		x = X[Y==c]
		y = Y[Y==c]
		'''
		# downsample outlier points
		if c == 2:
			idx_list = [i for i in range(y.shape[0])]
			list_idxs = random.choices(idx_list, k=int(y.shape[0] * 0.1))
			x = x[list_idxs]
			y = y[list_idxs]
		'''
		plt.scatter(x[:, 0], x[:, 1], s=5, color=scalarMap.to_rgba(c))


	plt.grid()
	xmin, xmax = np.min(X[:, 0])-2, np.max(X[:, 0])+2
	ymin, ymax = np.min(X[:, 1])-2, np.max(X[:, 1])+2
	plt.axis([xmin, xmax, ymin, ymax], 'equal')
	plt.gca().set_aspect("equal")
	#plt.show()
	plt.savefig('result_imgs/{}.jpg'.format(name))
	plt.close()

class perception(nn.Module):
	def __init__(self, input_size=2, hidden_size=8, ft_size=FT_SIZE, output_size=2):
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

	def forward_class_1(self, x):
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


n_train_samples = 50000
centers = [(-5, -5), (5, 5)]
train_X, train_Y = make_blobs(n_samples=n_train_samples, centers=centers, shuffle=False, random_state=40)
n_test_samples = 10000
centers = [(-5, -5), (5, 5)]
test_X, test_Y = make_blobs(n_samples=n_test_samples, centers=centers, shuffle=False, random_state=40)
# add points to cover the whole input space
x_lin = np.linspace(-10, 10, 100)
y_lin = np.linspace(-10, 10, 100)
xx, yy = np.meshgrid(x_lin, y_lin) ## xx.shape: 100x100, yy.shape: 100 x 100
X_grid = np.column_stack([xx.flatten(), yy.flatten()]) # X_grid: 10000 x 2
Y_grid = np.ones(X_grid.shape[0], dtype=int) * 2 # outlier has class to be 2
'''
idx_list = [i for i in range(Y_grid.shape[0])]
list_idxs = random.choices(idx_list, k=int(Y_grid.shape[0] * 0.1))
X_grid = X_grid[list_idxs]
Y_grid = Y_grid[list_idxs]
n_test_samples += Y_grid.shape[0]
test_X = np.concatenate((test_X, X_grid), axis=0)
test_Y = np.concatenate((test_Y, Y_grid))
'''

#vis_points(train_X, train_Y)

device = torch.device('cuda:0')
model = perception().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
criterion = torch.nn.CrossEntropyLoss()

num_epochs = 5
batch_size = 32
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

		x_0 = x[y==0]
		y_0 = y[y==0]
		set_bn_train(model)
		out_0, _ = model.forward_class_0(x_0)
		loss_0 = criterion(out_0, y_0)
		loss_0.backward()
		optimizer.step()

		x_1 = x[y==1]
		y_1 = y[y==1]
		set_bn_eval(model)
		out_1, _ = model.forward_class_1(x_1)
		loss_1 = criterion(out_1, y_1)
		loss_1.backward()
		optimizer.step()

		loss = loss_0 + loss_1

		epoch_loss.append(loss.item())
		#if epoch_num == 1:
		#	assert 1==2
		#loss.backward()
		#optimizer.step()
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
