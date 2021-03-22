import numpy as np 
import matplotlib.pyplot as plt 
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import make_blobs
import random
import matplotlib.cm as cmx
import matplotlib.colors as colors
import sklearn.datasets
import torch.utils.data

np.random.seed(20)
torch.manual_seed(20)
torch.cuda.manual_seed_all(20)

exp_id = 30
flag_train = True


def vis_points(X, Y, name='temp', cats=[0,1,2]):
	# define the colormap
	cmap = plt.get_cmap('viridis')
	cNorm = colors.Normalize(vmin=cats[0], vmax=cats[-1]+1)
	scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
	for c in cats[::-1]:
		x = X[Y==c]
		y = Y[Y==c]
		plt.scatter(x[:, 0], x[:, 1], s=20, color=scalarMap.to_rgba(c))

	plt.grid()
	xmin, xmax = np.min(X[:, 0])-2, np.max(X[:, 0])+2
	ymin, ymax = np.min(X[:, 1])-2, np.max(X[:, 1])+2
	plt.axis([xmin, xmax, ymin, ymax], 'equal')
	plt.gca().set_aspect("equal")
	#plt.show()
	plt.savefig('result_imgs/{}.jpg'.format(name))
	plt.close()

def vis_conf(X, Y, name='temp', cats=[0,1,2]):
	plt.scatter(X[:, 0], X[:, 1], c=Y, s=20, cmap='viridis')
	plt.grid()
	xmin, xmax = np.min(X[:, 0])-2, np.max(X[:, 0])+2
	ymin, ymax = np.min(X[:, 1])-2, np.max(X[:, 1])+2
	plt.axis([xmin, xmax, ymin, ymax], 'equal')
	plt.gca().set_aspect("equal")
	plt.title('certainty')
	#plt.show()
	plt.savefig('result_imgs/{}.jpg'.format(name))
	plt.close()

class duq(nn.Module):
	def __init__(self, input_size=2, features=20, num_classes=2):
		super(duq, self).__init__()

		self.gamma = 0.99
		self.sigma = 0.3
		
		embedding_size = 10
		
		self.fc1 = nn.Linear(input_size, features)
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

		return y_pred, z

	def update_embeddings(self, x, y):
		z = self.embed(x)
		# normalizing value per class, assumes y is one_hot encoded
		# implement Eq (4)
		self.N = torch.max(self.gamma * self.N + (1 - self.gamma) * y.sum(0), torch.ones_like(self.N))
		# compute sum of embeddings on class by class basis
		features_sum = torch.einsum('ijk,ik->jk', (z, y))
		# implement Eq (5)
		self.m = self.gamma * self.m + (1 - self.gamma) * features_sum


n_train_samples = 20000
centers = [(0, 0), (10, 10)]
train_X, train_Y = make_blobs(n_samples=n_train_samples, centers=centers, shuffle=False, random_state=40)
y_train_targets = np.zeros((train_Y.shape[0], 2))
y_train_targets[train_Y==0, 0] = 1
y_train_targets[train_Y==1, 1] = 1
ds_train = torch.utils.data.TensorDataset(torch.from_numpy(train_X).float(), torch.from_numpy(y_train_targets).float())
dl_train = torch.utils.data.DataLoader(ds_train, batch_size=64, shuffle=True, drop_last=True)

n_test_samples = 10000
centers = [(0, 0), (10, 10)]
test_X, test_Y = make_blobs(n_samples=n_test_samples, centers=centers, shuffle=False, random_state=20)

#'''
# add points to cover the whole input space
x_lin = np.linspace(-20, 20, 100)
y_lin = np.linspace(-20, 20, 100)
xx, yy = np.meshgrid(x_lin, y_lin) ## xx.shape: 100x100, yy.shape: 100 x 100
X_grid = np.column_stack([xx.flatten(), yy.flatten()]) # X_grid: 10000 x 2
Y_grid = np.ones(X_grid.shape[0], dtype=int) * 2 # outlier has class to be 2
idx_list = [i for i in range(Y_grid.shape[0])]
list_idxs = random.choices(idx_list, k=int(Y_grid.shape[0] * 0.1))
X_grid = X_grid[list_idxs]
Y_grid = Y_grid[list_idxs]
n_test_samples += Y_grid.shape[0]
test_X = np.concatenate((test_X, X_grid), axis=0)
test_Y = np.concatenate((test_Y, Y_grid))
#'''


#vis_points(train_X, train_Y)

device = torch.device('cuda:0')
model = duq(num_classes=2).to(device)
if not flag_train:
	model.load_state_dict(torch.load('trained_model/duq_1_class.pth'))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)


if flag_train:
	num_epochs = 10
	model.train()
	for epoch_num in range(num_epochs):
		print('epoch_num = {}'.format(epoch_num))
		epoch_loss = []

		for iter_num, (x, y) in enumerate(dl_train):
			x = torch.tensor(x, dtype=torch.float).to(device)
			y = torch.tensor(y, dtype=torch.float).to(device)

			model.train()
			optimizer.zero_grad()

			x.requires_grad_(True)
			
			y_pred, _ = model(x)
			
			loss1 =  F.binary_cross_entropy(y_pred, y)
			loss = loss1
			
			loss.backward()
			optimizer.step()

			with torch.no_grad():
				#assert 1==2
				model.update_embeddings(x, y)

			epoch_loss.append(loss.item())

			if iter_num % 100 == 0:
				print('iter = {}, Loss: {:.4f}'.format(iter_num, sum(epoch_loss) / len(epoch_loss)))

print('-----------------start testing -----------------------------------')

#-------------------------------------- eval on ---------------------------------
flag_eval = True
model.eval()
pred_Y = np.ones((test_Y.shape)) * -1
test_ft = np.zeros((n_test_samples))
batch_size = 10
with torch.no_grad():
	for iter_num in range(int(n_test_samples/batch_size)):
		x = test_X[iter_num*batch_size:(iter_num+1)*batch_size]
		x = torch.tensor(x, dtype=torch.float).to(device)
		y = test_Y[iter_num*batch_size:(iter_num+1)*batch_size]
		y = torch.tensor(y, dtype=torch.long).to(device)
		out, z = model.forward(x)
		'''
		loss = criterion(out, y)
		if iter_num % 100 == 0:
			print('iter = {}, Loss: {:.4f}'.format(iter_num, loss.item()))
		'''
		out = out.cpu().numpy()
		#print('pred.shape = {}'.format(pred.shape))
		out = out[:, 0]
		pred = np.zeros(out.shape[0])
		pred[out < 0.9] = 1
		pred_Y[iter_num*batch_size:(iter_num+1)*batch_size] = pred

		confidence = out
		test_ft[iter_num*batch_size:(iter_num+1)*batch_size] = confidence

		#assert 1==2

vis_points(test_X, pred_Y, 'exp{}, eval={}, X_pred_Y'.format(exp_id, flag_eval))
vis_points(test_X, test_Y, 'exp{}, eval={}, X_gt_Y'.format(exp_id, flag_eval))
vis_conf(test_X, test_ft, 'exp{}, eval={}, certainty'.format(exp_id, flag_eval))

