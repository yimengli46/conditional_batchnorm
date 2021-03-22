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
from scipy.stats import entropy
from scipy.special import softmax

np.random.seed(20)
torch.manual_seed(20)
torch.cuda.manual_seed_all(20)

exp_id = 31
flag_train = True
num_forward_pass = 10
num_classes = 2

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

class perception(nn.Module):
	def __init__(self, input_size=2, hidden_size=20, ft_size=20, num_classes=2):
		super(perception, self).__init__()
		self.linear1 = nn.Linear(input_size, hidden_size)
		self.linear2 = nn.Linear(hidden_size, ft_size)
		self.linear3 = nn.Linear(ft_size, num_classes)
		self.bn1 = nn.BatchNorm1d(hidden_size)
		self.bn2 = nn.BatchNorm1d(ft_size)

	def forward(self, x):
		x = F.relu(self.bn1(self.linear1(x)))
		x = F.dropout(x, p=0.2, training=True)
		x = F.relu(self.bn2(self.linear2(x)))
		x = F.dropout(x, p=0.2, training=True)
		out = self.linear3(x)
		return out


n_train_samples = 20000
centers = [(0, 0), (10, 10)]
train_X, train_Y = make_blobs(n_samples=n_train_samples, centers=centers, shuffle=False, random_state=40)
ds_train = torch.utils.data.TensorDataset(torch.from_numpy(train_X).float(), torch.from_numpy(train_Y).float())
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
model = perception(num_classes=num_classes).to(device)
if not flag_train:
	model.load_state_dict(torch.load('trained_model/duq_1_class.pth'))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
criterion = torch.nn.CrossEntropyLoss()

if flag_train:
	num_epochs = 2
	model.train()
	for epoch_num in range(num_epochs):
		print('epoch_num = {}'.format(epoch_num))
		epoch_loss = []

		for iter_num, (x, y) in enumerate(dl_train):
			x = torch.tensor(x, dtype=torch.float).to(device)
			y = torch.tensor(y, dtype=torch.float).to(device).long()

			out = model(x)
			loss = criterion(out, y)
			
			loss.backward()
			optimizer.step()

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

		N = y.shape[0]

		pass_logits = torch.zeros((num_forward_pass, N, num_classes))
		for p in range(num_forward_pass):
			out = model(x)
			pass_logits[p] = out

		logits = torch.mean(pass_logits, dim=0) # N x num_classes

		out = logits.cpu().numpy()
		#print('pred.shape = {}'.format(pred.shape))
		#out = out[:, 0]
		#pred = np.zeros(out.shape[0])
		#pred[out < 0.9] = 1
		pred = np.argmax(out, axis=1)
		pred_Y[iter_num*batch_size:(iter_num+1)*batch_size] = pred

		#confidence = out
		confidence = entropy(softmax(out, axis=1), axis=1, base=2)
		test_ft[iter_num*batch_size:(iter_num+1)*batch_size] = confidence

		#assert 1==2

vis_points(test_X, pred_Y, 'exp{}, eval={}, X_pred_Y'.format(exp_id, flag_eval))
vis_points(test_X, test_Y, 'exp{}, eval={}, X_gt_Y'.format(exp_id, flag_eval))
vis_conf(test_X, test_ft, 'exp{}, eval={}, certainty'.format(exp_id, flag_eval))

