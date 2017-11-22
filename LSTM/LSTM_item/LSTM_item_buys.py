from keras.layers.core import Dense, Activation, TimeDistributedDense
from keras.models import Sequential
from keras.layers import LSTM
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn import metrics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import theano
import theano.tensor as T

def find_auc(y_real, y_prob):
	precision, recall, thresholds = precision_recall_curve(y_real, y_prob)
	np.savez_compressed('precision_recall', precision=precision, recall=recall, thresholds=thresholds)
	auc = metrics.auc(recall, precision, reorder=True)
	return auc

def load_data(file_path, n, l, s, item_vec):
	x = []
	y = []
	for i in range(n):
		d = np.load(file_path + str(i) + '.npz')
		a = d['data']
		b = d['lab']
		x = x + a.tolist()
		y = y + b.tolist()
	x = np.array(x)
	z = []
	for i in y:
		if i == '0':
			z.append([1, 0])
		else:
			z.append([0, 1])
	y = np.array(z)
	x = Process_data(x, l, s, item_vec)
	return x, y

def Process_data(x, l, s, item_vec):
	a = [0] * s
	z = []
	for i in range(x.shape[0]):
		m = x[i].shape[0]
		y = []
		if (l-m) >= 0:
			for j in range(l-m):
				y.append(a)
			for j in x[i]:
				y.append(item_vec[j])
		else:
			for j in range(l):
				y.append(item_vec[x[i][j]])
		z.append(y)
	z = np.array(z)
	return z

def validate_data(x, y, model):
	p = model.predict_proba(x)
	score = find_auc(y[:, 1], p[:, 1])
	return score

def Test_model(test_path, limits, max_len, s, item_vec):
	labl = []
	for i in range(limits):
		file_path = test_path + str(i) + '.npz'
		d = np.load(file_path)
		x = d['data']
		y = d['lab']
		x = Process_data(x, max_len, s, item_vec)
		clas = model.predict_proba(x)
		labl.append(clas.tolist())
	return labl

max_len = 30
s = 64
print('Build model...')
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(max_len, s)))
model.add(LSTM(64))
model.add(Dense(2, activation='softmax'))
model.compile(loss='binary_crossentropy', optimizer='adam')

a = np.load('../../../Full_data/Word2Vec_vectors.npz')
item_vec = a['vectors']
a = np.load('../../../Full_data/Word2Vec_tokens.npz')
tokens = a['tokens']

print("Training...")
score = 10000000000
batch_size = 512
train_path = '../../../Train_data/Train/LSTM_item/LSTM_item'
val_path = '../../../Train_data/Val/LSTM_item/LSTM_item'
[X_train, y_train] = load_data(train_path, 7, max_len, s, item_vec)
[X_val, y_val] = load_data(val_path, 1, max_len, s, item_vec)
t_auc = []
v_auc = []
ls = int(len(X_train)/batch_size)
class_weight = {0:1,1:3.0}
for e in range(20):
	idx = np.random.permutation(len(X_train))
	train_loss = 0.0
	for b in range(ls):
		x = X_train[idx[b*batch_size: (b+1)*batch_size]]
		y = y_train[idx[b*batch_size: (b+1)*batch_size]]
		l = model.train_on_batch(x, y, class_weight=class_weight)
		train_loss += l
	print('Epoch: ' + str(e) + ' Loss: ' + str(train_loss))
	model.save_weights('model/lstm_item_buy_w_' + str(e), overwrite=True)
	v_score = model.evaluate(X_val, y_val)
	t_auc.append(np.log(train_loss * 1.0 / ls))
	v_auc.append(np.log(v_score))
	if score > v_score:
		score = v_score
		model.save_weights('model/best_lstm_item_buy_w', overwrite=True)
print('Training complete .......')

print('Plotting train and val loss...')
plt.plot(t_auc, 'b', label='Train Loss')
plt.plot(v_auc, 'r', label='Validation Loss')
plt.title('Train and Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Log loss')
plt.xlabel('Epochs')
plt.savefig('LSTM_item_train_val_loss.png')
plt.close()

model.load_weights('model/best_lstm_item_buy_w')
print('saving layer weights ...')
layer_wt = [layer for layer in model.layers]
weights = [wt.get_weights() for wt in layer_wt]
np.savez('layer_weights', wts=weights)

print("Testing...")
model.load_weights('model/best_lstm_item_buy_w')
test_path = '../../../Train_data/Test/LSTM_item/LSTM_item'
prba = Test_model(test_path, 2, max_len, s, item_vec)
	
f = open('Buys_test_out.csv', 'w');
for i in prba:
	for j in i:
		if j[0] > j[1]:
			f.write('0' + '\n')
		else:
			f.write('1' + '\n')
print('Finished .......')
f.close()

f = open('Buys_test_prob.csv', 'w');
for i in prba:
	for j in i:
		f.write(str(j[0]) + ',' + str(j[1]) + '\n')
print('Finished .......')
f.close()
