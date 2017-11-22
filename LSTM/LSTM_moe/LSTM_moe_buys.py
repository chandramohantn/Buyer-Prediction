from keras.layers.core import Dense, Activation
from keras.models import Model
from keras.layers import LSTM, Input, merge
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn import metrics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load_data(file_path, n):
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
	return x, y

def find_auc(y_real, y_prob):
	precision, recall, thresholds = precision_recall_curve(y_real, y_prob)
	np.savez_compressed('precision_recall', precision=precision, recall=recall, thresholds=thresholds)
	auc = metrics.auc(recall, precision, reorder=True)
	return auc

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
		file_path = test_path + '/LSTM_category/LSTM_category' + str(i) + '.npz'
		d = np.load(file_path)
		x1 = d['data']
		file_path = test_path + '/LSTM_item/LSTM_item' + str(i) + '.npz'
		d = np.load(file_path)
		x2 = d['data']
		y = d['lab']
		x2 = Process_data(x2, max_len, s, item_vec)
		x = np.concatenate((x1, x2), axis=2)
		file_path = test_path + '/LSTM_time/LSTM_time' + str(i) + '.npz'
		d = np.load(file_path)
		x3 = d['data']
		x = np.concatenate((x, x3), axis=2)
		clas = model.predict([x1, x2, x3, x])
		labl.append(clas.tolist())
	return labl

max_len = 30
s = 15
a = np.load('../LSTM_category/layer_weights.npz')
weights = a['wts']
print('Build expert 1...')
input1 = Input(shape=(max_len, s))
m11 = LSTM(128, return_sequences=True, weights=weights[0], trainable=False)(input1)
m12 = LSTM(64, weights=weights[1], trainable=False)(m11)
m13 = Dense(2, activation='softmax', weights=weights[2], trainable=False)(m12)

s = 64
a = np.load('../LSTM_item/layer_weights.npz')
weights = a['wts']
print('Build expert 2...')
input2 = Input(shape=(max_len, s))
m21 = LSTM(128, return_sequences=True, weights=weights[0], trainable=False)(input2)
m22 = LSTM(64, weights=weights[1], trainable=False)(m21)
m23 = Dense(2, activation='softmax', weights=weights[2], trainable=False)(m22)

s = 194
a = np.load('../LSTM_time/layer_weights.npz')
weights = a['wts']
print('Build expert 3...')
input3 = Input(shape=(max_len, s))
m31 = LSTM(128, return_sequences=True, weights=weights[0], trainable=False)(input3)
m32 = LSTM(64, weights=weights[1], trainable=False)(m31)
m33 = Dense(2, activation='softmax', weights=weights[2], trainable=False)(m32)

stacker = merge([m13, m23, m33], mode='concat')

s = 273
print('Build Gating Network ....')
input123 = Input(shape=(max_len, s))
g1 = LSTM(128, return_sequences=True)(input123)
g2 = LSTM(64)(g1)
g3 = Dense(6, activation='softmax')(g2)

g4 = merge([stacker, g3], mode='mul')
g5 = Dense(2, activation='softmax')(g4)

model = Model(input=[input1, input2, input3, input123], output=[g5])
model.compile(loss='binary_crossentropy', optimizer='adam')

a = np.load('../../../Full_data/Word2Vec_vectors.npz')
item_vec = a['vectors']
a = np.load('../../../Full_data/Word2Vec_tokens.npz')
tokens = a['tokens']

print("Training...")
score = 10000000000
batch_size = 512
train_path = '../../../Train_data/Train/LSTM_category/LSTM_category'
val_path = '../../../Train_data/Val/LSTM_category/LSTM_category'
[X_train1, y_train] = load_data(train_path, 7)
[X_val1, y_val] = load_data(val_path, 1)
train_path = '../../../Train_data/Train/LSTM_item/LSTM_item'
val_path = '../../../Train_data/Val/LSTM_item/LSTM_item'
[X_train2, y_train] = load_data(train_path, 7)
X_train2 = Process_data(X_train2, max_len, 64, item_vec)
X_train = np.concatenate((X_train1, X_train2), axis=2)
[X_val2, y_val] = load_data(val_path, 1)
X_val2 = Process_data(X_val2, max_len, 64, item_vec)
X_val = np.concatenate((X_val1, X_val2), axis=2)
train_path = '../../../Train_data/Train/LSTM_time/LSTM_time'
val_path = '../../../Train_data/Val/LSTM_time/LSTM_time'
[X_train3, y_train] = load_data(train_path, 7)
X_train = np.concatenate((X_train, X_train3), axis=2)
[X_val3, y_val] = load_data(val_path, 1)
X_val = np.concatenate((X_val, X_val3), axis=2)

t_auc = []
v_auc = []
ls = int(len(X_train)/batch_size)
class_weight = {0:1,1:3.0}
for e in range(20):
	idx = np.random.permutation(len(X_train1))
	train_loss = 0.0
	for b in range(ls):
		x1 = X_train1[idx[b*batch_size: (b+1)*batch_size]]
		x2 = X_train2[idx[b*batch_size: (b+1)*batch_size]]
		x3 = X_train3[idx[b*batch_size: (b+1)*batch_size]]
		x = X_train[idx[b*batch_size: (b+1)*batch_size]]
		y = y_train[idx[b*batch_size: (b+1)*batch_size]]
		#l = model.train_on_batch([x1, x2, x3, x], y)
		l = model.train_on_batch([x1, x2, x3, x], y, class_weight=class_weight)
		train_loss += l
	print('Epoch: ' + str(e) + ' Loss: ' + str(train_loss))
	model.save_weights('model/lstm_moe_buy_w_' + str(e), overwrite=True)
	v_score = model.evaluate([X_val1, X_val2, X_val3, X_val], y_val)
	t_auc.append(np.log(train_loss * 1.0 / ls))
	v_auc.append(np.log(v_score))
	if score > v_score:
		score = v_score
		model.save_weights('model/best_lstm_moe_buy_w', overwrite=True)
print('Training complete .......')

print('Plotting train and val loss...')
plt.plot(t_auc, 'b', label='Train Loss')
plt.plot(v_auc, 'r', label='Validation Loss')
plt.title('Train and Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Log loss')
plt.xlabel('Epochs')
plt.savefig('LSTM_moe_train_val_loss.png')
plt.close()

model.load_weights('model/best_lstm_moe_buy_w')
print('saving layer weights ...')
layer_wt = [layer for layer in model.layers]
weights = [wt.get_weights() for wt in layer_wt]
np.savez('layer_weights', wts=weights)

print("Testing...")
model.load_weights('model/best_lstm_moe_buy_w')
test_path = '../../../Train_data/Test/'
prba = Test_model(test_path, 2, max_len, 64, item_vec)
	
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

