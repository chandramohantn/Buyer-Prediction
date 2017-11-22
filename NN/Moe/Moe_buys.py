from keras.models import Sequential
from keras.layers import Merge, Dense
from keras.layers.core import Dropout, Activation
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn import metrics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load_sess_data(file_path):
	x = []
	y = []
	f = open(file_path + 'Session/Session.dat', 'r')
	while 1:
		line = f.readline()
		line = line[0:-1]
		if len(line) > 0:
			a = []
			items = line.split(',')
			for i in items:
				a.append(float(i))
			x.append(a)
		else:
			f.close()
			break
	x = np.array(x)
	f = open(file_path + 'Labels.dat', 'r')
	while 1:
		line = f.readline()
		line = line[0:-1]
		if len(line) > 0:
			items = line.split(',')
			if items[1] == '0':
				y.append([1, 0])
			else:
				y.append([0, 1])
		else:
			f.close()
			break
	y = np.array(y)
	y = np.reshape(y, (len(y), 2))
	return x, y

def load_seq_data(file_path):
	x = []
	y = []
	f = open(file_path + 'Sequence/Sequence4.dat', 'r')
	while 1:
		line = f.readline()
		line = line[0:-1]
		if len(line) > 0:
			a = []
			items = line.split(',')
			for i in items:
				a.append(float(i))
			x.append(a)
		else:
			f.close()
			break
	x = np.array(x)
	f = open(file_path + 'Labels.dat', 'r')
	while 1:
		line = f.readline()
		line = line[0:-1]
		if len(line) > 0:
			items = line.split(',')
			if items[1] == '0':
				y.append([1, 0])
			else:
				y.append([0, 1])
		else:
			f.close()
			break
	y = np.array(y)
	y = np.reshape(y, (len(y), 2))
	return x, y

def load_temp_data(file_path):
	x = []
	y = []
	f = open(file_path + 'Temporal/Temporal.dat', 'r')
	while 1:
		line = f.readline()
		line = line[0:-1]
		if len(line) > 0:
			a = []
			items = line.split(',')
			for i in items:
				a.append(float(i))
			x.append(a)
		else:
			f.close()
			break
	x = np.array(x)
	f = open(file_path + 'Labels.dat', 'r')
	while 1:
		line = f.readline()
		line = line[0:-1]
		if len(line) > 0:
			items = line.split(',')
			if items[1] == '0':
				y.append([1, 0])
			else:
				y.append([0, 1])
		else:
			f.close()
			break
	y = np.array(y)
	y = np.reshape(y, (len(y), 2))
	return x, y

n = 92
print('Build model 1...')
l1 = [Dense(64, input_dim=n, init='uniform', activation='sigmoid'), Dense(2, activation='softmax')]
for l in l1:
	l.trainable = False
model_1 = Sequential(l1)
model_1.load_weights('../Session/model/best_session_buy_w')

n = 1463
print('Build model 2...')
l2 = [Dense(64, input_dim=n, init='uniform', activation='sigmoid'), Dense(2, activation='softmax')]
for l in l2:
	l.trainable = False
model_2 = Sequential(l2)
model_2.load_weights('../Sequence/model/best_sequence_buy_w')

n = 985
print('Build model 3...')
l3 = [Dense(64, input_dim=n, init='uniform', activation='sigmoid'), Dense(2, activation='softmax')]
for l in l3:
	l.trainable = False
model_3 = Sequential(l3)
model_3.load_weights('../Temporal/model/best_temporal_buy_w')

print('Build model stacker...')
model_c = Sequential()
model_c.add(Merge([model_1, model_2, model_3], mode='concat'))

print('Build model gating...')
model_g = Sequential()
model_g.add(Dense(64, input_dim=2540, init='uniform', activation='sigmoid'))
model_g.add(Dense(6, activation='softmax'))

print('Build model moe...')
model = Sequential()
model.add(Merge([model_c, model_g], mode='mul'))
model.add(Dense(2, activation='softmax'))
model.compile(loss='binary_crossentropy', optimizer='adam')

print("Training...")
score = 10000000000
batch_size = 512
[X_train1, y_train] = load_sess_data('../../../Train_data/Train/')
[X_val1, y_val] = load_sess_data('../../../Train_data/Val/')
[X_train2, y_train] = load_seq_data('../../../Train_data/Train/')
[X_val2, y_val] = load_seq_data('../../../Train_data/Val/')
[X_train3, y_train] = load_temp_data('../../../Train_data/Train/')
[X_val3, y_val] = load_temp_data('../../../Train_data/Val/')
tmp1 = np.concatenate((X_train1, X_train2), axis=1)
tmp1 = np.concatenate((tmp1, X_train3), axis=1)
tmp2 = np.concatenate((X_val1, X_val2), axis=1)
tmp2 = np.concatenate((tmp2, X_val3), axis=1)
t_auc = []
v_auc = []
ls = int(len(X_train1)/batch_size)
class_weight = {0:1,1:1.8}
for e in range(20):
	idx = np.random.permutation(len(X_train1))
	train_loss = 0.0
	for b in range(ls):
		x1 = X_train1[idx[b*batch_size: (b+1)*batch_size], :]
		x2 = X_train2[idx[b*batch_size: (b+1)*batch_size], :]
		x3 = X_train3[idx[b*batch_size: (b+1)*batch_size], :]
		x = tmp1[idx[b*batch_size: (b+1)*batch_size], :]
		y = y_train[idx[b*batch_size: (b+1)*batch_size], :]
		l = model.train_on_batch([x1, x2, x3, x], y, class_weight=class_weight)
		train_loss += l
	print('Epoch: ' + str(e) + ' Loss: ' + str(train_loss))
	model.save_weights('model/moe_buy_w_' + str(e), overwrite=True)
	v_score = model.evaluate([X_val1, X_val2, X_val3, tmp2], y_val)
	t_auc.append(np.log(train_loss * 1.0 / ls))
	v_auc.append(np.log(v_score))
	if score > v_score:
		score = v_score
		model.save_weights('model/best_moe_buy_w', overwrite=True)
print('Training complete .......')

print('Plotting train and val loss...')
plt.plot(t_auc, 'b', label='Train Loss')
plt.plot(v_auc, 'r', label='Validation Loss')
plt.title('Train and Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Log loss')
plt.xlabel('Epochs')
plt.savefig('Moe_train_val_loss.png')
plt.close()

model.load_weights('model/best_moe_buy_w')
print('saving layer weights ...')
layer_wt = [layer for layer in model.layers]
weights = [wt.get_weights() for wt in layer_wt]
np.savez('layer_weights', wts=weights)

model.load_weights('model/best_moe_buy_w')
print("Testing...")
[X_train1, y_train] = load_sess_data('../../../Train_data/Test/')
[X_train2, y_train] = load_seq_data('../../../Train_data/Test/')
[X_train3, y_train] = load_temp_data('../../../Train_data/Test/')
tmp = np.concatenate((X_train1, X_train2), axis=1)
tmp = np.concatenate((tmp, X_train3), axis=1)
prba = model.predict_proba([X_train1, X_train2, X_train3, tmp])
	
f = open('Buys_test_out.csv', 'w');
for i in prba:
	if i[0] > i[1]:
		f.write('0' + '\n')
	else:
		f.write('1' + '\n')
print('Finished .......')

f = open('Buys_test_prob.csv', 'w');
for i in prba:
	f.write(str(i[0]) + ',' + str(i[1]) + '\n')
print('Finished .......')

