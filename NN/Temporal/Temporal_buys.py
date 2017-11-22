from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn import metrics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def find_auc(y_real, y_prob):
	precision, recall, thresholds = precision_recall_curve(y_real, y_prob)
	np.savez_compressed('precision_recall', precision=precision, recall=recall, thresholds=thresholds)
	auc = metrics.auc(recall, precision, reorder=True)
	return auc

def load_data(file_path):
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

def validate_data(x, y, model):
	prob = model.predict_proba(x)
	score = find_auc(y[:, 1], prob[:, 1])
	return score

n = 985
print('Build model...')
model = Sequential()
model.add(Dense(64, input_dim=n, init='uniform', activation='sigmoid'))
model.add(Dense(2, activation='softmax'))
model.compile(loss='binary_crossentropy', optimizer='adam')

print("Training...")
score = 10000000000
batch_size = 512
[X_train, y_train] = load_data('../../../Train_data/Train/')
print(X_train.shape)
[X_val, y_val] = load_data('../../../Train_data/Val/')
t_auc = []
v_auc = []
ls = int(len(X_train)/batch_size)
class_weight = {0:1,1:1.8}
for e in range(20):
	idx = np.random.permutation(len(X_train))
	train_loss = 0.0
	for b in range(ls):
		x = X_train[idx[b*batch_size: (b+1)*batch_size], :]
		y = y_train[idx[b*batch_size: (b+1)*batch_size], :]
		l = model.train_on_batch(x, y, class_weight=class_weight)
		train_loss += l
	print('Epoch: ' + str(e) + ' Loss: ' + str(train_loss))
	v_score = model.evaluate(X_val, y_val)
	t_auc.append(np.log(train_loss * 1.0 / ls))
	v_auc.append(np.log(v_score))
	model.save_weights('model/temporal_buy_w_' + str(e), overwrite=True)
	if score > v_score:
		score = v_score
		model.save_weights('model/best_temporal_buy_w', overwrite=True)
print('Training complete .......')

print('Plotting train and val loss...')
plt.plot(t_auc, 'b', label='Train Loss')
plt.plot(v_auc, 'r', label='Validation Loss')
plt.title('Train and Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Log loss')
plt.xlabel('Epochs')
plt.savefig('Temporal_train_val_loss.png')
plt.close()

model.load_weights('model/best_temporal_buy_w')
print('saving layer weights ...')
layer_wt = [layer for layer in model.layers]
weights = [wt.get_weights() for wt in layer_wt]
np.savez('layer_weights', wts=weights)

model.load_weights('model/best_temporal_buy_w')
print("Testing...")
[X_train, y_train] = load_data('../../../Train_data/Test/')
prba = model.predict_proba(X_train)
	
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

