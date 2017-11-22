from keras.layers.core import Dense, Activation, TimeDistributedDense
from keras.models import Sequential
from keras.layers import LSTM
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import theano
import theano.tensor as T
import sys

def load_data(file_path, n):
	x = []
	y = []
	for i in range(n):
		d = np.load(file_path + str(i) + '.npz')
		a = d['data']
		b = d['lab']
		x = x + a.tolist()
		y = y + b.tolist()
	z = []
	for i in range(len(y)):
		if y[i] == '1':
			z.append(x[i])
	return z

def load_item_data(file_path, n, item_vec):
	x = []
	y = []
	for i in range(n):
		d = np.load(file_path + str(i) + '.npz')
		a = d['data']
		b = d['lab']
		x = x + a.tolist()
		y = y + b.tolist()
	z = []
	for i in range(len(y)):
		if y[i] == '1':
			z.append(x[i])
	z = np.array(z)
	z = Process_data(z, 30, 64, item_vec)
	return z

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

def main():
	max_len = 30
	s = 15 + 64 + 194
	model = Sequential()
	model.add(LSTM(128, return_sequences=True, input_shape=(max_len, s)))
	model.add(LSTM(64))
	model.add(Dense(2, activation='softmax'))
	model.compile(loss='binary_crossentropy', optimizer='adam')

	a = np.load('../../../Full_data/Word2Vec_vectors.npz')
	item_vec = a['vectors']
	print('Loading buyers .....')
	c_buyers = load_data('../../../Train_data/Train/LSTM_category/LSTM_category', 7)
	i_buyers = load_item_data('../../../Train_data/Train/LSTM_item/LSTM_item', 7, item_vec)
	t_buyers = load_data('../../../Train_data/Train/LSTM_time/LSTM_time', 7)
	buyers = np.concatenate((c_buyers, i_buyers), axis=2)
	buyers = np.concatenate((buyers, t_buyers), axis=2)

	print('Plotting gradients ........')
	model.load_weights('model/best_lstm_concat_buy_w')
	get_output = theano.function([model.layers[0].input], model.layers[-1].output, allow_input_downcast=True)
	fx = theano.function([model.layers[0].input], T.jacobian(model.layers[-1].output.flatten(), model.layers[0].input), allow_input_downcast=True)

	#idx = np.random.permutation(len(buyers))
	#np.savez_compressed('Permuted_buyers_idx', idx=idx)
	a = np.load('../Permuted_buyers_idx.npz')
	idx = a['idx']
	for i in range(10):
		grad = fx([buyers[idx[i]]])
		fig, ax = plt.subplots()
		heatmap = ax.imshow(np.transpose(grad[0][0]), cmap=plt.cm.hot_r, interpolation='nearest')
		plt.axes().set_aspect(0.5)
		#ax.set_title('Feature Excitation in Concatenation of all Views')
		cbar = plt.colorbar(heatmap)
		plt.tight_layout()
		plt.savefig('grad_plots/Gradients_buyers_concat' + str(i) + '.png')
		plt.close()

if __name__ == '__main__':
	main()
#, bbox_inches='tight',dpi=100
# ,
# , extent=[0,30, 0,273], aspect='auto'
