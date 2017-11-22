x_tr = csvread('../../RBF/Train.csv');
x_te = csvread('../../RBF/Test.csv');
%x_va = csvread('../../RBF/Val.csv');

sigma = 2e-3;
rbfKernel = @(X, Y) exp(-sigma .* pdist2(X, Y, 'euclidean').^2);

%train_kernel = csvread('../../DTW/DTW_train_kernel.csv');
%test_kernel = csvread('../../DTW/DTW_test_kernel.csv');
%val_kernel = csvread('../../DTW/DTW_val_kernel.csv');
y_tr = csvread('../../Train_labels.csv');
%y_va = csvread('../../Val_labels.csv');
y_te = csvread('../../Test_labels.csv');

train_kernel = [(1:size(y_tr, 1))', rbfKernel(x_tr, x_tr)];
%val_kernel = [(1:size(y_va, 1))', rbfKernel(x_va, x_tr)];
test_kernel = [(1:size(y_te, 1))', rbfKernel(x_te, x_tr)];

model = svmtrain(y_tr, train_kernel, '-t 4');
[predClass, acc, decVals] = svmpredict(y_te, test_kernel, model);

tp = sum(y_te == 1 & predClass == 1);
tp_fp = sum(predClass == 1);
tp_fn = sum(y_te == 1);
prec = tp / tp_fp;
recl = tp / tp_fn;
if prec + recl > 0
  fscore1 = 2 * prec * recl / (prec + recl);
end
disp(fscore1);

tp = sum(y_te == 0 & predClass == 0);
tp_fp = sum(predClass == 0);
tp_fn = sum(y_te == 0);
prec = tp / tp_fp;
recl = tp / tp_fn;
if prec + recl > 0
  fscore0 = 2 * prec * recl / (prec + recl);
end
disp(fscore0);
