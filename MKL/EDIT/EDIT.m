train_kernel = csvread('../../EDIT/EDIT_train_kernel.csv');
test_kernel = csvread('../../EDIT/EDIT_test_kernel.csv');
%val_kernel = csvread('../../EDIT/EDIT_val_kernel.csv');
y_tr = csvread('../../Train_labels.csv');
%y_va = csvread('../../Val_labels.csv');
y_te = csvread('../../Test_labels.csv');

train_kernel = [(1:size(y_tr, 1))', train_kernel];
%val_kernel = [(1:size(y_va, 1))', val_kernel];
test_kernel = [(1:size(y_te, 1))', test_kernel];

model = svmtrain(y_tr, train_kernel, '-t 4');
[predClass, acc, decVals] = svmpredict(y_te, test_kernel, model);
%[predClass, acc, decVals] = svmpredict(y_va, val_kernel, model);

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
