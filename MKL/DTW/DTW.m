train_kernel = csvread('../../DTW/DTW_train_kernel.csv');
test_kernel = csvread('../../DTW/DTW_test_kernel.csv');
val_kernel = csvread('../../DTW/DTW_val_kernel.csv');
y_tr = csvread('../../Train_labels.csv');
y_va = csvread('../../Val_labels.csv');
y_te = csvread('../../Test_labels.csv');

train_kernel = [(1:size(y_tr, 1))', train_kernel];
val_kernel = [(1:size(y_va, 1))', val_kernel];
test_kernel = [(1:size(y_te, 1))', test_kernel];

model = svmtrain(y_tr, train_kernel, '-t 4');
[predClass, acc, decVals] = svmpredict(y_va, val_kernel, model);

%{
tp = sum(y_va == 1 & dec >= 0);
tp_fp = sum(dec >= 0);
ret = tp / tp_fp;

function ret = recall(dec, label)
tp = sum(label == 1 & dec >= 0);
tp_fn = sum(label == 1);
if tp_fn == 0;
  disp(sprintf('warning: No postive true label.'));
  ret = 0;
else
  ret = tp / tp_fn;
end
disp(sprintf('Recall = %g%% (%d/%d)', 100.0 * ret, tp, tp_fn));

function ret = fscore1(dec, label)
tp = sum(label == 1 & dec >= 0);
tp_fp = sum(dec >= 0);
tp_fn = sum(label == 1);
if tp_fp == 0;
  disp(sprintf('warning: No positive predict label.'));
  precision = 0;
else
  precision = tp / tp_fp;
end
if tp_fn == 0;
  disp(sprintf('warning: No postive true label.'));
  recall = 0;
else
  recall = tp / tp_fn;
end
if precision + recall == 0;
  disp(sprintf('warning: precision + recall = 0.'));
  ret = 0;
else
  ret = 2 * precision * recall / (precision + recall);
end
%disp(sprintf('F-score = %g', ret));
%}
