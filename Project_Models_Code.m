% Machine Learning and Deep Learning Project - Classification problem


clc;
clear;

% --------------------------------------------------------------------------
% Start Preprocessing 

load ('BankData.mat');

% Creating Dummy variables for the Categorical variables
BankData.djob            = dummyvar(BankData.job);
BankData.dmarital        = dummyvar(BankData.marital);
BankData.deducation      = dummyvar(BankData.education);
BankData.ddefault        = dummyvar(BankData.default);
BankData.dhousing        = dummyvar(BankData.housing);
BankData.dloan           = dummyvar(BankData.loan);
BankData.dcontact        = dummyvar(BankData.contact);
BankData.dmonth          = dummyvar(BankData.month);
BankData.dday_of_week    = dummyvar(BankData.day_of_week);
BankData.dpoutcome       = dummyvar(BankData.poutcome);

% Creating Reference group of above dummy variables
% job = admin.	blue-collar	  entrepreneur	 housemaid	  management	retired	  self-employed	  services	student	 technician	  unemployed	unknown
BankData.djob_refer = BankData.djob(:, 2:12);               % Reference group = admin 

% marital = divorced	married	  single	 unknown
BankData.dmarital_refer = BankData.dmarital(:, 2:4);        % Reference group = divorced 

% education = basic.4y	  basic.6y	  basic.9y	  high.school	illiterate	professional.course	  university.degree	  unknown
BankData.deducation_refer = BankData.deducation(:, 2:8);    % Reference group = basic.4y

% default = no	 unknown    yes
BankData.ddefault_refer = BankData.ddefault(:, 2:3);        % Reference group = no 

% housing = no	 unknown    yes
BankData.dhousing_refer = BankData.dhousing(:, 2:3);        % Reference group = no 

% loan = no	 unknown    yes
BankData.dloan_refer = BankData.dloan(:, 2:3);              % Reference group = no 

% contact = cellular	 telephone
BankData.dcontact_refer = BankData.dcontact(:, 2);          % Reference group = cellular

% month = mar	apr     may	   jun	jul	  aug	sep	  oct	nov	  dec
BankData.dmonth_refer = BankData.dmonth(:, 2:10);            % Reference group = mar 

% day_of_week = mon	tue	wed	thu	fri
BankData.dday_of_week_refer = BankData.dday_of_week(:, 2:5);  % Reference group = mon 

% poutcome = failure     nonexistent     success
BankData.dpoutcome_refer = BankData.dpoutcome(:, 2:3);      % Reference group = failure

% Standardization of numerical predictors
BankData.age_zs          = zscore(BankData.age);
BankData.duration_zs     = zscore(BankData.duration);
BankData.campaign_zs     = zscore(BankData.campaign);
BankData.pdays_zs        = zscore(BankData.pdays);
BankData.previous_zs     = zscore(BankData.previous);
BankData.empvarrate_zs   = zscore(BankData.empvarrate);
BankData.conspriceidx_zs = zscore(BankData.conspriceidx);
BankData.consconfidx_zs  = zscore(BankData.consconfidx);
BankData.euribor3m_zs    = zscore(BankData.euribor3m);
BankData.nremployed_zs   = zscore(BankData.nremployed);

% Standardization of response variable
BankData.dY = dummyvar(BankData.y);
BankData.Y  = BankData.dY(:,2); %%% no=0 and yes=1

%%%% Data split into Train and Test
[train_indicies,valid_indicies,test_indicies] = dividerand(41188,0.5,0,0.5);
traindata = BankData(train_indicies,:);
testdata = BankData(test_indicies,:);

% All 53 predictors stored in X
BankXY = [BankData.age_zs  BankData.djob_refer BankData.dmarital_refer BankData.deducation_refer BankData.ddefault_refer BankData.dhousing_refer ...
          BankData.dloan_refer BankData.dcontact_refer BankData.dmonth_refer BankData.dday_of_week_refer BankData.duration_zs ...
          BankData.campaign_zs BankData.pdays_zs BankData.previous_zs  BankData.dpoutcome_refer BankData.empvarrate_zs ...
          BankData.conspriceidx_zs BankData.consconfidx_zs BankData.euribor3m_zs BankData.nremployed_zs BankData.Y];


%%%% Create Matricies With the Previuosly Split TRAIN AND TEST data %%%%%%%%%%%%% SPLIT Y ALSO
trainX = [traindata.age_zs  traindata.djob_refer traindata.dmarital_refer traindata.deducation_refer traindata.ddefault_refer traindata.dhousing_refer ...
    traindata.dloan_refer traindata.dcontact_refer traindata.dmonth_refer traindata.dday_of_week_refer traindata.duration_zs ...
    traindata.campaign_zs traindata.pdays_zs traindata.previous_zs  traindata.dpoutcome_refer traindata.empvarrate_zs ...
    traindata.conspriceidx_zs traindata.consconfidx_zs traindata.euribor3m_zs traindata.nremployed_zs];
trainY = traindata.Y;

testX = [testdata.age_zs  testdata.djob_refer testdata.dmarital_refer testdata.deducation_refer testdata.ddefault_refer testdata.dhousing_refer ...
    testdata.dloan_refer testdata.dcontact_refer testdata.dmonth_refer testdata.dday_of_week_refer testdata.duration_zs ...
    testdata.campaign_zs testdata.pdays_zs testdata.previous_zs  testdata.dpoutcome_refer testdata.empvarrate_zs ...
    testdata.conspriceidx_zs testdata.consconfidx_zs testdata.euribor3m_zs testdata.nremployed_zs];
testY = testdata.Y;

% End Preprocessing 
% --------------------------------------------------------------------------








% --------------------------------------------------------------------------
% Begin TECHNIQUE 1: LOGISTIC REGRESSION
% 

fprintf('Starting technique 1 at: %s\n', datestr(now,'HH:MM:SS.FFF'));

% Logistic Regresion model with all predictors
fprintf('\n\n Logistic Regresion model with all predictors');
model_logistic = fitglm(trainX, traindata.Y, 'distr', 'binomial', 'link', 'logit')      %%% XTRAIN  YTRAIN
p = model_logistic.Fitted.Response;        % probability
Z = model_logistic.Fitted.LinearPredictor; % Z = qTX
figure, gscatter(Z, p, traindata.Y, 'br'); grid on
zmin = min(Z); zmax = max(Z);
ylim([-0.05 1.05]), xlabel('\bf Z'),
ylabel('\bf P'),
title(num2str(model_logistic.Coefficients.Estimate'));


%%%%%% CHECKING FOR OVERFITTING BY COMPARING ROC OF TRAIN AND TEST DATA%%%%%%%%

%---------- First evaluating model on Training(seen) Data
scores_logistic_train = predict(model_logistic, trainX);           % XTRAIN
labels_logistic_train = double(scores_logistic_train >= 0.5);

fprintf('\n\n Confusion Matrix on Train Data');
CFM_logistic_train = confusionmat(trainY, labels_logistic_train, 'Order', [1, 0])         % XTRAIN   YTRAIN
accuracy_logistic_train = sum(diag(CFM_logistic_train))/sum(CFM_logistic_train(:));
technique1_train_accuracy_class0 = accuracy_logistic_train;
technique1_train_accuracy_class1 = accuracy_logistic_train;

%%% Recall
for i =1:size(CFM_logistic_train,1)
    recall_logistic_train(i)=CFM_logistic_train(i,i)/sum(CFM_logistic_train(i,:));
end

technique1_train_recall_class0 = recall_logistic_train(2);
technique1_train_recall_class1 = recall_logistic_train(1);

%%% Précision
for i =1:size(CFM_logistic_train,1)
    precision_logistic_train(i)=CFM_logistic_train(i,i)/sum(CFM_logistic_train(:,i));
end

technique1_train_precision_class0 = precision_logistic_train(2);
technique1_train_precision_class1 = precision_logistic_train(1);

% F1 Score
Fscore_logistic_train = 2*recall_logistic_train.*precision_logistic_train./(precision_logistic_train + recall_logistic_train)
technique1_train_f1_class0 = Fscore_logistic_train(2);
technique1_train_f1_class1 = Fscore_logistic_train(1);

%%%%%%% ROC ON Training Data
[xpos, ypos, T, AUC0] = perfcurve(trainY, 1-scores_logistic_train, 0);     % ROC curve against NO(=0) class
figure, plot(xpos, ypos)        % plot ROC
hold on
[xpos, ypos, T, AUC1] = perfcurve(trainY, scores_logistic_train, 1);     % ROC curve against YES(=1) class
plot(xpos, ypos)        % plot ROC
xlim([-0.05 1.05]), ylim([-0.05 1.05])
xlabel('\bf FP rate'), ylabel('\bf TP rate')
title('\bf ROC by Logit for Training Data')

l = cell(1,2);
l{1}='ROC for "NO"'; l{2}='ROC for "YES"'; 
legend(l,'Location','southeast');


%------------ Now evaluating model on Testing(unseen) Data
scores_logistic_test = predict(model_logistic, testX);           % XTEST
labels_logistic_test = double(scores_logistic_test >= 0.5);
 
fprintf('\n\n Confusion Matrix on Test Data');
CFM_logistic_test = confusionmat(testY, labels_logistic_test, 'Order', [1, 0])         % XTEST   YTEST
accuracy_logistic_test = sum(diag(CFM_logistic_test))/sum(CFM_logistic_test(:));
technique1_test_accuracy_class0 = accuracy_logistic_test;
technique1_test_accuracy_class1 = accuracy_logistic_test;
 
%%% Recall
for i =1:size(CFM_logistic_test,1)
    recall_logistic_test(i)=CFM_logistic_test(i,i)/sum(CFM_logistic_test(i,:));
end
 
technique1_test_recall_class0 = recall_logistic_test(2);
technique1_test_recall_class1 = recall_logistic_test(1);
 
%%% Précision
for i =1:size(CFM_logistic_test,1)
    precision_logistic_test(i)=CFM_logistic_test(i,i)/sum(CFM_logistic_test(:,i));
end
 
technique1_test_precision_class0 = precision_logistic_test(2);
technique1_test_precision_class1 = precision_logistic_test(1);
 
% F1 Score
Fscore_logistic_test = 2*recall_logistic_test.*precision_logistic_test./(precision_logistic_test + recall_logistic_test)
technique1_test_f1_class0 = Fscore_logistic_test(2);
technique1_test_f1_class1 = Fscore_logistic_test(1);
 
%%%%%%% ROC ON Testing Data
[xpos, ypos, T, AUC0] = perfcurve(testY, 1-scores_logistic_test, 0);     % ROC curve against NO(=0) class
figure, plot(xpos, ypos)        % plot ROC
hold on
[xpos, ypos, T, AUC1] = perfcurve(testY, scores_logistic_test, 1);     % ROC curve against YES(=1) class
plot(xpos, ypos)        % plot ROC
xlim([-0.05 1.05]), ylim([-0.05 1.05])
xlabel('\bf FP rate'), ylabel('\bf TP rate')
title('\bf ROC by Logit for Testing Data')
 
l = cell(1,2);
l{1}='ROC for "NO"'; l{2}='ROC for "YES"'; 
legend(l,'Location','southeast');


%%%%%%%%%%%%% LASSO REGULARIZATION
[LB_Lasso, FitInfo] = lassoglm(trainX, trainY, 'binomial', 'NumLambda', 25, 'CV', 10);       %XTRAIN  YTRAIN

% Trace Plot of coefficients  fit by Lasso
lassoPlot(LB_Lasso, FitInfo, 'PlotType', 'Lambda', 'XScale', 'log');
legend('show');

% Cross-validated Deviance of Lasso fit
lassoPlot(LB_Lasso, FitInfo, 'PlotType', 'CV');               
legend('show');


%%%%% After Lasso, Building Logistic Regresion model with predictors AT INDEX 20

% Set the Lamda value (selected based on several trial runs)
lamda = 17;
nonZeroLamdaIndicies = find(LB_Lasso(:,lamda) ~= 0);
LassoTrainX = trainX(:, nonZeroLamdaIndicies);
LassoTestX  = testX(:, nonZeroLamdaIndicies);

% LassoTrainX = [traindata.djob_refer(:,1) traindata.deducation_refer(:, 6) traindata.ddefault_refer(:,1) traindata.dcontact_refer ...
% traindata.dmonth_refer(:, 5:7) traindata.duration_zs  traindata.pdays_zs  traindata.dpoutcome_refer  traindata.empvarrate_zs...
% traindata.consconfidx_zs  traindata.nremployed_zs];

% LassoTestX = [testdata.djob_refer(:,1) testdata.deducation_refer(:, 6) testdata.ddefault_refer(:,1) testdata.dcontact_refer ...
%  testdata.dmonth_refer(:, 5:7) testdata.duration_zs  testdata.pdays_zs  testdata.dpoutcome_refer  testdata.empvarrate_zs...
%  testdata.consconfidx_zs  testdata.nremployed_zs];

fprintf('\n\n After Lasso, Logistic Regresion model with all predictors AT INDEX 17');
model_logistic_after_lasso = fitglm(LassoTrainX, traindata.Y, 'distr', 'binomial', 'link', 'logit')      %%% XTRAIN  YTRAIN
p = model_logistic_after_lasso.Fitted.Response;        % probability
Z = model_logistic_after_lasso.Fitted.LinearPredictor; % Z = qTX
figure, gscatter(Z, p, traindata.Y, 'br'); grid on
zmin = min(Z); zmax = max(Z);
ylim([-0.05 1.05]), xlabel('\bf Z'), 
ylabel('\bf P'),
title(num2str(model_logistic_after_lasso.Coefficients.Estimate'));

% Evaluating model on Test Data
scores_logistic_after_lasso = predict(model_logistic_after_lasso, LassoTestX);           
labels_logistic_after_lasso = double(scores_logistic_after_lasso >= 0.5);

fprintf('\n\n Confusion Matrix on TestData');
CFM_logistic_after_lasso = confusionmat(testY, labels_logistic_after_lasso, 'Order', [1, 0])
accuracy_logistic_after_lasso = sum(diag(CFM_logistic_after_lasso))/sum(CFM_logistic_after_lasso(:));
technique1_accuracy_after_lasso_class0 = accuracy_logistic_after_lasso;
technique1_accuracy_after_lasso_class1 = accuracy_logistic_after_lasso;

% Recall
for i =1:size(CFM_logistic_after_lasso,1)
    recall_logistic_after_lasso(i)=CFM_logistic_after_lasso(i,i)/sum(CFM_logistic_after_lasso(i,:));
end

technique1_recall_after_lasso_class0 = recall_logistic_after_lasso(2);
technique1_recall_after_lasso_class1 = recall_logistic_after_lasso(1);

% Précision
for i =1:size(CFM_logistic_after_lasso,1)
    precision_logistic_after_lasso(i)=CFM_logistic_after_lasso(i,i)/sum(CFM_logistic_after_lasso(:,i));
end

technique1_precision_after_lasso_class0 = precision_logistic_after_lasso(2);
technique1_precision_after_lasso_class1 = precision_logistic_after_lasso(1);

% F1 Score
Fscore_logistic_after_lasso = 2*recall_logistic_after_lasso.*precision_logistic_after_lasso./(precision_logistic_after_lasso + recall_logistic_after_lasso)
technique1_f1_after_lasso_class0 = Fscore_logistic_after_lasso(2);
technique1_f1_after_lasso_class1 = Fscore_logistic_after_lasso(1);
fprintf('Ending technique 1 at: %s\n', datestr(now,'HH:MM:SS.FFF'));

% Compare the previoius model with the lasso model
f1_scores = [technique1_test_f1_class0 technique1_test_f1_class1;
             technique1_f1_after_lasso_class0 technique1_f1_after_lasso_class1];
         
figure('rend','painters','pos',[1000 1000 900 600]);         
b = bar(f1_scores);

ax = gca;
ax.XAxis.TickLabelsMode = 'manual';
ax.XAxis.TickLabels = {'Logistic Regression W/O Lasso Simplification','Logistic Lasso Regression'};
l = cell(1,2);
l{1}='Will Not Open an Account'; l{2}='Will Open an Account';
legend(b,l,'Location','north');
title('Comparison of F1 Scores');

% ROC Plot for class 0
[xpos0, ypos0, T0, AUC0] = perfcurve(testY, 1-scores_logistic_test, 0);
[xpos1, ypos1, T1, AUC1] = perfcurve(testY, 1-scores_logistic_after_lasso, 0);

figure;
plot(xpos0, ypos0), hold on, plot(xpos1, ypos1);
l = cell(1,2);
l{1}='Logistic Regression W/O Lasso Simplification'; l{2}='Logistic Lasso Regression'; 
legend(l,'Location','southeast');
xlim([-0.05 1.05]), ylim([-0.05 1.05])
xlabel('\bf FP rate'), ylabel('\bf TP rate')
title('\bf ROC Plots: <Will Not Open an Account>')

% ROC Plot for class 1
[xpos0, ypos0, T0, AUC0] = perfcurve(testY, scores_logistic_test, 1);
[xpos1, ypos1, T1, AUC1] = perfcurve(testY, scores_logistic_after_lasso, 1);
figure;
plot(xpos0, ypos0), hold on, plot(xpos1, ypos1);
l = cell(1,2);
l{1}='Logistic Regression W/O Lasso Simplification'; l{2}='Logistic Lasso Regression';
legend(l,'Location','southeast');
xlim([-0.05 1.05]), ylim([-0.05 1.05])
xlabel('\bf FP rate'), ylabel('\bf TP rate')
title('\bf ROC Plots: <Will Open an Account>')

% End TECHNIQUE 1
% --------------------------------------------------------------------------


% --------------------------------------------------------------------------
% Begin TECHNIQUE 2
% Support Vector Machine

fprintf('Starting technique 2 at: %s\n', datestr(now,'HH:MM:SS.FFF'));

% Run the model using all predictors
myModel = fitcsvm(trainX, trainY, 'KernelScale', .1, 'BoxConstraint', 1);

% Run the model using just the preditors that survived the Lasso regression
myModel_lasso = fitcsvm(LassoTrainX, trainY, 'KernelScale', .1, 'BoxConstraint', 1);

% predict the labels using all predictors
[labels_svm, scores_svm] = predict(myModel, testX);

% predict the labels using just the preditors that survived the Lasso regression
[labels_lasso_svm, scores_lasso_svm] = predict(myModel_lasso, LassoTestX);

CFM_SVM = confusionmat(testY, labels_svm, 'Order', [1, 0]);
CFM_SVM_lasso = confusionmat(testY, labels_lasso_svm, 'Order', [1, 0]);

accuracy_svm = sum(diag(CFM_SVM))/sum(CFM_SVM(:));
technique2_accuracy_class0 = accuracy_svm;
technique2_accuracy_class1 = accuracy_svm;

accuracy_lasso_svm = sum(diag(CFM_SVM_lasso))/sum(CFM_SVM_lasso(:));
technique2_lasso_accuracy_class0 = accuracy_lasso_svm;
technique2_lasso_accuracy_class1 = accuracy_lasso_svm;

% Recall
for i =1:size(CFM_SVM,1)
    recall_svm(i)=CFM_SVM(i,i)/sum(CFM_SVM(i,:));
end

technique2_recall_class0 = recall_svm(2);
technique2_recall_class1 = recall_svm(1);

for i =1:size(CFM_SVM_lasso,1)
    recall_lasso_svm(i)=CFM_SVM_lasso(i,i)/sum(CFM_SVM_lasso(i,:));
end

technique2_lasso_recall_class0 = recall_lasso_svm(2);
technique2_lasso_recall_class1 = recall_lasso_svm(1);

% Précision
for i =1:size(CFM_SVM,1)
    precision_svm(i)=CFM_SVM(i,i)/sum(CFM_SVM(:,i));
end

technique2_precision_class0 = precision_svm(2);
technique2_precision_class1 = precision_svm(1);

for i =1:size(CFM_SVM_lasso,1)
    precision_lasso_svm(i)=CFM_SVM_lasso(i,i)/sum(CFM_SVM_lasso(:,i));
end

technique2_lasso_precision_class0 = precision_lasso_svm(2);
technique2_lasso_precision_class1 = precision_lasso_svm(1);

% F1 Score
Fscore_svm = 2*recall_svm.*precision_svm./(precision_svm + recall_svm)
technique2_f1_class0 = Fscore_svm(2);
technique2_f1_class1 = Fscore_svm(1);

Fscore_lasso_svm = 2*recall_lasso_svm.*precision_lasso_svm./(precision_lasso_svm + recall_lasso_svm)
technique2_f1_lasso_class0 = Fscore_lasso_svm(2);
technique2_f1_lasso_class1 = Fscore_lasso_svm(1);

% Compare the previoius model with the lasso model
f1_scores = [technique2_f1_class0 technique2_f1_class1;
             technique2_f1_lasso_class0 technique2_f1_lasso_class1];
         
figure('rend','painters','pos',[1000 1000 900 600]);         
b = bar(f1_scores);

ax = gca;
ax.XAxis.TickLabelsMode = 'manual';
ax.XAxis.TickLabels = {'SVM No Kernel W/O Lasso Simplification','SVM NO Kernel With Lasso Regression'};
l = cell(1,2);
l{1}='Will Not Open an Account'; l{2}='Will Open an Account';
legend(b,l,'Location','north');
title('Comparison of F1 Scores');

% ROC Plot for class 0
[xpos0, ypos0, T0, AUC0] = perfcurve(testY, scores_svm(:, 1), 0);
[xpos1, ypos1, T1, AUC1] = perfcurve(testY, scores_lasso_svm(:, 1), 0);

figure;
plot(xpos0, ypos0), hold on, plot(xpos1, ypos1);
l = cell(1,2);
l{1}='SVM No Kernel W/O Lasso Simplification'; l{2}='SVM No Kernel With Lasso Regression'; 
legend(l,'Location','southeast');
xlim([-0.05 1.05]), ylim([-0.05 1.05])
xlabel('\bf FP rate'), ylabel('\bf TP rate')
title('\bf ROC Plots: <Will Not Open an Account>')

% ROC Plot for class 1
[xpos0, ypos0, T0, AUC0] = perfcurve(testY, scores_svm(:, 2), 1);
[xpos1, ypos1, T1, AUC1] = perfcurve(testY, scores_lasso_svm(:, 2), 1);
figure;
plot(xpos0, ypos0), hold on, plot(xpos1, ypos1);
l = cell(1,2);
l{1}='SVM No Kernel W/O Lasso Simplification'; l{2}='SVM No Kernel With Lasso Regression';
legend(l,'Location','southeast');
xlim([-0.05 1.05]), ylim([-0.05 1.05])
xlabel('\bf FP rate'), ylabel('\bf TP rate')
title('\bf ROC Plots: <Will Open an Account>')


% End TECHNIQUE 2
% --------------------------------------------------------------------------





% --------------------------------------------------------------------------
% Begin TECHNIQUE 3
% SVM method with RBF Kernel

fprintf('Starting technique 3 at: %s\n', datestr(now,'HH:MM:SS.FFF'));

% Run the model using all predictors
SVM_Kernel = fitcsvm(trainX, trainY,'KernelFunction', 'rbf', 'KernelScale', 1, 'BoxConstraint', 1);

% Run the model using just the preditors that survived the Lasso regression
SVM_Kernel_Lasso = fitcsvm(LassoTrainX, trainY,'KernelFunction', 'rbf', 'KernelScale', 1, 'BoxConstraint', 1);

% predict the labels for the model representing all predictors
[labels_kernel, scores_kernel] = predict(SVM_Kernel,testX);
CFM_kernel = confusionmat(testY, labels_kernel, 'Order', [1, 0])

% predict the labels for the model using just the preditors that survived the Lasso regression
[labels_kernel_lasso, scores_kernel_lasso] = predict(SVM_Kernel_Lasso,LassoTestX);
CFM_kernel_lasso = confusionmat(testY, labels_kernel_lasso, 'Order', [1, 0])

% Accuracy
accuracy_kernel = sum(diag(CFM_kernel))/sum(CFM_kernel(:))
technique3_accuracy_class0 = accuracy_kernel;
technique3_accuracy_class1 = accuracy_kernel;

accuracy_kernel_lasso = sum(diag(CFM_kernel_lasso))/sum(CFM_kernel_lasso(:))
technique3_accuracy_lasso_class0 = accuracy_kernel_lasso;
technique3_accuracy_lasso_class1 = accuracy_kernel_lasso;

% Recall
for i =1:size(CFM_kernel,1)
    recall_kernel(i)=CFM_kernel(i,i)/sum(CFM_kernel(i,:));
end
technique3_recall_class0 = recall_kernel(2);
technique3_recall_class1 = recall_kernel(1);

for i =1:size(CFM_kernel_lasso,1)
    recall_kernel_lasso(i)=CFM_kernel_lasso(i,i)/sum(CFM_kernel_lasso(i,:));
end
technique3_recall_lasso_class0 = recall_kernel_lasso(2);
technique3_recall_lasso_class1 = recall_kernel_lasso(1);

% Précision
for i =1:size(CFM_kernel,1)
    precision_kernel(i)=CFM_kernel(i,i)/sum(CFM_kernel(:,i));
end
technique3_precision_class0 = precision_kernel(2);
technique3_precision_class1 = precision_kernel(1);

for i =1:size(CFM_kernel_lasso,1)
    precision_kernel_lasso(i)=CFM_kernel_lasso(i,i)/sum(CFM_kernel_lasso(:,i));
end
technique3_precision_lasso_class0 = precision_kernel_lasso(2);
technique3_precision_lasso_class1 = precision_kernel_lasso(1);

% F1 Score
Fscore_kernel = 2*recall_kernel.*precision_kernel./(precision_kernel + recall_kernel)
technique3_f1_class0 = Fscore_kernel(2);
technique3_f1_class1 = Fscore_kernel(1);

Fscore_kernel_lasso = 2*recall_kernel_lasso.*precision_kernel_lasso./(precision_kernel_lasso + recall_kernel_lasso)
technique3_f1_lasso_class0 = Fscore_kernel_lasso(2);
technique3_f1_lasso_class1 = Fscore_kernel_lasso(1);

% Compare the previoius model with the lasso model
f1_svm_kerner_scores = [technique3_f1_class0 technique3_f1_class1;
                        technique3_f1_lasso_class0 technique3_f1_lasso_class1];
         
figure('rend','painters','pos',[1000 1000 900 600]);         
b = bar(f1_svm_kerner_scores);

ax = gca;
ax.XAxis.TickLabelsMode = 'manual';
ax.XAxis.TickLabels = {'SVM Kernel W/O Lasso Simplification','SVM Kernel With Lasso Regression'};
l = cell(1,2);
l{1}='Will Not Open an Account'; l{2}='Will Open an Account';
legend(b,l,'Location','north');
title('Comparison of F1 Scores');

% ROC Plot for class 0
[xpos0, ypos0, T0, AUC0] = perfcurve(testY, scores_kernel(:, 1), 0);
[xpos1, ypos1, T1, AUC1] = perfcurve(testY, scores_kernel_lasso(:, 1), 0);

figure;
plot(xpos0, ypos0), hold on, plot(xpos1, ypos1);
l = cell(1,2);
l{1}='SVM Kernel W/O Lasso Simplification'; l{2}='SVM Kernel With Lasso Regression'; 
legend(l,'Location','southeast');
xlim([-0.05 1.05]), ylim([-0.05 1.05])
xlabel('\bf FP rate'), ylabel('\bf TP rate')
title('\bf ROC Plots: <Will Not Open an Account>')

% ROC Plot for class 1
[xpos0, ypos0, T0, AUC0] = perfcurve(testY, scores_kernel(:, 2), 1);
[xpos1, ypos1, T1, AUC1] = perfcurve(testY, scores_kernel_lasso(:, 2), 1);
figure;
plot(xpos0, ypos0), hold on, plot(xpos1, ypos1);
l = cell(1,2);
l{1}='SVM Kernel W/O Lasso Simplification'; l{2}='SVM Kernel With Lasso Regression'; 
legend(l,'Location','southeast');
xlim([-0.05 1.05]), ylim([-0.05 1.05])
xlabel('\bf FP rate'), ylabel('\bf TP rate')
title('\bf ROC Plots: <Will Open an Account>')
fprintf('Ending technique 3 at: %s\n', datestr(now,'HH:MM:SS.FFF'));

% End TECHNIQUE 3
% --------------------------------------------------------------------------







% --------------------------------------------------------------------------
% Begin TECHNIQUE 4
% 

fprintf('Starting technique 4 at: %s\n', datestr(now,'HH:MM:SS.FFF'));
% Create the one hot matrix using the dummyvar function.
% The dummyvar function does not support a zero based index so 
% we must convert the vector to be 1 based. 
trainYOneHot = trainY + 1;
trainYOneHot = dummyvar(trainYOneHot);
testYOneHot = testY + 1;
testYOneHot = dummyvar(testYOneHot);

% Transpose the inputs to prepare for patternnet procesing.
trainXColumnBased = trainX';
trainYOneHotColumnBased = trainYOneHot';
testXColumnBased = testX';
testYOneHotColumnBased = testYOneHot';

% Run the classification analysis using a patternnet 
% neural network (one deep)
ffn = patternnet(2);
ffn.trainParam.goal = 1e-20;
ffn.divideFcn = 'dividetrain';
ffn = train(ffn, trainXColumnBased, trainYOneHotColumnBased);
view(ffn);
scores_nn = sim(ffn, testXColumnBased);

% Run the classification analysis using a patternnet 
% neural network (two deep)
ffn_8_3 = patternnet([8 3]);
ffn_8_3.trainParam.goal = 1e-20;
ffn_8_3.divideFcn = 'dividetrain';
ffn_8_3 = train(ffn_8_3, trainXColumnBased, trainYOneHotColumnBased);
view(ffn_8_3);
scores_nn_8_3 = sim(ffn_8_3, testXColumnBased);


% Create a vector of predicted labels using the output
% of the simulation (scores_nn). 
[Vs, labels_nn] = max(scores_nn);

% Create a vector of predicted labels using the output
% of the simulation (scores_nn). 
[Vs_8_3, labels_nn_8_3] = max(scores_nn_8_3);

% Adjust the index to be zero based so that it can be compared to the 
% response.
labels_nn = labels_nn - 1;
labels_nn_8_3 = labels_nn_8_3 - 1;

CFM_nn = confusionmat(trainYOneHotColumnBased(2, :), labels_nn,  'Order', [1, 0]);
accuracy_nn = sum(diag(CFM_nn))/sum(CFM_nn(:))
technique4_accuracy_class1 = accuracy_nn;
technique4_accuracy_class0 = accuracy_nn;

CFM_nn_8_3 = confusionmat(trainYOneHotColumnBased(2, :), labels_nn_8_3,  'Order', [1, 0]);
accuracy_nn_8_3 = sum(diag(CFM_nn_8_3))/sum(CFM_nn_8_3(:))
technique4_accuracy_8_3_class1 = accuracy_nn_8_3;
technique4_accuracy_8_3_class0 = accuracy_nn_8_3;

% Recall
for i =1:size(CFM_nn,1)
    recall_nn(i)=CFM_nn(i,i)/sum(CFM_nn(i,:));
end
technique4_recall_class1 = recall_nn(1);
technique4_recall_class0 = recall_nn(2);

for i =1:size(CFM_nn_8_3,1)
    recall_nn_8_3(i)=CFM_nn_8_3(i,i)/sum(CFM_nn_8_3(i,:));
end
technique4_recall_8_3_class1 = recall_nn_8_3(1);
technique4_recall_8_3_class0 = recall_nn_8_3(2);

% Précision
for i =1:size(CFM_nn,1)
    precision_nn(i)=CFM_nn(i,i)/sum(CFM_nn(:,i));
end
technique4_precision_class1 = precision_nn(1);
technique4_precision_class0 = precision_nn(2);

for i =1:size(CFM_nn_8_3,1)
    precision_8_3_nn(i)=CFM_nn_8_3(i,i)/sum(CFM_nn_8_3(:,i));
end
technique4_precision_8_3_class1 = precision_8_3_nn(1);
technique4_precision_8_3_class0 = precision_8_3_nn(2);


% F1 Score
Fscore_nn = 2*recall_nn.*precision_nn./(precision_nn + recall_nn)
technique4_f1_class1 = Fscore_nn(1);
technique4_f1_class0 = Fscore_nn(2);

Fscore_8_3_nn = 2*recall_nn_8_3.*precision_8_3_nn./(precision_8_3_nn + recall_nn_8_3)
technique4_f1_8_3_class1 = Fscore_8_3_nn(1);
technique4_f1_8_3_class0 = Fscore_8_3_nn(2);


f1_scores = [technique4_f1_class0 technique4_f1_class1;
             technique4_f1_8_3_class0 technique4_f1_8_3_class1
             ];
         
figure('rend','painters','pos',[1000 1000 900 600]);         
b = bar(f1_scores);

ax = gca;
ax.XAxis.TickLabelsMode = 'manual';
ax.XAxis.TickLabels = {'Neural Network With 1 Node 2 Deep','Neural Network With 8/3 Configuration'};
l = cell(1,2);
l{1}='Will Not Open an Account'; l{2}='Will Open an Account';
legend(b,l,'Location','north');
title('Comparison of F1 Scores For Two Neural Network Configurations');

% ROC Plot for class 0

[xpos0, ypos0, T0, AUC0] = perfcurve(testYOneHotColumnBased(2,:), scores_nn(1, :), 0);
[xpos1, ypos1, T1, AUC1] = perfcurve(testYOneHotColumnBased(2,:), scores_nn_8_3(1, :), 0);

figure;
plot(xpos0, ypos0), hold on, plot(xpos1, ypos1);
l = cell(1,2);
l{1}='Neural Network With 1 Node 2 Deep'; l{2}='Neural Network With 8/3 Configuration'; 
legend(l,'Location','southeast');
xlim([-0.05 1.05]), ylim([-0.05 1.05])
xlabel('\bf FP rate'), ylabel('\bf TP rate')
title('\bf ROC Plots: <Will Not Open an Account>')

% ROC Plot for class 1
[xpos2, ypos2, T2, AUC2] = perfcurve(testYOneHotColumnBased(2,:), scores_nn(2, :), 1);
[xpos3, ypos3, T3, AUC3] = perfcurve(testYOneHotColumnBased(2,:), scores_nn_8_3(2, :), 1);
figure;
plot(xpos0, ypos0), hold on, plot(xpos1, ypos1);
l = cell(1,2);
l{1}='Neural Network With 1 Node 2 Deep'; l{2}='Neural Network With 8/3 Configuration'; 
legend(l,'Location','southeast');
xlim([-0.05 1.05]), ylim([-0.05 1.05])
xlabel('\bf FP rate'), ylabel('\bf TP rate')
title('\bf ROC Plots: <Will Open an Account>')

fprintf('Ending technique 4 at: %s\n', datestr(now,'HH:MM:SS.FFF'));
% End TECHNIQUE 4
% --------------------------------------------------------------------------







% --------------------------------------------------------------------------
% Begin Ensemble Voting Technique

% Here are the list of labels we will use:
% labels_logistic_test
% labels_logistic_after_lasso
% labels_svm
% labels_kernel
% labels_nn

% By default, the vote is for class 0.
% We need at least 3 votes for class 1 for class 1 to win the voting for
% any given record.

% Initialize 0 to be the vote for all records:
number_of_records = length(testY);
final_ensemble_hard_vote = zeros(number_of_records,1);
final_ensemble_soft_vote = zeros(number_of_records,1);
ensemble_scores = zeros(number_of_records,1);

for index = 1:length(testY)
    
    % Determing the number of votes we have for class 1
    vote_total = labels_logistic_test(index) + ...
                 labels_logistic_after_lasso(index) + ...
                 labels_svm(index) + ...
                 labels_kernel(index) + ...
                 labels_nn(index) ;
    
    % Determing the avererage of votes we have for class 1
    vote_mean = mean([labels_logistic_test(index)  ...
                 labels_logistic_after_lasso(index)  ...
                 labels_svm(index)  ...
                 labels_kernel(index)  ...
                 labels_nn(index) ...
                 ]);
             
    % Update the final hard vote if we have at least 3 votes for class 1
    if (vote_total >= 3)
        final_ensemble_hard_vote(index) = 1;
    end
    
    % Update the final soft vote if we have at an average scores greater than .5 for class 1
    if (vote_mean >= 0.5)
        final_ensemble_soft_vote(index) = 1;
    end
    
    % Create a scores array for the ensemble method using a mean
    ensemble_scores(index) = mean([scores_logistic_test(index) scores_svm(index) scores_kernel(index) scores_nn(index)]);

end

CFM_hard_ensemble = confusionmat(testY, final_ensemble_hard_vote, 'Order', [1, 0]);
accuracy_hard_ensemble = sum(diag(CFM_hard_ensemble))/sum(CFM_hard_ensemble(:))
ensemble_hard_accuracy_class0 = accuracy_hard_ensemble;
ensemble_hard_accuracy_class1 = accuracy_hard_ensemble;

CFM_soft_ensemble = confusionmat(testY, final_ensemble_soft_vote, 'Order', [1, 0]);
accuracy_soft_ensemble = sum(diag(CFM_soft_ensemble))/sum(CFM_soft_ensemble(:))
ensemble_soft_accuracy_class0 = accuracy_soft_ensemble;
ensemble_soft_accuracy_class1 = accuracy_soft_ensemble;

% Recall
for i =1:size(CFM_hard_ensemble,1)
    ensemble_hard_recall(i)=CFM_hard_ensemble(i,i)/sum(CFM_hard_ensemble(i,:));
end
ensemble_hard_recall_class0 = ensemble_hard_recall(2);
ensemble_hard_recall_class1 = ensemble_hard_recall(1);

for i =1:size(CFM_soft_ensemble,1)
    ensemble_soft_recall(i)=CFM_soft_ensemble(i,i)/sum(CFM_soft_ensemble(i,:));
end
ensemble_soft_recall_class0 = ensemble_soft_recall(2);
ensemble_soft_recall_class1 = ensemble_soft_recall(1);

% Précision
for i =1:size(CFM_hard_ensemble,1)
    ensemble_hard_precision(i)=CFM_hard_ensemble(i,i)/sum(CFM_hard_ensemble(:,i));
end
ensemble_hard_precision_class0 = ensemble_hard_precision(2);
ensemble_hard_precision_class1 = ensemble_hard_precision(1);

for i =1:size(CFM_soft_ensemble,1)
    ensemble_soft_precision(i)=CFM_soft_ensemble(i,i)/sum(CFM_soft_ensemble(:,i));
end
ensemble_soft_precision_class0 = ensemble_soft_precision(2);
ensemble_soft_precision_class1 = ensemble_soft_precision(1);

% F1 Score
ensemble_hard_Fscore = 2*ensemble_hard_recall.*ensemble_hard_precision./(ensemble_hard_precision + ensemble_hard_recall)
ensemble_f1_hard_class0 = ensemble_hard_Fscore(2);
ensemble_f1_hard_class1 = ensemble_hard_Fscore(1);

ensemble_soft_Fscore = 2*ensemble_soft_recall.*ensemble_soft_precision./(ensemble_soft_precision + ensemble_soft_recall)
ensemble_f1_soft_class0 = ensemble_soft_Fscore(2);
ensemble_f1_soft_class1 = ensemble_soft_Fscore(1);

% End Begin Ensemble Voting Technique
% --------------------------------------------------------------------------






% --------------------------------------------------------------------------
% Begin Technique Comparisions

f1_scores = [technique1_f1_after_lasso_class0 technique1_f1_after_lasso_class1;
             technique2_f1_class0 technique2_f1_class1;
             technique3_f1_class0 technique3_f1_class1;
             technique4_f1_class0 technique4_f1_class1;
             ensemble_f1_hard_class0 ensemble_f1_hard_class1];
         
figure('rend','painters','pos',[1000 1000 900 600]);         
b = bar(f1_scores);

ax = gca;
ax.XAxis.TickLabelsMode = 'manual';
ax.XAxis.TickLabels = {'Logistic Lasso Regression','Linear SVM Classifier','NonLinear SVM Kernal Classifier', 'Neuro Network Classifier', 'Ensemble Classifier'};
l = cell(1,2);
l{1}='Will Not Open an Account'; l{2}='Will Open an Account';
legend(b,l,'Location','north');
title('Comparison of F1 Scores');

figure('rend','painters','pos',[5000 5000 900 600]);


precision_scores = [technique1_precision_after_lasso_class0 technique1_precision_after_lasso_class1;
                    technique2_precision_class0 technique2_precision_class1;
                    technique3_precision_class0 technique3_precision_class1;
                    technique4_precision_class0 technique4_precision_class1;
                    ensemble_hard_precision_class0 ensemble_hard_precision_class1];
b = bar(precision_scores);
ax = gca;
ax.XAxis.TickLabelsMode = 'manual';
ax.XAxis.TickLabels = {'Logistic Lasso Regression','Linear SVM Classifier','NonLinear SVM Kernal Classifier', 'Neuro Network Classifier', 'Ensemble Classifier'};

l = cell(1,2);
l{1}='Will Not Open an Account'; l{2}='Will Open an Account';
legend(b,l,'Location','north');
title('Comparison of Precision Scores');

figure;


accuracy_scores = [accuracy_logistic_after_lasso accuracy_logistic_after_lasso;
                   technique2_accuracy_class0 technique2_accuracy_class1;
                   technique3_accuracy_class0 technique3_accuracy_class1;
                   technique4_accuracy_class0 technique4_accuracy_class1;
                   ensemble_hard_accuracy_class0 ensemble_hard_accuracy_class1];
               
b = bar(accuracy_scores);
ax = gca;
ax.XAxis.TickLabelsMode = 'manual';
ax.XAxis.TickLabels = {'Logistic Lasso Regression','Linear SVM Classifier','NonLinear SVM Kernal Classifier', 'Neuro Network Classifier', 'Ensemble Classifier'};

l = cell(1,2);
l{1}='Will Not Open an Account'; l{2}='Will Open an Account';
legend(b,l,'Location','north');
title('Comparison of Accuracy Scores');

figure('rend','painters','pos',[9000 9000 900 600]);


recall_scores = [technique1_recall_after_lasso_class0 technique1_recall_after_lasso_class1;
                 technique2_recall_class0 technique2_recall_class1;
                 technique3_recall_class0 technique3_recall_class1;
                 technique4_recall_class0 technique4_recall_class1;
                 ensemble_hard_recall_class0 ensemble_hard_recall_class1];
b = bar(recall_scores);
ax = gca;

ax.XAxis.TickLabels = {'Logistic Lasso Regression','Linear SVM Classifier','NonLinear SVM Kernal Classifier', 'Neuro Network Classifier', 'Ensemble Classifier'};
l = cell(1,2);
l{1}='Will Not Open an Account'; l{2}='Will Open an Account';
legend(b,l,'Location','north');
title('Comparison of Recall Scores');

% ROC Plot for class 0
[xpos0, ypos0, T0, AUC0] = perfcurve(testY, 1-scores_logistic_test, 0);
[xpos1, ypos1, T1, AUC1] = perfcurve(testY, scores_svm(:, 1), 0);
[xpos2, ypos2, T2, AUC2] = perfcurve(testY, scores_kernel(:, 1), 0);
[xpos3, ypos3, T3, AUC3] = perfcurve(testYOneHotColumnBased(2,:), scores_nn(1, :), 0);
[xpos4, ypos4, T4, AUC4] = perfcurve(testY, ensemble_scores, 0);

figure;
plot(xpos0, ypos0), hold on, plot(xpos1, ypos1), plot(xpos2, ypos2), plot(xpos3, ypos3), plot(xpos4, ypos4);;
l = cell(1,5);
l{1}='Logistic Lasso Regression'; l{2}='Linear SVM Classifier'; l{3}='NonLinear SVM Kernal Classifier'; l{4}='Neuro Network Classifier'; l{5}='Ensemble Classifier';
legend(l,'Location','southeast');
xlim([-0.05 1.05]), ylim([-0.05 1.05])
xlabel('\bf FP rate'), ylabel('\bf TP rate')
title('\bf ROC Plots: <Will Not Open an Account>')

% ROC Plot for class 1
[xpos0, ypos0, T0, AUC0] = perfcurve(testY, scores_logistic_test, 1);
[xpos1, ypos1, T1, AUC1] = perfcurve(testY, scores_svm(:, 2), 1);
[xpos2, ypos2, T2, AUC2] = perfcurve(testY, scores_kernel(:, 2), 1);
[xpos3, ypos3, T3, AUC3] = perfcurve(testYOneHotColumnBased(2,:), scores_nn(2, :), 1);
[xpos4, ypos4, T4, AUC4] = perfcurve(testY, 1-ensemble_scores, 1);

figure;
plot(xpos0, ypos0), hold on, plot(xpos1, ypos1), plot(xpos2, ypos2), plot(xpos3, ypos3), plot(xpos4, ypos4);
l = cell(1,5);
l{1}='Logistic Lasso Regression'; l{2}='Linear SVM Classifier'; l{3}='NonLinear SVM Kernal Classifier'; l{4}='Neuro Network Classifier'; l{5}='Ensemble Classifier';
legend(l,'Location','southeast');
xlim([-0.05 1.05]), ylim([-0.05 1.05])
xlabel('\bf FP rate'), ylabel('\bf TP rate')
title('\bf ROC Plots: <Will Open an Account>')
% End Technique Comparisions
% --------------------------------------------------------------------------


