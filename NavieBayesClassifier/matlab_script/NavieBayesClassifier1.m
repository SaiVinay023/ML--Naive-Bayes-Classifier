% Saivinay Manda- S4845876

%Naive Bayes Classifier
clear all;
close all; 
clc; 
%% Task 1: Data preprocessing
%how is the data set composed? 
%the num of rows of data matrix corrisponds to the num of instances, instead the num of columns corresponds to the num of predictors
%(attribute + target).
load data.txt; 

%% Check the value of 2 matrices greater than 1
for i=1:size(data,1)
    for j=1:size(data,2)
        if data(i,j) < 1 
            disp('Error: at least one value less then 1');
        end
    end
end


%% Create 2 matrices : Training set and Test set 
%define 2 matrices 
% which are dividing data set into 2 parts: 
%a set of data, as a n x (d+1) matrix, to be used as the training set; 
%another set of data, as a m x c matrix, to be used as the test set;
[indexes, set] = create_set(data); 

%% Display two data_sets : test set and training set 
Training_set = table(set.training,'VariableName',{'TrainingSet'}); 
disp(Training_set); 
Test_set = table(set.test,'VariableName',{'TestSet'}); 
disp(Test_set); 
Real_Target_test = table(set.real_target_test, 'VariableName', {'RealTarget'});
disp(Real_Target_test); 


%% Check that the number c of columns of the test matrix is at least the number d of columns of the training matrix - 1
a = isempty(set.real_target_test)
if a == 1
    disp('the test set does not contain the target'); 
elseif a == 0
    disp('the test set contains the target'); 
end 
    
    
%% Task 2: Build a naive Bayes classifier
% Train a Naive Bayes classifier on the training set (first input argument), weare using its last column as the target

[B, num, Target, P] = probability(set.training); 

%%  Here we are Displaying the Properties of Training set
Properties_Training = table(num.rows, num.predictors, num.levels_classes, num.levels_for_each_input,...
    'VariableNames', {'num instances', 'num variables', 'num levels of target attribute', 'levels of each variable'},...
    'RowName', {'Training set'});

    disp(Properties_Training); 


%% Classify the test set according to the inferred rule
[Num_t, P_t] = classify_testset(set.test,num.levels_classes,P.likelihoods, P.prior_probability_class);

%% Display the Properties of Test set
Properties_Test = table(Num_t.rows, Num_t.columns, num.levels_classes, num.levels_for_each_input,...
    'VariableNames', {'num instances', 'num variables', 'num levels of target attribute', 'levels of each variable'},...
    'RowName', {'Test set'});

    disp(Properties_Test); 
    
%% Display Prior Probability of class
%classtype = cell(1);
for i = 1: num.levels_classes 
    classtypes{i} = sprintf('class_%d', i);

end

    PriorProbabilityClass = table((P.prior_probability_class)',...
    'VariableNames', {'Prior Probability Class'},...
    'RowNames', classtypes);
disp(PriorProbabilityClass); 



%% Display Posterior Probability for each target's level
for e = 1:Num_t.rows
    num_instances_test{e} = sprintf('num.instances.testset_%d', e); 
end

Posterior_Probability = table(P_t.PosteriorProbability,...
    'VariableName', {'Posterior'},...
    'RowNames', num_instances_test);

disp(Posterior_Probability); 


%% Check the predicted level for each instance
%Check the max probability 
Max = max(P_t.PosteriorProbability, [], 2); 
Predicted_target = zeros(Num_t.columns,1); 
for e = 1:Num_t.columns
    for i=1:num.levels_classes
        if Max(e) == P_t.PosteriorProbability(e,i) 
            Predicted_target(e,1) = Predicted_target(e,1)+i; 
        end
    end
end

%% Disp the comparison of real target of test set with the predicted target 
Comparison_target = table(set.real_target_test, Predicted_target, 'VariableName', {'RealTarget','PredictedTarget'});
disp(Comparison_target); 
%% compute and return the error rate obtained (number of errors / m)
Error_Rate = 0;
for e = 1:Num_t.columns %num of instance of test set
    if (set.real_target_test(e) ~= Predicted_target(e)) 
        Error_Rate = (Error_Rate + 1)/Num_t.columns; 
    end
end
Error_Rate 

%% Task 3: Improve the classifier with Laplace (additive) smoothing

%first point : In the data preparation step, add the information about the number of levels. 
%This means that for each data column you should add the number of possible different values for that column.

New_set.training = [num.levels_for_each_input, 0; set.training]; 
New_set.test = [num.levels_for_each_input; set.test]; 

%% Compute probability improving Navie Bayes classifier

[P_Laplace] = probability_Laplace (B, New_set.training, num.predictors,num.max_levels, num.levels_for_each_input, num.levels_classes, num.appearance_of_each_level_class);

%% Classify the test set with the improved probability
[Num_t_Laplace, P_t_Laplace] = classify_testset(set.test, num.levels_classes, P_Laplace, P.prior_probability_class);

%% %% Display Posterior Probability with additive smoothing for each target's level

for e = 1:Num_t_Laplace.rows
    num_instances_test{e} = sprintf('num.instances.testset_%d', e); 
end
Posterior_Probability_Laplace = table(P_t_Laplace.PosteriorProbability,...
    'VariableName', {'Posterior (Laplace)'},...
    'RowNames', num_instances_test);

disp(Posterior_Probability_Laplace); 


%% Check the predicted level for each instance
%Check the max probability 
Max_Laplace = max(P_t_Laplace.PosteriorProbability, [], 2); 
Predicted_target_Laplace = zeros(Num_t_Laplace.columns,1); 
for e = 1:Num_t_Laplace.columns
    for i=1:num.levels_classes
        if Max_Laplace(e) == P_t_Laplace.PosteriorProbability(e,i) 
            Predicted_target_Laplace(e,1) = Predicted_target_Laplace(e,1)+i; 
        end
    end
end

%% Compare real target of test set with the predicted target (Laplace) 
Comparison_target_laplace = table(set.real_target_test, Predicted_target_Laplace, 'VariableName', {'RealTarget','PredictedTarget (Laplace)'});
disp(Comparison_target_laplace); 

%% compute and return the error rate obtained (number of errors / m)
Error_Rate_Laplace = 0;
for e = 1:Num_t_Laplace.columns
    if (set.real_target_test(e) == Predicted_target_Laplace(e)) 
        Error_Rate_Laplace = (Error_Rate_Laplace + 1)/Num_t_Laplace.columns; 
    end
end
Error_Rate_Laplace 

    


