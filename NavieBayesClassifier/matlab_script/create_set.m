% Saivinay Manda- S4845876
function [indexes,set] = create_set(data_set)

[indexes.rows, indexes.columns] = size(data_set);%indexes.rows= num rows and indexes.columns = num columns. 
indexes.indexes_rand = randperm(indexes.rows);%take randomly indexes of rows of data_set
indexes.indexes_training = indexes.indexes_rand(1:10); %taking randomly 10 rows from dataset for training set 
indexes.indexes_test = indexes.indexes_rand(11:indexes.rows); %taking the rest of rows of dataset to compute test set
set.training = data_set(indexes.indexes_training,:); %create matrix for training with all columns (attribute + target) of dataset 
set.test = data_set(indexes.indexes_test, 1:indexes.columns - 1); %create matrix for testing with only attribute columns (without target column) 
set.real_target_test = data_set(indexes.indexes_test, indexes.columns) %a vector column with just the real target of the data set 

%Consider that the num of rows of data matrix corrisponds to the num of
%instances, instead the num of columns correspond to the num of predictors
%(attribute + target) 


      
    
    

end

% EXAMPLE
% data = (1:100)*10;
% indexes=randperm(length(data)); %rand permutation of numbers 1 to 100
% data(indexes); % data ordered randomly according to the indexes
% indexes_subset = indexes(1:30); %if we want 30 randomly chosen values
% random_subset = data(indexes_subset);