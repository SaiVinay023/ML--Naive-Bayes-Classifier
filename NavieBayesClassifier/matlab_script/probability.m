%Saivinay Manda - S4845876
function [B, num, Target, P] = probability(training_set)
[num.rows, num.columns] = size(training_set); 
%% Compute prior probability of class: P(c)
Target= training_set(:,num.columns);
%Calculate how many levels are there for target attribute
A = unique(Target); 
num.levels_classes = size(A,1); %num of levels of target attribute %2

%Compute how many level i is present
num.appearance_of_each_level_class = zeros(1,num.levels_classes);
for j = 1:num.levels_classes
    
    for i = (Target)'
            if (i == j)
                num.appearance_of_each_level_class(j) = num.appearance_of_each_level_class(j) + 1; 
            end 
    end
end
%num.ofeach_levels_in_the_attribute_class 
%num of each level i for the training step 
P.prior_probability_class = zeros(1,num.levels_classes); 
for j=1:num.levels_classes
    P.prior_probability_class(j) = num.appearance_of_each_level_class(j)/num.rows; 
end
%P.prior_probability_class


%% Compute Likelihoods P(x|c) 
% calculate num of columns - 1 is the num of predictors
num.predictors = num.columns - 1 ; 



%finding vector rows for each possible level and predictors

num.levels_for_each_input = zeros(1, num.predictors); 
for i = 1:num.predictors
    b = training_set(:,i); 
    a = unique(b); 
    num.levels_for_each_input(i) = length(a);    
end
%number of levels for each input 

%vector row with number of levels nd each predictor 

num.max_levels = max(num.levels_for_each_input); % we have 3 levels

% Here we are inizializiation of a null matrix (3D)
B = zeros(num.predictors, num.max_levels, num.levels_classes); 

%compute how many class levels  i (of target attribute) associated with a

%possible level s of the predictor j 


for e = 1:num.rows %here we are taking number of instances
    i = Target(e); %can be 1 or 2 which is (yes or no) 
    for j = 1:num.predictors 
        for s = 1:num.levels_for_each_input(j)
            if training_set(e,j) == s
                B(j,s,i) = B(j,s,i) + 1; 
            end
        end
    end
end
%B


P.likelihoods = zeros(num.predictors, num.max_levels, num.levels_classes);
for i = 1:num.levels_classes
    for j = 1:num.predictors 
        for s=1:num.levels_for_each_input(j)
            P.likelihoods(j,s,i) = B(j,s,i)/num.appearance_of_each_level_class(i); 
        end
    end
end
%P.likelihoods 
            


end




         
    
  
       


    
    


            
        
        
 



    


   


    








