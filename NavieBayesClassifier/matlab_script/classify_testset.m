%S4845876- Saivinay Manda
function [num_t, P_t] = classify_testset(test_set, num_levels_target, likelihood_matrix, prior_probab_classes)
[num_t.rows, num_t.columns] = size(test_set); %num_t.rows = num of instances; num_t.columns = num of predictors (=4) 
%% Compute the predicted probabilities of each predictor's level of the test set based on each target's level: P(x1=s|c)
P_t.PredictedProbability = zeros(num_t.rows, num_t.columns, num_levels_target); 
for i=1:num_levels_target 
    for e = 1:num_t.rows %num of instances 
        for j = 1:num_t.columns %num_t.columns = num. predictors 
            s = test_set(e,j); 
            P_t.PredictedProbability(e,j,i) = P_t.PredictedProbability(e,j,i) + likelihood_matrix(j,s,i); 
        end
    end
end

%% Compute the likelihoods oh each instance based on each target's level : P(x|c) = P(x_1=s1|c)*P(x_2=s2|c)*..*P(x_j=s3|c)
%NOTA: we have to compute the Likelihoods of each attribute in this way because we're
%supposing that all predictors are indipendent with each other. 


 
P_t.Likelihoods_Test =  prod(P_t.PredictedProbability,2);



%% Compute the Posterior Probability for each target's level: P(c|x) = P(x|c)*P(c)/P(x) --> Bayes Theorem
%where: P(x|c) is the likelihoods, P(c) is the prior probability of each
%target level, P(x) is the prior probability of each attribute 


%NOTA: We can avoid to compute P(x)=prior probability of each predictor
%because they are indipendent with each other, indeed: 
%P(x) = P(x1)*P(x2)*...*P(x_j) = 1/100 always 

%therefore P(c|x) = P(x|c) * P(c) 
P_t.PosteriorProbability = zeros(num_t.rows, num_levels_target);
for i = 1:num_levels_target
    P_t.PosteriorProbability(:,i) = (P_t.PosteriorProbability(:,i) + [P_t.Likelihoods_Test(:,:,i)*prior_probab_classes(1,i)])/0.01; 
end 
    
end

