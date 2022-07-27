%S4845876- Saivinay Manda
function [P_Laplace] = probability_Laplace (B, new_training, predictors, max_levels,  levels_for_each_input, levels_classes, num_appearance_of_each_level_class )
%% Compute Likelihoods P(x|c) 
a = 1; 
P_Laplace = zeros(predictors, max_levels, levels_classes);
for i = 1:levels_classes
    for j = 1:predictors 
        for s=1:levels_for_each_input(j)
            P_Laplace(j,s,i) = (B(j,s,i) + a) /(num_appearance_of_each_level_class(i) + a*new_training(1,j)); 
        end
    end
end

            

end

