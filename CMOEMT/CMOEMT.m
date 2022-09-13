classdef CMOEMT < ALGORITHM
    % <multi> <real> <constrained>
    % EMT and knowledge transfer based CMOEA
    % delta --- 0.9 --- The probability of choosing parents locally
    % type --- 1 --- Type of operator (1. GA 2. DE)
    methods
        function main(Algorithm,Problem)
            
            %% Parameter setting
            [delta,type] = Algorithm.ParameterSet(0.9,1);
            
            %% Generate the weight vectors
            [W,Problem.N] = UniformPoint(Problem.N,Problem.M);
            T = ceil(Problem.N/10);
            
            %% Detect the neighbours of each solution
            B = pdist2(W,W);
            [~,B] = sort(B,2);
            B = B(:,1:T);
            
            %% Generate random population
            Population{1} = Problem.Initialization();
            Population{2} = Problem.Initialization();
            Z             = min(Population{2}.objs,[],1);
            Population{3} = Problem.Initialization();
            LengthO{1} = Problem.N/2; LengthO{2} = Problem.N/2; LengthO{3} = Problem.N/2;
            Fitness{1}    = CalFitness(Population{1}.objs,Population{1}.cons);
            Fitness{3}    = CalFitness(Population{3}.objs);
            
            %% Evaluate the Population
            Tc               = 0.9 * ceil(Problem.maxFE/Problem.N);
            last_gen         = 20;
            change_threshold = 1e-1;
            search_stage     = 1; % 1 for push stage,otherwise,it is in pull stage.
            max_change       = 1;
            epsilon_k        = 0;
            epsilon_0        = 0;
            cp               = 2;
            alpha1            = 0.95;
            tao              = 0.05;
            ideal_points     = zeros(ceil(Problem.maxFE/Problem.N),Problem.M);
            nadir_points     = zeros(ceil(Problem.maxFE/Problem.N),Problem.M);
            
            %% Optimization
            while Algorithm.NotTerminated(Population{1})
                gen        = ceil(Problem.FE/(2*Problem.N));
                pop_cons   = Population{2}.cons;
                cv         = overall_cv(pop_cons);
                population = [Population{2}.decs,Population{2}.objs,cv];
                rf         = sum(cv <= 1e-6) / Problem.N;
                ideal_points(gen,:) = Z;
                nadir_points(gen,:) = max(population(:,Problem.D + 1 : Problem.D + Problem.M),[],1);
                
                % The maximumrate of change of ideal and nadir points rk is calculated.
                if gen >= last_gen
                    max_change = calc_maxchange(ideal_points,nadir_points,gen,last_gen);
                end
                
                % The value of e(k) and the search strategy are set.
                if gen < Tc
                    if max_change <= change_threshold && search_stage == 1
                        search_stage = -1;
                        epsilon_0 = max(population(:,end),[],1);
                        epsilon_k = epsilon_0;
                    end
                    if search_stage == -1
                        epsilon_k =  update_epsilon(tao,epsilon_k,epsilon_0,rf,alpha1,gen,Tc,cp);
                    end
                else
                    epsilon_k = 0;
                end
                
                if Problem.FE < Problem.maxFE/2
                    %non transfer
                    
                  %% Offspring Reproduction
                    if type == 1
                        MatingPool{1} = TournamentSelection(2,2*LengthO{1},Fitness{1});
                        Offspring{1}  = OperatorGAhalf(Population{1}(MatingPool{1}));
                    else
                        MatingPool{1} = TournamentSelection(2,2*Problem.N,Fitness{1});
                        Offspring{1}  = OperatorDE(Population{1},Population{1}(MatingPool{1}(1:end/2)),Population{1}(MatingPool{1}(end/2+1:end)));
                    end
                    
                    Offspring{2} = [];
                    for subgeneration = 1 : 5
                        Bounday = find(sum(W<1e-3,2)==Problem.M-1)';
                        Bounday = [Bounday,floor(length(W)/2)];
                        I = [Bounday,randi(length(W),1,floor(Problem.N/5)-length(Bounday))];
                        for j = 1 : length(I)
                            i = I(j);
                            
                            %                    for i = 1 : Problem.N
                            % Choose the parents
                            if rand < delta
                                P = B(i,randperm(size(B,2)));
                            else
                                P = randperm(Problem.N);
                            end
                            
                            % Generate an offspring
                            offspring = OperatorDE(Population{2}(i),Population{2}(P(1)),Population{2}(P(2)));
                            Offspring{2} = [Offspring{2},offspring];

                        end
                    end
                    Z = min(Z,min(Offspring{2}.objs,[],1));
                    
                    if type == 1
                        MatingPool{3} = TournamentSelection(2,2*LengthO{3},Fitness{3});
                        Offspring{3}  = OperatorGAhalf(Population{3}(MatingPool{3}));
                    else
                        MatingPool{3} = TournamentSelection(2,2*Problem.N,Fitness{3});
                        Offspring{3}  = OperatorDE(Population{3},Population{3}(MatingPool{3}(1:end/2)),Population{3}(MatingPool{3}(end/2+1:end)));
                    end
                    
                  %% Environmental Selection
                    [Population{1},Fitness{1},~] = EnvironmentalSelectionT1([Population{1},Offspring{1:3}],Problem.N);    
                    [Population{2},Fitness{2},~] = EnvironmentalSelectionT2([Population{2},Offspring{1:3}],Problem.N,epsilon_k);
                    [Population{3},Fitness{3},~] = EnvironmentalSelectionT3([Population{3},Offspring{1:3}],Problem.N);
                    
                else
                    %transfer
                    
                  %% Offspring Reproduction
                    if type == 1
                        MatingPool{1} = TournamentSelection(2,2*LengthO{1},Fitness{1});
                        Offspring{1}  = OperatorGAhalf(Population{1}(MatingPool{1}));
                    else
                        MatingPool{1} = TournamentSelection(2,2*Problem.N,Fitness{1});
                        Offspring{1}  = OperatorDE(Population{1},Population{1}(MatingPool{1}(1:end/2)),Population{1}(MatingPool{1}(end/2+1:end)));
                    end
                    
                    Offspring{2}  = [];
                    for subgeneration = 1 : 5
                        Bounday = find(sum(W<1e-3,2)==Problem.M-1)';
                        Bounday = [Bounday,floor(length(W)/2)];
                        I = [Bounday,randi(length(W),1,floor(Problem.N/5)-length(Bounday))];
                        for j = 1 : length(I)
                            i = I(j);
                            
                            %                    for i = 1 : Problem.N
                            % Choose the parents
                            if rand < delta
                                P = B(i,randperm(size(B,2)));
                            else
                                P = randperm(Problem.N);
                            end
                            
                            % Generate an offspring
                            offspring = OperatorDE(Population{2}(i),Population{2}(P(1)),Population{2}(P(2)));
                            Offspring{2} = [Offspring{2},offspring];
                            
                        end
                    end
                    Z = min(Z,min(Offspring{2}.objs,[],1));
                    
                    if type == 1
                        MatingPool{3} = TournamentSelection(2,2*LengthO{3},Fitness{3});
                        Offspring{3}  = OperatorGAhalf(Population{3}(MatingPool{3}));
                    else
                        MatingPool{3} = TournamentSelection(2,2*Problem.N,Fitness{3});
                        Offspring{3}  = OperatorDE(Population{3},Population{3}(MatingPool{3}(1:end/2)),Population{3}(MatingPool{3}(end/2+1:end)));
                    end
                    
                  %% Environmental selection
                    % for Task1
                    [~,~,Next1] = EnvironmentalSelectionT1([Population{1},Offspring{1},Population{2},Offspring{2},Population{3},Offspring{3}],Problem.N);
                    succ_num1_2 =  (sum(Next1(length(Population{1})+length(Offspring{1})+1:length(Population{1})+length(Offspring{1})+length(Population{2})+length(Offspring{2}))));
                    succ_num1_3 =  (sum(Next1(length(Population{1})+length(Offspring{1})+length(Population{2})+length(Offspring{2})+1:end)));
                    succ_num1 = [succ_num1_2,succ_num1_3];
                    [~,best_task1] = max(succ_num1);
                    [Population{1},Fitness{1},~] = EnvironmentalSelectionT1([Population{1},Offspring{1},Population{best_task1+1},Offspring{best_task1+1}],Problem.N);
                    
                    % for Task2
                    [~,~,Next1] = EnvironmentalSelectionT2([Population{2},Offspring{2},Population{1},Offspring{1},Population{3},Offspring{3}],Problem.N,epsilon_k);
                    succ_num2_1 =  (sum(Next1(length(Population{2})+length(Offspring{2})+1:length(Population{2})+length(Offspring{2})+length(Population{1})+length(Offspring{1}))));
                    succ_num2_3 =  (sum(Next1(length(Population{1})+length(Offspring{1})+length(Population{2})+length(Offspring{2})+1:end)));
                    succ_num2 = [succ_num2_1,succ_num2_3];
                    [~,best_task2] = max(succ_num2);
                    [Population{2},Fitness{2},~] = EnvironmentalSelectionT2([Population{1},Offspring{1},Population{best_task2+1},Offspring{best_task2+1}],Problem.N,epsilon_k);
                    
                    % for Task3
                    [~,~,Next1] = EnvironmentalSelectionT3([Population{3},Offspring{3},Population{1},Offspring{1},Population{2},Offspring{2}],Problem.N);
                    succ_num3_1 =  (sum(Next1(length(Population{3})+length(Offspring{3})+1:length(Population{3})+length(Offspring{3})+length(Population{1})+length(Offspring{1}))));
                    succ_num3_2 =  (sum(Next1(length(Population{3})+length(Offspring{3})+length(Population{1})+length(Offspring{1})+1:end)));
                    succ_num3 = [succ_num3_1,succ_num3_2];
                    [~,best_task3] = max(succ_num3);
                    [Population{3},Fitness{3},~] = EnvironmentalSelectionT3([Population{3},Offspring{3},Population{best_task3+1},Offspring{best_task3+1}],Problem.N);
                    
                end
            end
        end
    end
end

% The Overall Constraint Violation
function result = overall_cv(cv)
    cv(cv <= 0) = 0;cv = abs(cv);
    result = sum(cv,2);
end

% Calculate the Maximum Rate of Change
function max_change = calc_maxchange(ideal_points,nadir_points,gen,last_gen)
    delta_value = 1e-6 * ones(1,size(ideal_points,2));
    rz = abs((ideal_points(gen,:) - ideal_points(gen - last_gen + 1,:)) ./ max(ideal_points(gen - last_gen + 1,:),delta_value));
    nrz = abs((nadir_points(gen,:) - nadir_points(gen - last_gen + 1,:)) ./ max(nadir_points(gen - last_gen + 1,:),delta_value));
    max_change = max([rz, nrz]);
end

function result = update_epsilon(tao,epsilon_k,epsilon_0,rf,alpha,gen,Tc,cp)
    if rf < alpha
        result = (1 - tao) * epsilon_k;
    else
        result = epsilon_0 * ((1 - (gen / Tc)) ^ cp);
    end
end