
clear; close; clc;

% define Rastrigin's function - as a local function (coppied from mathworks)
ras = @(x,y) 20 + x.^2 + y.^2 - 10 * (cos(2*pi*x) + cos(2*pi*y));
matyas = @(x,y) 0.26 * (x.^2 + y.^2) - 0.48 * x .* y
six_hump_camel = @(x,y) (4 - 2.1.*x.^2 + (x.^4)./3).*x.^2 + x.*y + (-4 + 4.*y.^2).*y.^2;


% plot the solution space
%objective_function = @(x,y) ras(x/10, y/10); % mathworks scales the function here - idk why
%objective_function = @(x,y) x.^2+y.^2; % rotated hyper elipsoid
%objective_function = @(x,y) matyas(x,y)
%objective_function = @(x,y) six_hump_camel(x,y)

function_types = ["many_local_minima","bowl_shaped", "steep_ridges_and_drops", "plate_shaped", "valley_shaped", "other"];

for i = 1:length(function_types)
    function_folder_path = "optimization_test_functions/" + function_types(i)

    addpath(function_folder_path)
    [objective_functions, function_names] = create_objective_functions(function_folder_path)

    % % for debug
    % test_point = [1, 2]; % Example test point
    %
    % % test successful calling of the functions
    % for i = 1:length(objective_functions)
    %     try
    %         result = objective_functions{i}(test_point);
    %         fprintf('Function %d result: %f\n', i, result);
    %     catch ME
    %         fprintf('Error with function %d: %s\n', i, ME.message);
    %     end
    % end

    for j = 1:length(objective_functions)
        disp(function_names{j})
        objective_function = @(x,y) arrayfun(@(x,y) objective_functions{j}([x,y]), x,y);

        fsurf(objective_function, [-3, 3], "ShowContours", "on");
        title("Objective function");
        xlabel("x");
        ylabel("y");



        stats = evaluate_optimizers(objective_function);

        % Display the results
        disp(stats)
    end
end

function [objective_functions, function_names] = create_objective_functions(folder_path)
    % Add the folder to path
    addpath(folder_path);
    
    % Get all .m files in the folder
    files = dir(fullfile(folder_path, '*.m'));
    
    % Initialize cell array to store function handles
    objective_functions = {};
    function_names = {};
    
    % Loop through each file
    for i = 1:length(files)
        [~, func_name, ~] = fileparts(files(i).name);
        
        % Create function handle that takes [x,y] as input
        % Assumes your functions take a vector input
        objective_functions{i} = @(xy) feval(func_name, xy);
        function_names{i} = func_name;
    end
    
    % Display available functions
    fprintf('Created %d objective functions:\n', length(objective_functions));
    for i = 1:length(function_names)
        fprintf('%d: %s\n', i, function_names{i});
    end
end


function results = test_optimizer(solver, objective_function, x0, bounds, optim_opts)
    % design variables
    if all(isinf(bounds),"all") == false % are design variables bounded?
        x = optimvar("x", "LowerBound",bounds(1,1), "UpperBound",bounds(1,2));
        y = optimvar("y", "LowerBound",bounds(2,1), "UpperBound",bounds(2,2));
    else % design variables are unbounded
        x = optimvar("x");
        y = optimvar("y");
    end
    
    % define the objectove funtion
    prob = optimproblem("Objective", objective_function(x,y));
    
    % reset random generator for consitent results from stochastic optimizers
    rng default;
    
    % run the optimizer (and store the results)
    [sol, fval, eflag, output] = solve(prob, x0,"Solver", solver, "Options", optim_opts);
    
    % Store results
    results = struct( ...
        "sol", sol, ...
        "fval", fval, ...
        "exitflag", eflag, ...
        "output", output);
    
    % check that it has actually converged
    if results.exitflag ~= 1 % did we fail to find a solution?
        warning("solver %s failed to converge", solver) % warn the user that convergence was not achived
    end
end

function func_count = extract_function_count(output_struct, method_name)
    % Extract function evaluations (handle different field names and hybrid methods)
    if isfield(output_struct, 'total_funccount')
        % This is a hybrid method with summed function counts
        func_count = output_struct.total_funccount;
    elseif isfield(output_struct, 'funcCount')
        func_count = output_struct.funcCount;
    elseif isfield(output_struct, 'funccount')
        func_count = output_struct.funccount;
    else
        % Some optimizers might have different field names
        warning('Function count field not found for %s', method_name);
        func_count = NaN;
    end
end

function stats = evaluate_optimizers(objective_function)
% dictionary to store results
% optim_results = configureDictionary("string", "cell"); % with r2023b and later
optim_results = dictionary(string.empty, cell.empty); % r2023a and eariler

% dictionarry to store the inputs
inputs = dictionary(string.empty, cell.empty); % r2023a and eariler
optimizers = ["fminunc", "patternsearch", "ga", "particleswarm", "simulannealbnd", "surrogateopt"];

% default intial condition
x0.x = 20;
x0.y = 30;

% configure default solver parameters
for optimizer = optimizers
    inputs{optimizer} = ...
        struct( ...
        "initial_condition", x0, ...
        "options", optimoptions(optimizer), ...
        "bounds", [-Inf, Inf; -Inf, Inf],...
        "Display", "none"...
        );
end

% configure the differing parameters for each solver

% define GA initial population: w/ start near [20, 30]
popSize = 20;
x_init = 10 * randn(popSize, 1) + 20;
y_init = 10 * randn(popSize, 1) + 30;
initialPop = [x_init, y_init];  % 20×2 matrix
% - Note the mathworks example creates an initial population but never
% passes it to the solver

% Set up options with custom initial population 
inputs{"ga"}.options = optimoptions("ga", ...
    "PopulationSize", popSize, ...
    "InitialPopulationMatrix", initialPop);

% surrogate optimiztion
inputs{"surrogateopt"}.bounds = [-70, 130; -70, 130]; % requires bounded design variables
inputs{"surrogateopt"}.options = optimoptions("surrogateopt","PlotFcn",[]); % disable plotting

% test the solvers
for solver = inputs.keys' % transopse to get a row vector for itteration
    % test the optimizer
    results = test_optimizer(solver, objective_function, inputs{solver}.initial_condition, inputs{solver}.bounds, inputs{solver}.options);
    % Store results
    optim_results{solver} = results;
end
hybrid_base_methods = ["ga", "simulannealbnd"];

for base_method = hybrid_base_methods
    hybrid_name = base_method + "+fminunc";
    
    % Use the final solution from the base method as initial guess for fminunc
    hybrid_results = test_optimizer("fminunc", objective_function, optim_results{base_method}.sol, ...
        inputs{"fminunc"}.bounds, inputs{"fminunc"}.options);
    
    % Get base method function evaluations
    base_results = optim_results{base_method};
    base_fevals = extract_function_count(base_results.output, base_method);
    
    % Get hybrid method function evaluations  
    hybrid_fevals = extract_function_count(hybrid_results.output, hybrid_name);
    
    % Sum the function evaluations for the complete hybrid process
    hybrid_results.output.total_funccount = base_fevals + hybrid_fevals;
    
    % Store hybrid results in dictionary
    optim_results{hybrid_name} = hybrid_results;
end

% Create extended list of all methods (original + hybrid)
hybrid_method_names = hybrid_base_methods + "+fminunc";
all_methods = [optimizers, hybrid_method_names];

% Initialize arrays for table data
sols = zeros(length(all_methods), 2);
fvals = zeros(length(all_methods), 1);
fevals = zeros(length(all_methods), 1);

% Extract data from dictionary
for i = 1:length(all_methods)
    solver = all_methods(i);
    results = optim_results{solver};
    
    % Extract solution values
    sols(i, :) = [results.sol.x, results.sol.y];
    
    % Extract objective function value
    fvals(i) = results.fval;
    
    % Extract function evaluations using local helper function
    fevals(i) = extract_function_count(results.output, solver);
end

% Create and format the table
stats = table(sols, fvals, fevals);
stats.Properties.RowNames = all_methods;
stats.Properties.VariableNames = ["Solution" "Objective" "# Fevals"];
end

