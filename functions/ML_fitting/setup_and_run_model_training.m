function SANDIinput = setup_and_run_model_training(SANDIinput)

% Main script to setup and train the Random
% Forest (RF) or multi-layers perceptron (MLP) regressors used to fit the
% SANDI model
%
% Author:
% Dr. Marco Palombo
% Cardiff University Brain Research Imaging Centre (CUBRIC)
% Cardiff University, UK
% 4th August 2022
% Email: palombom@cardiff.ac.uk

% Read the input
delta = SANDIinput.delta ;
smalldel = SANDIinput.smalldel ;
sigma_SHresiduals = SANDIinput.sigma_SHresiduals;
sigma_mppca = SANDIinput.sigma_mppca;
output_folder = SANDIinput.output_folder;
Dsoma = SANDIinput.Dsoma;
Din_UB = SANDIinput.Din_UB ;
Rsoma_UB = SANDIinput.Rsoma_UB ;
De_UB = SANDIinput.De_UB ;
seed = SANDIinput.seed ;
MLmodel = SANDIinput.MLmodel ;
FittingMethod = SANDIinput.FittingMethod ;
diagnostics = SANDIinput.diagnostics ;
DoTestPerformances = SANDIinput.DoTestPerformances;

rng(seed) % for reproducibility

if ~isempty(output_folder)
    mkdir(output_folder) % create the output folder
end

if isempty(Dsoma)
    Dsoma = 3;
    SANDIinput.Dsoma = Dsoma;
end
if isempty(Din_UB)
    Din_UB = 3;
    SANDIinput.Din_UB = Din_UB;
end
if isempty(Rsoma_UB) || Rsoma_UB==0
    fsphere = @(p,x) my_murdaycotts(delta,smalldel,p,Dsoma,x);
    Rsoma_UB = fminunc(@(p) abs(1 - fsphere(p,1)), 10); % Authomatically set the Rsoma to a max value given the diffusion time and the set Dsoma so to not exceed an apparent diffusivity = 1. This threshold has been chosen given realistic soma sizes and extracellular diffusivities.
    SANDIinput.Rsoma_UB = Rsoma_UB;
end
if isempty(De_UB)
    De_UB = 3;
    SANDIinput.De_UB = De_UB;
end

if isempty(FittingMethod)
    FittingMethod=0;
end

%% Build the model and the corresponding training set

% Build the SANDI model to fit (as in the Neuroimage 2020) sampling the signal fraction from a Dirichlet distribution, to guarantee that they sum up to 1 and cover uniformely the area of the triangle defined by them

disp('* Training will be performed with Dirichlet sampling for the signal fractions and uniform sampling for the other parameters within the ranges:')
fprintf(SANDIinput.LogFileID,'* Training will be performed with Dirichlet sampling for the signal fractions and uniform sampling for the other parameters within the ranges:\n');

model = struct;
model.Nset = SANDIinput.Nset; % After some testing, between 1e4 and 1e5 is large enough to converge to satisfactory performances. By default we choose 1e4 to keep the training faster, but it can and should be changed according to the dataset.

% Sample sigma for Gaussian noise from the sigma distribution from SH residuals

Nsamples = numel(sigma_SHresiduals);
sigma_SHresiduals_sampled = sigma_SHresiduals(randi(Nsamples,[model.Nset 1]));
sigma_SHresiduals = sigma_SHresiduals_sampled;

% If a noisemap from MPPCA denoising is provided, it will use the distribution of noise variances within the volume to add Rician noise floor bias to train the model. If a noisemap is not provided, it will use the user defined SNR.

Nsamples = numel(sigma_mppca);
sigma_mppca_sampled = sigma_mppca(randi(Nsamples,[model.Nset 1]));
sigma_mppca = sigma_mppca_sampled;

if diagnostics==1

    %r = report_generator([], fullfile(SANDIinput.StudyMainFolder,'Report_ML_Training_Performance'));
    %r.open();

    h = figure('Name','Distribution of noise variances');
    subplot(1,2,1)
    histogram(sigma_mppca_sampled, 100), title(['MPPCA (SNR~' num2str(round(1/nanmedian(sigma_mppca_sampled))) ')']), hold on
    subplot(1,2,2)
    histogram(sigma_SHresiduals_sampled, 100), title(['SH-fit (SNR~' num2str(round(1/nanmedian(sigma_SHresiduals_sampled))) ')']), hold on

    T = getframe(h);

    mkdir(fullfile(SANDIinput.StudyMainFolder, 'Report_ML_Training_Performance'));
    imwrite(T.cdata, fullfile(SANDIinput.StudyMainFolder , 'Report_ML_Training_Performance', 'Distribution_of_noise_variances.tiff'))

    %r.section('Distribution of Noise Variances');
    %r.add_text('The plot shows the distribution of noise variances used to inject noise to simulated noiseless signals for training.');
    %r.add_figure(gcf,['Distribution of Noise Variances. [LEFT] Noise variance distribution as estimated from MPPCA denoising from the raw data. Median SNR is ' num2str(1/nanmedian(sigma_mppca_sampled)) '. It is used to sample the variance to add the Rician floor bias. [RIGHT] Noise variance distribution as estimated from the SH fit of the fully processed data. Median SNR is ' num2str(1/nanmedian(sigma_SHresiduals_sampled)) '. It is used to add Gaussian noise on top of Rician bias signals, for each acquired direction.'],'left');
    %r.end_section();

    savefig(h,fullfile(SANDIinput.StudyMainFolder, 'Report_ML_Training_Performance','Distribution_of_noise_variances.fig'));
    close(h);
end

model.sigma_mppca = sigma_mppca;
model.sigma_SHresiduals = sigma_SHresiduals;

model.delta = delta;
model.smalldel = smalldel;
model.Dsoma = Dsoma;
model.paramsrange = [0 1; 0 1; Dsoma/6 Din_UB; 1 Rsoma_UB; Dsoma/6 De_UB];
model.Nparams = size(model.paramsrange,1);
model.boost = FittingMethod;

disp(['   - Dsoma = ' num2str(Dsoma) ' um^2/ms'])
fprintf(SANDIinput.LogFileID,['   - Dsoma = ' num2str(Dsoma) ' um^2/ms\n']);

disp(['   - Rsoma = [' num2str(model.paramsrange(4,1)) ', ' num2str(model.paramsrange(4,2)) '] um'])
fprintf(SANDIinput.LogFileID,['   - Rsoma = [' num2str(model.paramsrange(4,1)) ', ' num2str(model.paramsrange(4,2)) '] um\n']);

disp(['   - De = [' num2str(model.paramsrange(5,1)) ', ' num2str(model.paramsrange(5,2)) '] um^2/ms'])
fprintf(SANDIinput.LogFileID,['   - De = [' num2str(model.paramsrange(5,1)) ', ' num2str(model.paramsrange(5,2)) '] um^2/ms\n']);

disp(['   - Din = [' num2str(model.paramsrange(3,1)) ', ' num2str(model.paramsrange(3,2)) '] um^2/ms'])
fprintf(SANDIinput.LogFileID,['   - Din = [' num2str(model.paramsrange(3,1)) ', ' num2str(model.paramsrange(3,2)) '] um^2/ms\n']);

if ~isempty(output_folder), save(fullfile(output_folder, 'model.mat'), 'model'); end % Save the model used

% Build the training set
SANDIinput.model = model;
disp(['* Building training set: #samples = ' num2str(model.Nset)])
fprintf(SANDIinput.LogFileID,['* Building training set: #samples = ' num2str(model.Nset) '\n']);

[~, database_train_noisy, params_train, sigma_mppca, bvals, Ndirs_per_Shell] = build_training_set(SANDIinput);
model.bvals = bvals;
model.Ndirs_per_Shell = Ndirs_per_Shell;

SANDIinput.bunique = bvals;
SANDIinput.Ndirs_per_Shell = Ndirs_per_Shell;

% Define the SANDI model function to be used for the NLLS fitting. For the signal fractions we are using a
% transformation to ensure sum to unity: fstick = cos(p(1)).^2 and fsphere
% = (1-cos(p(1)).^2).*cos(p(2)).^2.

fellipse = @(p,x) exp(-x.*p(2)).*sqrt(pi./(4.*x.*(p(1)-p(2)))).*erf(sqrt(x.*(p(1)-p(2))));
fstick = @(p,x) fellipse([p,0],x);
fsphere = @(p,x) exp(-my_murdaycotts(delta,smalldel,p,Dsoma,x) );
fball = @(p,x) exp(-p.*x);

ffit = @(p,x) cos(p(1)).^2.*fstick(p(3),x) + ...
    (1-cos(p(1)).^2).*cos(p(2)).^2.*fsphere(p(4),x) + ...
    (1 - cos(p(1)).^2 - (1-cos(p(1)).^2).*cos(p(2)).^2).*fball(p(5),x);

model.fmodel = ffit;

SANDIinput.model = model;

f_rm = @(p,x) RiceMean(ffit([p(1),p(2),p(3),p(4),p(5)],x),p(6));

if model.boost
    tic

    options = optimset('Display', 'off');

    lb = model.paramsrange(:,1);
    ub = model.paramsrange(:,2);

    ub(1:2) = [pi/2 pi/2];

    xini = params_train;
    xini(:,1) = acos(sqrt(params_train(:,1)));
    xini(:,2) = acos(sqrt(params_train(:,2)./(1-params_train(:,1))));

    bvals(bvals==0) = 1E-6;

    disp('* Fitting noisy simulated signals using NLLS estimation for ground truth model parameters in case of Rician noise')
    fprintf(SANDIinput.LogFileID,'* Fitting noisy simulated signals using NLLS estimation for ground truth model parameters in case of Rician noise\n');

    params_train_NLLS = zeros(size(params_train,1), size(params_train,2));

    parfor i = 1:size(database_train_noisy,1)
        % Fixing the sigma using the function RiceMean
        params_train_NLLS(i,:) = lsqcurvefit(@(p,x) f_rm([p(1), p(2), p(3), p(4), p(5), sigma_mppca(i)],x), xini(i,:), bvals, database_train_noisy(i,:), lb, ub, options);
        % Fixing the sigma using the funciton RicianLogLik
        %params_train_NLLS(i,:) = fmincon(@(p) -RicianLogLik(database_train_noisy(i,:)', ffit(p,bvals)', sigma_mppca(i)), xini(i,:), [], [] , [],[], lb, ub,[], options);

    end

    f1 = cos(params_train_NLLS(:,1)).^2;
    f2 = (1-cos(params_train_NLLS(:,1)).^2).*cos(params_train_NLLS(:,2)).^2;

    params_train_NLLS(:,1) = f1;
    params_train_NLLS(:,2) = f2;

    params_train = params_train_NLLS(:,1:5);

    tt = toc;

    disp(['DONE - NLLS estimation took ' num2str(round(tt)) ' sec.'])
    fprintf(SANDIinput.LogFileID,['DONE - NLLS estimation took ' num2str(round(tt)) ' sec.\n']);
end

if isempty(MLmodel), MLmodel = 'RF'; end

switch MLmodel

    case 'RF'
        %% RF train

        % --- Using Matlab
        n_trees = 200;
        disp(['* Training a Random Forest with ' num2str(n_trees) ' trees for each model parameter...'])
        fprintf(SANDIinput.LogFileID,['* Training a Random Forest with ' num2str(n_trees) ' trees for each model parameter...\n']);

        trainedML = train_RF_matlab((database_train_noisy), params_train, n_trees, SANDIinput);
        train_perf = cell(5,1);
        Mdl = trainedML.Mdl;
        for i=1:5
            train_perf{i} = oobError(Mdl{i});
        end

        % save([output_folder '/trained_RFmodel.mat'], 'Mdl', '-v7.3')

    case 'MLP'
        %% MLP train

        % --- Using Matlab
        n_MLPs = 1; % NOTE: training will take n_MLPs times longer! Training is performed using n_MLPs randomly initiailized for each model parameter. The final prediciton is the average prediciton among the n_MLPS. This shuold mitigate issues with local minima during training according to the "wisdom of crowd" principle.
        n_layers = 3;
        n_units = 30; %3*min(size(database_train_noisy,1),size(database_train_noisy,2)); % Recommend network between 3 x number of b-shells and 5 x number of b-shells

        disp(['* Training ' num2str(n_MLPs) ' MLP(s) with ' num2str(n_layers) ' hidden layer(s) and ' num2str(n_units) ' units per layer for each model parameter...'])
        fprintf(SANDIinput.LogFileID,['* Training ' num2str(n_MLPs) ' MLP(s) with ' num2str(n_layers) ' hidden layer(s) and ' num2str(n_units) ' units per layer for each model parameter...\n']);

        trainedML = train_MLP_matlab(database_train_noisy, params_train, n_layers, n_units, n_MLPs, SANDIinput);
        train_perf = trainedML.training_performances;
        % save([output_folder '/trained_MLPmodel.mat'], 'Mdl', '-v7.3')

end

Mdl = trainedML.Mdl;

SANDIinput.Slope_train=trainedML.Slope;
SANDIinput.Intercept_train = trainedML.Intercept;

SANDIinput.model = model;
SANDIinput.Mdl = Mdl;
SANDIinput.trainedML = trainedML;
SANDIinput.database_train_noisy = database_train_noisy;
SANDIinput.params_train = params_train;
SANDIinput.train_perf = train_perf;

MLdebias = SANDIinput.MLdebias;

%% Compare performances of ML and NLLS fitting on unseen testdata
if DoTestPerformances==1

    rng(123);
    model.Nset = 2.5e3;
    SANDIinput.model = model;

    disp(['* Building testing set: #samples = ' num2str(model.Nset)])
    fprintf(SANDIinput.LogFileID,['* Building testing set: #samples = ' num2str(model.Nset) '\n']);
    [~, database_test_noisy, params_test, sigma_mppca_test, bvals] = build_training_set(SANDIinput);

    MLpredictions = zeros(size(database_test_noisy,1), size(params_test,2));
    switch MLmodel

        case 'RF'
            %% RF fit

            disp('Fitting using a Random Forest regressor implemented in matlab ')
            fprintf(SANDIinput.LogFileID,'Fitting using a Random Forest regressor implemented in matlab\n');
            % --- Using Matlab

            MLpredictions = apply_RF_matlab((database_test_noisy), trainedML, MLdebias);

        case 'MLP'

            %% MLP fit

            disp('Fitting using a MLP regressor implemented in matlab ')
            fprintf(SANDIinput.LogFileID,'Fitting using a MLP regressor implemented in matlab\n');
            % --- Using Matlab

            MLpredictions = apply_MLP_matlab(database_test_noisy, trainedML, MLdebias);

    end

    tic

    options = optimset('Display', 'off');

    lb = model.paramsrange(:,1);
    ub = model.paramsrange(:,2);

    ub(1:2) = [pi/2 pi/2];

    % Convert model parameters to fitting variables.
    xini0 = params_test;
    xini0(:,1) = acos(sqrt(params_test(:,1)));
    xini0(:,2) = acos(sqrt(params_test(:,2)./(1-params_test(:,1))));

    bvals(bvals==0) = 1E-6;

    xini = grid_search(database_test_noisy, ffit, bvals, lb, ub, 10);

    disp('* Fitting noisy simulated unseen signals using NLLS estimation for ground truth model parameters in case of Rician noise')
    fprintf(SANDIinput.LogFileID,'* Fitting noisy simulated unseen signals using NLLS estimation for ground truth model parameters in case of Rician noise\n');

    bestest_NLLS = zeros(size(params_test,1), size(params_test,2));
    bestest_NLLS_noisy = zeros(size(params_test,1), size(params_test,2));

    parfor i = 1:size(database_test_noisy,1)

        try
            % Assumes Rician noise and fix the sigma of noise
            bestest_NLLS(i,:) = lsqcurvefit(@(p,x) f_rm([p(1), p(2), p(3), p(4), p(5), sigma_mppca_test(i)],x), xini0(i,:), bvals, database_test_noisy(i,:), lb, ub, options);
            bestest_NLLS_noisy(i,:) = lsqcurvefit(@(p,x) f_rm([p(1), p(2), p(3), p(4), p(5), sigma_mppca_test(i)],x), xini(i,:), bvals, database_test_noisy(i,:), lb, ub, options);

        catch
            bestest_NLLS(i,:) = nan;
            bestest_NLLS_noisy(i,:) = nan;
        end
    end

    % Convert fitted variables to model parameters
    f1 = cos(bestest_NLLS(:,1)).^2;
    f2 = (1-cos(bestest_NLLS(:,1)).^2).*cos(bestest_NLLS(:,2)).^2;
    bestest_NLLS(:,1) = f1;
    bestest_NLLS(:,2) = f2;

    f1 = cos(bestest_NLLS_noisy(:,1)).^2;
    f2 = (1-cos(bestest_NLLS_noisy(:,1)).^2).*cos(bestest_NLLS_noisy(:,2)).^2;
    bestest_NLLS_noisy(:,1) = f1;
    bestest_NLLS_noisy(:,2) = f2;

    tt = toc;

    disp(['DONE - NLLS estimation took ' num2str(round(tt)) ' sec.'])
    fprintf(SANDIinput.LogFileID,['DONE - NLLS estimation took ' num2str(round(tt)) ' sec.\n']);
    % Calculate the MSEs
    mse_nlls = mean((bestest_NLLS(:,1:5) - params_test).^2,1);
    mse_nlls_noisy = mean((bestest_NLLS_noisy(:,1:5) - params_test).^2,1);
    mse_ml = mean((MLpredictions(:,1:5) - params_test).^2,1);

    try
        h = figure('Name','Model parameters cross-correlation');
        hold on
        corrplot(bestest_NLLS(:,1:5))

        T = getframe(h);
        imwrite(T.cdata, fullfile(SANDIinput.StudyMainFolder , 'Report_ML_Training_Performance', 'Model_parameters_crosscorrelation.tiff'))

        %r.section('Model Parameters Cross-Correlations');
        %r.add_text('The plot shows the estimated model parameters cross-correlations from noisy simulated signals using NLLS estimation for ground truth model parameters in case of Rician noise ');
        %r.add_figure(gcf,'Model parameters cross-correlation plot','left');
        %r.end_section();

        savefig(h,fullfile(SANDIinput.StudyMainFolder , 'Report_ML_Training_Performance', 'Model_parameters_crosscorrelation.fig'));
        close(h);
    catch
        warning('Unable to calculate model parameters cross-coprrelations! MAybe a function is missing. Please check line 302 in the function ''setup_and_run_model_training''')
        fprintf(SANDIinput.LogFileID,'WARNING! Unable to calculate model parameters cross-coprrelations! MAybe a function is missing. Please check line 302 in the function ''setup_and_run_model_training''\n');
    end

    % Calculate the rescaled fitting errors. They are rescaled to help
    % visualizing the fitting errors on the same plot

    max_values = model.paramsrange(:,2);
    error_nlls = (bestest_NLLS(:,1:5) - params_test)./repmat(max_values',[model.Nset 1]);
    error_nlls_noisy = (bestest_NLLS_noisy(:,1:5) - params_test)./repmat(max_values',[model.Nset 1]);
    error_ml = (MLpredictions(:,1:5) - params_test)./repmat(max_values',[model.Nset 1]);

    Rsq = zeros(size(Mdl,1), 3);
    Intercept = zeros(size(Mdl,1), 3);
    Slope = zeros(size(Mdl,1), 3);

    for i = 1:size(Mdl,1)

        X = params_test(:,i);
        Y = bestest_NLLS(:,i);

        %[Rsq(i,1),Slope(i,1),Intercept(i,1)] = regression(X',Y');
        beta = fitlm(X',Y');
        Slope(i,1) = beta.Coefficients.Estimate(2);
        Intercept(i,1) = beta.Coefficients.Estimate(1);
        Rsq(i,1) = beta.Rsquared.Ordinary;

        Y = bestest_NLLS_noisy(:,i);

        %[Rsq(i,2),Slope(i,2),Intercept(i,2)] = regression(X',Y');
        beta = fitlm(X',Y');
        Slope(i,2) = beta.Coefficients.Estimate(2);
        Intercept(i,2) = beta.Coefficients.Estimate(1);
        Rsq(i,2) = beta.Rsquared.Ordinary;

        Y = MLpredictions(:,i);

        %[Rsq(i,3),Slope(i,3),Intercept(i,3)] = regression(X',Y');
        beta = fitlm(X',Y');
        Slope(i,3) = beta.Coefficients.Estimate(2);
        Intercept(i,3) = beta.Coefficients.Estimate(1);
        Rsq(i,3) = beta.Rsquared.Ordinary;

    end

    NLLS_perf = struct;
    NLLS_perf.mse_nlls = mse_nlls;
    NLLS_perf.mse_nlls_noisy = mse_nlls_noisy;

    SANDIinput.NLLS_perf = NLLS_perf;

    % Plot results
    if diagnostics==1


        titles = {'fneurite', 'fsoma', 'Din', 'Rsoma', 'De'};

        h = figure('Name','Performance of Machine Learning estimation'); hold on

        subplot(1,3,1), hold on

        X = zeros(model.Nset,size(Mdl,1)*3);
        L = cell(1,size(Mdl,1)*3);
        C = cell(1,size(Mdl,1)*3);

        k = 1;

        for i=1:size(Mdl,1)

            X(:,k:k+2) = [error_nlls(:,i), error_nlls_noisy(:,i), error_ml(:,i)];

            L{k} = [titles{i} ' NLLS (GT)'];
            L{k+1} = [titles{i} ' NLLS (Grid Search)'];
            L{k+2} = [titles{i} ' ML'];

            C{k} = 'r';
            C{k+1} = 'y';
            C{k+2} = 'm';

            k = k+3;

        end

        boxplot(X, 'Labels',L, 'PlotStyle','compact', 'ColorGroup', C), grid on

        subplot(1,3,2), hold on

        Y = zeros(size(Mdl,1),3);
        L = cell(size(Mdl,1),3);

        for ii = 1:size(Mdl,1)

            for j=1:3
                Y(ii,j) = Rsq(ii,j);

                L{ii,j} = titles{ii};
            end

        end

        L = categorical(L);
        L = reordercats(L,titles);

        b = bar(L,Y);
        b(1).FaceColor = [1 0 0];
        b(2).FaceColor = [0 1 0];
        b(3).FaceColor = [0 0 1];

        try
            for kk = 1:3
                xtips1 = b(kk).XEndPoints;
                ytips1 = b(kk).YEndPoints;
                labels1 = string(round(b(kk).YData.*100)./100);
                text(xtips1,ytips1,labels1,'HorizontalAlignment','center','VerticalAlignment','bottom')
            end
        catch
        end
        ylim([0 1])

        subplot(1,3,3), hold on

        Y = zeros(size(Mdl,1),3);
        L = cell(size(Mdl,1),3);

        for ii = 1:size(Mdl,1)

            Y(ii,1) = sqrt(mse_nlls(ii));
            Y(ii,2) = sqrt(mse_nlls_noisy(ii));
            Y(ii,3) = sqrt(mse_ml(ii));

            for j=1:3
                L{ii,j} = titles{ii};
            end

        end

        L = categorical(L);
        L = reordercats(L,titles);

        b = bar(L,Y);
        b(1).FaceColor = [1 0 0];
        b(2).FaceColor = [0 1 0];
        b(3).FaceColor = [0 0 1];

        try
            for kk = 1:3
                xtips1 = b(kk).XEndPoints;
                ytips1 = b(kk).YEndPoints;
                labels1 = string(round(b(kk).YData.*100)./100);
                text(xtips1,ytips1,labels1,'HorizontalAlignment','center','VerticalAlignment','bottom')
            end
        catch
        end
        ylim([0 4.0])

        T = getframe(h);
        imwrite(T.cdata, fullfile(SANDIinput.StudyMainFolder , 'Report_ML_Training_Performance','Performance_of_Machine_Learning_estimation.tiff'));

        %r.section('General Performance of Machine Learning Estimation');
        %r.add_text('The plot shows a summary of the Machine Learning estimation performance , compared to NLLS when initialized using the known ground truth (GT) or using grid search (Grid Search). The results (GT) are representative of the ideal scenario, where a very good guess of the real values of the model parameters is known a priori. The results (Grid Search) represents instead a more realistic scenario, where we do not have a good guess for the model parameters and we use a grid search approach to initialize the NLLS.');
        %r.add_figure(gcf,'General Performance of Machine Learning Estimation. [RED] NLLS (GT); [GREEN] NLLS (Grid Search); [BLU] Machine Learning. [Panel 1] Fitting Error normalized by the upper bound values; [Panel 2] Adjusted R squared of linear fit; [Panel 3] Root Mean Squared Error normalized by the upper bound values','left');
        %r.end_section();


        savefig(h, fullfile(SANDIinput.StudyMainFolder , 'Report_ML_Training_Performance','Performance_of_Machine_Learning_estimation.fig'));
        close(h);

        % Plot Error as a function of parameter (bias and variance together)

        h = figure('Name','Error as a function of model parameter GT values (normalized by their max)'); hold on

        step_size = 0.1;
        lb_h = 0:step_size:1;
        ub_h = 0+step_size:step_size:1+step_size;

        for param_to_test = 1:5

            subplot(2,3,param_to_test), hold on, plot(0:1, zeros(size(0:1)), 'k-', 'LineWidth', 2.0),hold on


            TT_nlls = zeros(numel(lb_h),1);
            PQ_nlls = zeros(numel(lb_h),1);
            QQ_nlls = zeros(numel(lb_h),1);

            TT_nlls_noisy = zeros(numel(lb_h),1);
            PQ_nlls_noisy = zeros(numel(lb_h),1);
            QQ_nlls_noisy = zeros(numel(lb_h),1);

            TT_ml = zeros(numel(lb_h),1);
            PQ_ml = zeros(numel(lb_h),1);
            QQ_ml = zeros(numel(lb_h),1);


            Xplot = zeros(numel(lb_h),1);

            XX = params_test(:,param_to_test)./repmat(max_values(param_to_test)',[model.Nset 1]);

            for ii = 1:numel(lb_h)

                tmp_nlls = error_nlls(XX(:,1)>=lb_h(ii) & XX(:,1)<=ub_h(ii),param_to_test);
                tmp_nlls_noisy = error_nlls_noisy(XX(:,1)>=lb_h(ii) & XX(:,1)<=ub_h(ii),param_to_test);
                tmp_ml = error_ml(XX(:,1)>=lb_h(ii) & XX(:,1)<=ub_h(ii),param_to_test);

                q_nlls = quantile(tmp_nlls,[0.25, 0.50, 0.75]);
                TT_nlls(ii) = q_nlls(:,2);
                PQ_nlls(ii) = q_nlls(:,1);
                QQ_nlls(ii) = q_nlls(:,3);

                q_nlls_noisy = quantile(tmp_nlls_noisy,[0.25, 0.50, 0.75]);
                TT_nlls_noisy(ii) = q_nlls_noisy(:,2);
                PQ_nlls_noisy(ii) = q_nlls_noisy(:,1);
                QQ_nlls_noisy(ii) = q_nlls_noisy(:,3);

                q_ml = quantile(tmp_ml,[0.25, 0.50, 0.75]);
                TT_ml(ii) = q_ml(:,2);
                PQ_ml(ii) = q_ml(:,1);
                QQ_ml(ii) = q_ml(:,3);

                Xplot(ii) = (ub_h(ii)+lb_h(ii))/2;
            end

            errorbar(Xplot, TT_nlls, (QQ_nlls - PQ_nlls)./2, 'ro-', 'LineWidth', 2.0)
            errorbar(Xplot, TT_nlls_noisy, (QQ_nlls_noisy - PQ_nlls_noisy)./2, 'go-', 'LineWidth', 2.0)
            errorbar(Xplot, TT_ml, (QQ_ml - PQ_ml)./2, 'bo-', 'LineWidth', 2.0)

            axis([0 1 -1 1])

            title(titles{param_to_test})
            xlabel('Normalized Model Parameter')
            ylabel('Error: Prediction - GT')
        end

        T = getframe(h);
        imwrite(T.cdata, fullfile(SANDIinput.StudyMainFolder , 'Report_ML_Training_Performance', 'Error_vs_model_parameter_GT.tiff'));

        %r.section('Bias of the Machine Learning Estimation');
        %r.add_text('The plot shows the expected bias from the Machine LEarning estimation, compared to NLLS (GT) and NLLS (Grid Search). Ideally, we would like the bias to be 0 along the whole parameter axis (x-axis).');
        %r.add_figure(gcf,'Bias of the Machine Learning Estimation as a function of each model parameter normalized by the maximum value. [BLACK] Ideal case; [RED] NLLS (GT); [GREEN] NLLS (Grid Search); [BLU] Machine Learning','left');
        %r.end_section();

        savefig(h, fullfile(SANDIinput.StudyMainFolder , 'Report_ML_Training_Performance', 'Error_vs_model_parameter_GT.fig'));
        close(h);
        %r.close();

    end
end
end
