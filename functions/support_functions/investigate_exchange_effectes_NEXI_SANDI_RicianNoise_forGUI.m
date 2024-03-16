function SANDIinput = investigate_exchange_effectes_NEXI_SANDI_RicianNoise_forGUI(SANDIinput, fig_axes)

%% Investigate the effect of exchange. Create sythetic signal using a modified NEXI model, made of NEXI + spherical compartment

% Read the input
delta = SANDIinput.delta ;
smalldel = SANDIinput.smalldel ;
sigma_SHresiduals = SANDIinput.sigma_SHresiduals;
sigma_mppca = SANDIinput.sigma_mppca;
Dsoma = SANDIinput.Dsoma;
Din_UB = SANDIinput.Din_UB ;
Rsoma_UB = SANDIinput.Rsoma_UB ;
De_UB = SANDIinput.De_UB ;
b = SANDIinput.bunique;
Ndirs_per_Shell = SANDIinput.Ndirs_per_Shell;

Ntest = 2.5E3;

disp('* Investigating the effect of inter-compartmental exchange on SANDI model parameters estimates')
fprintf(SANDIinput.LogFileID,'* Investigating the effect of inter-compartmental exchange on SANDI model parameters estimates\n');

% Make sure b values are not 0s
b(b==0) = 1e-6;
Nbvals = numel(b);

T = drchrnd([1 1 1], Ntest); % To ensure uniform coverage of the simplex p1+p2+p3 = 1
p1 = T(:,1); 
p2 = T(:,2);
p3 = rand(Ntest,1).*Din_UB;
p4 = (Rsoma_UB-1).*rand(Ntest,1)+1;
p5 = rand(Ntest,1).*De_UB;

tex = [5 10 30 50 100 500 inf]; % Exchange time in ms

p_GT = [p1(:), p2(:)./(1-p1(:)), p3(:), p4(:), p5(:)];

GT = p_GT;

GT(:,1) = (1-p_GT(:,1)).*p_GT(:,2); % fneurite in SANDI model
GT(:,2) = p_GT(:,1); % fsoma in SANDI model

% Convert to suitable form for fitting the SANDI model, where the
% transofrmation fneurite = cos(p1).^2 and fsoma = (1-p1).*cos(p2).^2 have
% been used to ensure that fneurite + fsoma <= 1
p1 = acos(sqrt(GT(:,1))); % fenurite after the transformation for the fitting
p2 = acos(sqrt(GT(:,2)./(1-GT(:,1)))); % fsoma after the transformation for the fitting
GT(:,1) = p1;
GT(:,2) = p2;

S_GT = zeros(Ntest,Nbvals,numel(tex));

options = optimset('display', 'off');

for i=1:numel(tex)
    tt = tex(i);
    parfor j = 1:Ntest
        S_GT(j,:,i) = NEXIS([p_GT(j,:), tt], b, delta, smalldel,Dsoma);
    end
end

%% Fit the SANDI model to evaluate the impact of the non-exchanging compartments

rr_sandi = zeros(Ntest,5,numel(tex));

for i=1:numel(tex)
    parfor j = 1:Ntest
        rr_sandi(j,:,i) = lsqcurvefit(@(p,x) SANDImodel(p, x, delta, smalldel, Dsoma), GT(j,:), b, S_GT(j,:,i), [0 0 0 1 0], [pi/2 pi/2 Din_UB Rsoma_UB De_UB], options);
    end

    p1 = cos(rr_sandi(:,1,i)).^2;
    p2 = (1-cos(rr_sandi(:,1,i)).^2).*cos(rr_sandi(:,2,i)).^2;
    rr_sandi(:,1,i) = p1;
    rr_sandi(:,2,i) = p2;

end

%% Investigate the effect of exchange under the experimental noise conditions

median_sigma_mppca = nanmedian(sigma_mppca);
median_sigma_SHresiduals = nanmedian(sigma_SHresiduals);
median_sigma_SHresiduals_vec = median_sigma_SHresiduals./sqrt(Ndirs_per_Shell);
median_sigma_SHresiduals_vec = (median_sigma_SHresiduals_vec)';

for i=1:numel(tex)
    parfor j = 1:Ntest    
        tmp = RiceMean(S_GT(j,:,i), median_sigma_mppca);    
        S_GT_with_rician_bias_noisy(j,:,i) =  tmp + median_sigma_SHresiduals_vec.*randn(size(tmp));
    end
end

options = optimset('display', 'off');

%% Fit the SANDI model to evaluate the impact of model assumptions.

rr_sandi_noise = zeros(Ntest,5,numel(tex));

ffit = @(p,x) RiceMean(SANDImodel(p, x, delta, smalldel, Dsoma), median_sigma_mppca);


for i=1:numel(tex)
    parfor j = 1:Ntest
        try
           rr_sandi_noise(j,:,i) = lsqcurvefit(ffit, GT(j,:), b, S_GT_with_rician_bias_noisy(j,:,i), [0 0 0 1 0], [pi/2 pi/2 Din_UB Rsoma_UB De_UB], options);
        catch
           rr_sandi_noise(j,:,i) = nan(1,5);
        end
    end
    p1 = cos(rr_sandi_noise(:,1,i)).^2;
    p2 = (1-cos(rr_sandi_noise(:,1,i)).^2).*cos(rr_sandi_noise(:,2,i)).^2;
    rr_sandi_noise(:,1,i) = p1;
    rr_sandi_noise(:,2,i) = p2;

end


%% Compute stats for SANDI signals in case of exchange

GT(:,1) = (1-p_GT(:,1)).*p_GT(:,2); % fneurite in SANDI model
GT(:,2) = p_GT(:,1); % fsoma in SANDI model

titles = {'fneurite', 'fsoma', 'Din', 'Rsoma', 'De'};


for i=1:5

    step_size = max(GT(:,i))/10;

    lb = 0:step_size:max(GT(:,i));
    ub = 0+step_size:step_size:max(GT(:,i))+step_size;

    Ymean = zeros(numel(lb), 1);
    Ystd = zeros(numel(lb), 1);

    for j=1:numel(lb)
        condition = GT(:,i)>=lb(j) & GT(:,i)<ub(j);
        tmp = (GT(condition,i) - rr_sandi(condition,i,tex==30));
        Ymean(j) = mean(tmp);
        Ystd(j) = std(tmp);
    end

    current_ax = subplot(2,3,i, 'Parent', fig_axes{1});
    hold(current_ax,'on');
    
    plot(current_ax, GT(:,i), GT(:,i)-rr_sandi(:,i,end), 'k-')
    plot(current_ax, lb,(Ymean),'r-')
    plot(current_ax, lb,(Ymean+Ystd),'r:')
    plot(current_ax, lb,(Ymean-Ystd),'r:')

    ylabel(current_ax, 'Error')
    xlabel(current_ax, 'Ground Truth')
    title(current_ax, titles{i})
    axis(current_ax, [min(GT(:,i)) max(GT(:,i)) -max(GT(:,i))./2 max(GT(:,i))./2])

end
    
rmse = zeros(5, numel(tex));

for i=1:5

    for j = 1:numel(tex)
        rmse(i,j) = sqrt( mean( (GT(:,i)-rr_sandi(:,i,j)).^2,1 ));
    end

    current_ax = subplot(2,3,i, 'Parent', fig_axes{2});
    hold(current_ax, 'on');
    
    plot(current_ax, tex, rmse(i,:), 'k-'),
    plot(current_ax, tex(tex==30), rmse(i,tex==30), 'r*')

    ylabel(current_ax, 'RMSE')
    xlabel(current_ax, 't_{ex} [ms]')
    title(current_ax, titles{i})

end
    
error = zeros(size(GT,1), 5, numel(tex));

for i=1:5

    labels = cell(numel(tex), 1);

    for j = 1:numel(tex)
        error(:,i,j) = GT(:,i)-rr_sandi(:,i,j);
        labels{j} = num2str(tex(j));
    end

    current_ax = subplot(2,3,i, 'Parent', fig_axes{3});
    hold(current_ax, 'on');
    boxplot(current_ax, squeeze(error(:,i,:)),'label', labels, 'PlotStyle','compact', 'OutlierSize',0.001), 
    grid(current_ax, 'on')
    ylabel(current_ax, 'Error')
    ylim(current_ax, [-max(GT(:,i))./2 max(GT(:,i))./2]);
    title(current_ax, titles{i})

end

%% Compute stats for noisy NEXIS signals 

for i=1:5

    step_size = max(GT(:,i))/10;

    lb = 0:step_size:max(GT(:,i));
    ub = 0+step_size:step_size:max(GT(:,i))+step_size;

    Ymean = zeros(numel(lb), 1);
    Ystd = zeros(numel(lb), 1);

    for j=1:numel(lb)
        condition = GT(:,i)>=lb(j) & GT(:,i)<ub(j);
        tmp = (GT(condition,i) - rr_sandi_noise(condition,i,tex==30));
        Ymean(j) = mean(tmp);
        Ystd(j) = std(tmp);
    end

    current_ax = subplot(2,3,i, 'Parent', fig_axes{4});
    hold(current_ax,'on');
    plot(current_ax, GT(:,i), GT(:,i)-rr_sandi(:,i,end), 'k-')
    plot(current_ax, lb,(Ymean),'r-')
    plot(current_ax, lb,(Ymean+Ystd),'r:')
    plot(current_ax, lb,(Ymean-Ystd),'r:')

    ylabel(current_ax, 'Error')
    xlabel(current_ax, 'Ground Truth')
    title(current_ax, titles{i})
    axis(current_ax, [min(GT(:,i)) max(GT(:,i)) -max(GT(:,i))./2 max(GT(:,i))./2])
end
    
rmse = zeros(5, numel(tex));

for i=1:5


    for j = 1:numel(tex)
        rmse(i,j) = sqrt( mean( (GT(:,i)-rr_sandi_noise(:,i,j)).^2,1 ));
    end

    current_ax = subplot(2,3,i,'Parent',fig_axes{5});
    hold(current_ax, 'on');
    plot(current_ax, tex, rmse(i,:), 'k-'), 
    plot(current_ax, tex(tex==30), rmse(i,tex==30), 'r*')
    ylabel(current_ax, 'RMSE')
    xlabel(current_ax, 't_{ex} [ms]')
    title(current_ax, titles{i})
end
    
error = zeros(size(GT,1), 5, numel(tex));

for i=1:5

    labels = cell(numel(tex), 1);

    for j = 1:numel(tex)
        error(:,i,j) = GT(:,i)-rr_sandi_noise(:,i,j);
        labels{j} = num2str(tex(j));
    end

    current_ax = subplot(2,3,i,'Parent',fig_axes{6}); 
    hold(current_ax,'on');
    boxplot(current_ax, squeeze(error(:,i,:)),'label', labels, 'PlotStyle','compact','OutlierSize',0.001), 
    grid(current_ax, 'on');
    ylabel(current_ax, 'Error')
    ylim(current_ax, [-max(GT(:,i))./2 max(GT(:,i))./2]);
    title(current_ax, titles{i})

end

%% Sensitivity analysis

trainedML = SANDIinput.trainedML;
MLmodel = SANDIinput.MLmodel ;
model = SANDIinput.model ;

if isempty(MLmodel), MLmodel='RF'; end

MLdebias = SANDIinput.MLdebias;

tmp_mppca = nanmedian(sigma_mppca);
median_sigma_SHresiduals = nanmedian(sigma_SHresiduals);
tmp_SHresiduals_vec = median_sigma_SHresiduals./sqrt(Ndirs_per_Shell);
tmp_SHresiduals_vec = max(tmp_SHresiduals_vec);

noises_mppca = [0 0 tmp_mppca tmp_mppca];
noises_SHresiduals = [0 0 tmp_SHresiduals_vec tmp_SHresiduals_vec];
tex_vec = [inf 30 inf 30];

main_titles = {'Sensitivity analysis (no exchange, no noise)';...
          'Sensitivity analysis (exchange tex=30 ms, no noise)'; 
          'Sensitivity analysis (no exchange, experimental noise)';...
          'Sensitivity analysis (exchange tex=30 ms, experimental noise)'};

for tt = 1:4

median_sigma_mppca = noises_mppca(tt);
median_sigma_SHresiduals_vec = noises_SHresiduals(tt);

tex = tex_vec(tt);

ffit = @(p,x) RiceMean(SANDImodel(p, x, delta, smalldel, Dsoma), median_sigma_mppca);

rr_sandi_modulated = zeros(Ntest,5);
rr_sandi_no_modulated = zeros(Ntest,5);
S_modulated = zeros(Ntest,Nbvals);
S_no_modulated = zeros(Ntest,Nbvals);

mean_changes = zeros(size(p_GT,2),size(p_GT,2));

for i=1:size(p_GT,2)
    
    p_GT_modulated = p_GT;
    p_GT_modulated(:,i) = p_GT(:,i) + 0.1.*p_GT(:,i);
    if i<=2
        p_GT_modulated(p_GT_modulated(:,i)>1,i) = 1;
        p_GT_modulated(p_GT_modulated(:,i)<0,i) = 0;
    end
    
    GT = p_GT;
    GT(:,1) = (1-p_GT(:,1)).*p_GT(:,2); % fneurite in SANDI model
    GT(:,2) = p_GT(:,1); % fsoma in SANDI model
    p1 = acos(sqrt(GT(:,1))); % fenurite after the transformation for the fitting
    p2 = acos(sqrt(GT(:,2)./(1-GT(:,1)))); % fsoma after the transformation for the fitting
    GT(:,1) = p1;
    GT(:,2) = p2;
    
    GT_modulated = p_GT_modulated;
    GT_modulated(:,1) = (1-p_GT_modulated(:,1)).*p_GT_modulated(:,2); % fneurite in SANDI model
    GT_modulated(:,2) = p_GT_modulated(:,1); % fsoma in SANDI model
    p1 = acos(sqrt(GT_modulated(:,1))); % fenurite after the transformation for the fitting
    p2 = acos(sqrt(GT_modulated(:,2)./(1-GT_modulated(:,1)))); % fsoma after the transformation for the fitting
    GT_modulated(:,1) = p1;
    GT_modulated(:,2) = p2;
    GT_modulated = abs(GT_modulated);
    
    parfor j = 1:Ntest
        
        tmp1 = NEXIS([p_GT_modulated(j,:), tex], b, delta, smalldel,Dsoma);
        tmp2 = RiceMean(tmp1, median_sigma_mppca);
        S_modulated(j,:) =  tmp2 + median_sigma_SHresiduals_vec.*randn(size(tmp2));
        
        tmp1 = NEXIS([p_GT(j,:), tex], b, delta, smalldel,Dsoma);
        tmp2 = RiceMean(tmp1, median_sigma_mppca);
        S_no_modulated(j,:) =  tmp2 + median_sigma_SHresiduals_vec.*randn(size(tmp2));
        
        %rr_sandi_modulated(j,:) = lsqcurvefit(ffit, GT_modulated(j,:), b, S_modulated(j,:), [0 0 0 1 0], [pi/2 pi/2 Din_UB Rsoma_UB De_UB], options);
        %rr_sandi_no_modulated(j,:) = lsqcurvefit(ffit, GT(j,:), b, S_no_modulated(j,:), [0 0 0 1 0], [pi/2 pi/2 Din_UB Rsoma_UB De_UB], options);
        
    end
    
    %p1 = cos(rr_sandi_modulated(:,1)).^2;
    %p2 = (1-cos(rr_sandi_modulated(:,1)).^2).*cos(rr_sandi_modulated(:,2)).^2;
    %rr_sandi_modulated(:,1) = p1;
    %rr_sandi_modulated(:,2) = p2;
    
    
    switch MLmodel
        
        case 'RF'
            %% RF fit
            
            rr_sandi_modulated = apply_RF_matlab(S_modulated, trainedML, MLdebias);
            
        case 'MLP'
            
            %% MLP fit
            
            rr_sandi_modulated = apply_MLP_matlab(S_modulated, trainedML, MLdebias);
            
    end
    
    p1 = rr_sandi_modulated(:,2);
    p2 = rr_sandi_modulated(:,1)./(1-p1);
    rr_sandi_modulated(:,1) = p1;
    rr_sandi_modulated(:,2) = p2;
    
    %p1 = cos(rr_sandi_no_modulated(:,1)).^2;
    %p2 = (1-cos(rr_sandi_no_modulated(:,1)).^2).*cos(rr_sandi_no_modulated(:,2)).^2;
    %rr_sandi_no_modulated(:,1) = p1;
    %rr_sandi_no_modulated(:,2) = p2;
    
    switch MLmodel
        
        case 'RF'
            %% RF fit
            
            rr_sandi_no_modulated = apply_RF_matlab(S_no_modulated, trainedML, MLdebias);
            
        case 'MLP'
            
            %% MLP fit
            
            rr_sandi_no_modulated = apply_MLP_matlab(S_no_modulated, trainedML, MLdebias);
            
    end
    
    p1 = rr_sandi_no_modulated(:,2);
    p2 = rr_sandi_no_modulated(:,1)./(1-p1);
    rr_sandi_no_modulated(:,1) = p1;
    rr_sandi_no_modulated(:,2) = p2;
    
    
    for j=1:size(p_GT,2)
        
        mean_changes(i,j) = nanmedian( (rr_sandi_modulated(:,j) - rr_sandi_no_modulated(:,j))./rr_sandi_no_modulated(:,j).*100,1 );
        
    end
    
end

cdata = mean_changes;
xvalues = titles;
yvalues = titles;
h = heatmap(fig_axes{6+tt}, xvalues,yvalues,cdata);

R = [ones(1,9), 1, 1:-0.1:0];
G = [0:0.1:0.9, 1, 0.9:-0.1:0];
B = [0:0.1:1, 1, ones(1,9)];

map = [R', G', B'];
h.Colormap = map;
h.ColorLimits = [-20 20];
h.Title = main_titles{tt};

end

end
