function xini = grid_search(y, f, x, lb, ub, NstepsGrid)

tic
disp('   - Performing Grid Search')

Nparams = numel(lb);

% Create the grid
disp('   - Creating the Grid')
for i=1:Nparams
    eval(['p' num2str(i) ' = linspace(' num2str(lb(i)) ',' num2str(ub(i)) ',' num2str(NstepsGrid) ');']);
end
command = '[p1';
for i=2:Nparams
    command = [command ',p' num2str(i)];
end
command = [command '] = ndgrid(p1'];
for i=2:Nparams
    command = [command ',p' num2str(i)];
end
command = [command ');'];
eval(command);
for i=1:Nparams
    eval(['p' num2str(i) ' = p' num2str(i) '(:);']);
end
command = '[p1';
for i=2:Nparams
    command = [command ',p' num2str(i)];
end
command = [command '];'];
xtmp = eval(command);

Sexample = f(xtmp(1,:), x);

S = zeros(size(xtmp,1), numel(Sexample));

parfor i = 1:size(xtmp,1)

    S(i,:) = f(xtmp(i,:), x);

end

idxmin = zeros(size(y,1), 1);

disp('   - Searching Optimal solution')

parfor i = 1:size(y,1)

    [~, idxmin(i)] = min( sum( (y(i,:) - S).^2,2 ) );

end

xini = xtmp(idxmin,:);

tt = toc;

disp(['   - DONE! Grid search performed in ' num2str(round(tt)) ' sec.'])

end