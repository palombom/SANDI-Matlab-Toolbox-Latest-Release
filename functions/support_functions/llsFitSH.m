function [coef, X] = llsFitSH(dirs, y, order)

% Author: Jelle Veraart  and Santiago Coelho (jelle.veraart@nyulangone.org) 
% copyright NYU School of Medicine, 2022

%X = getSH(order, dirs, 0);
X = getSH(order, dirs);
coef = X\y;

end


% function Y = getSH (Lmax, dirs, CS_phase)
% 
%             % Ylm_n = get_even_SH(dirs,Lmax,CS_phase)
%             %
%             % if CS_phase=1, then the definition uses the Condon-Shortley phase factor
%             % of (-1)^m. Default is CS_phase=0 (so this factor is ommited)
%             %
%             % By: Santiago Coelho (https://github.com/NYU-DiffusionMRI/SMI/blob/master/SMI.m)
%             
%             if size(dirs,2)~=3
%                 dirs=dirs';
%             end
%             Nmeas=size(dirs,1);
%             [PHI,THETA,~]=cart2sph(dirs(:,1),dirs(:,2),dirs(:,3)); THETA=pi/2-THETA;
%             l=0:2:Lmax;
%             l_all=[];
%             m_all=[];
%             for ii=1:length(l)
%                 l_all=[l_all, l(ii)*ones(1,2*l(ii)+1)];
%                 m_all=[m_all -l(ii):l(ii)];
%             end
%             K_lm=sqrt((2*l_all+1)./(4*pi) .* factorial(l_all-abs(m_all))./factorial(l_all+abs(m_all)));
%             if nargin==2 || isempty(CS_phase) || ~exist('CS_phase','var') || ~CS_phase
%                 extra_factor=ones(size(K_lm));
%                 extra_factor(m_all~=0)=sqrt(2);
%             else
%                 extra_factor=ones(size(K_lm));
%                 extra_factor(m_all~=0)=sqrt(2);
%                 extra_factor=extra_factor.*(-1).^(m_all);
%             end
%             P_l_in_cos_theta=zeros(length(l_all),Nmeas);
%             phi_term=zeros(length(l_all),Nmeas);
%             id_which_pl=zeros(1,length(l_all));
%             for ii=1:length(l_all)
%                 all_Pls=legendre(l_all(ii),cos(THETA));
%                 P_l_in_cos_theta(ii,:)=all_Pls(abs(m_all(ii))+1,:);
%                 id_which_pl(ii)=abs(m_all(ii))+1;
%                 if m_all(ii)>0
%                     phi_term(ii,:)=cos(m_all(ii)*PHI);
%                 elseif m_all(ii)==0
%                     phi_term(ii,:)=1;
%                 elseif m_all(ii)<0
%                     phi_term(ii,:)=sin(-m_all(ii)*PHI);
%                 end
%             end
%             Y_lm=repmat(extra_factor',1,Nmeas).*repmat(K_lm',1,Nmeas).*phi_term.*P_l_in_cos_theta;
%             Y=Y_lm';
% end

function Y = getSH (order, sphere_points)

n = size(sphere_points,1); 
[phi, theta] = cart2sph(sphere_points(:,1),sphere_points(:,2),sphere_points(:,3));

theta = pi/2 - theta;  
k =(order + 2)*(order + 1)/2;  
  
Y = zeros(n,k);

for l = 0:2:order  
    Pm = legendre(l,cos(theta')); 
    Pm = Pm';
    lconstant = sqrt((2*l + 1)/(4*pi));
    center = (l+1)*(l+2)/2 - l;
    
    Y(:,center) = lconstant*Pm(:,1);
    for m=1:l
        precoeff = lconstant * sqrt(factorial(l - m)/factorial(l + m));
        if mod(m,2) == 1
            precoeff = -precoeff;
        end
        Y(:, center + m) = sqrt(2)*precoeff*Pm(:,m+1).*cos(m*phi);
        Y(:, center - m) = sqrt(2)*precoeff*Pm(:,m+1).*sin(m*phi);         
    end
end
end