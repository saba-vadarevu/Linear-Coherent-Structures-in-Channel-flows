% Linear stochastic simulations of the perturbed dynamics
%
% Written by Armin Zare and Mihailo Jovanovic, April 2016
% 

%% =========================================================
% Load matrices from solution of optimization problem (CC) %
% ==========================================================
X = output.X;
Z = output.Z;

%% ===================
% Filter realization %
% ====================
% input dynamical generator A and matrices X and Z from problem (CC)
[Af,Bf,Cf,Df] = linfilter(A,X,Z);

%% ========================================
% Find the optimal feedback gain matrix K %
% =========================================

[mb, nb]=size(Bf);
Xsqrt = sqrtm(X);

cvx_clear
cvx_begin
    variable K(nb,mb);
    variable Rho(nb,nb)
    
    minimize norm(K*Xsqrt,'fro')
    subject to
    (Af-Bf*K)*X + X*(Af-Bf*K)' + Bf*Rho*Bf' == 0;
                                        Rho >= 0;
cvx_end

Af1 = Af - Bf*K;


%% ========================================================================
% Linear stochastic simulation - simulation with band-limited white noise %
% =========================================================================

Af_ext = Af1;
Bf_ext = Bf;
Cf_ext = -K;
Df_ext = Df;

[l1, l2] = size(Bf_ext);
m = l2;

clear tsim x yout covY covYea varUcea covYall
Snum = 20;
Ts = 0.01;
t = 0:Ts:800;

for n = 1:Snum
    [tsim(n,:),x(n,:,:),yout(n,:,:)] = sim('sim_mdl',t,simset('Decimation',100,'OutputPoints','specified'));
    [n]
end


%% ================================================================
% Post processing to compute covariance matrices from simulations %
% =================================================================
clear covY covYea varUcea covYall

ynew = x;
tnew = tsim;

[yl1, yl2, yl3] = size(ynew);
[tl1, tl2] = size(tnew);

% choose starting time for integration
tstart = 0;
% tstart = floor(tl2/4);

covY = zeros(Snum,tl2-tstart,yl3,yl3);
[a1, a2, a3, a4] = size(covY);
for i = 1:Snum
    for n = 1:a2
        covY(i,n,:,:) = squeeze(ynew(i,n+tstart,:))*squeeze(ynew(i,n+tstart,:))';
    end
    [i]
end

covYea = zeros(Snum,a2,2*N,2*N);
varUcea = zeros(Snum,a2);
for i = 1:Snum
    for n = 1:a2
        covYea(i,n,:,:) = squeeze(sum(covY(i,1:n,:,:),2)/n);
        varUcea(i,n) = trace(squeeze(covYea(i,n,:,:)));
    end
    [i]
end

covYall = squeeze(sum(covYea,1))/Snum;

varUc = zeros(a2,1);
for n = 1:a2
    varUc(n) = trace(squeeze(covYall(n,:,:)));
end

Yeval = squeeze(covYall(end,:,:));

