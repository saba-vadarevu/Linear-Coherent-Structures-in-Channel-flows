% Script for plotting figures from TAC paper
%
% Written by Armin Zare and Mihailo Jovanovic, April 2016
% 

% singular values of Z (Fig. 4)
figure; semilogy(svd(Z),'bo-')

% quality of solution - feasibility (Fig. 7)
figure; plot(1:N, diag(Sigma(1:N,1:N)), 1:N, diag(X(1:N,1:N)),'ro')
figure; plot(1:N, diag(Sigma(N+1:2*N,N+1:2*N)), 1:N, diag(X(N+1:2*N,N+1:2*N)),'ro')

% quality of solution - completion
figure; pcolor(1:N,1:N,flipud(Sigma(1:N,1:N))); shading interp; colorbar vert
figure; pcolor(1:N,1:N,flipud(X(1:N,1:N))); shading interp; colorbar vert

% path of stochastic simulations (Fig. 8)
% average of all simulations is marked by black line
figure; plot(tnew(1,:),varUcea,tnew(1,:),varUc,'ko')

% check quality of output covariances resulting from simulations (Fig. 9)
figure; pcolor(1:N,1:N,flipud(Sigma(1:N,1:N))); shading interp; colorbar vert
figure; pcolor(1:N,1:N,flipud(Yeval(1:N,1:N))); shading interp; colorbar vert


%% =========
% Printing %
% ==========
figure(1);
% cb = colorbar('vert');
h = get(gcf,'CurrentAxes');
set(h, 'FontName', 'cmr10', 'FontSize', 24, 'xscale', 'lin', 'yscale', 'log')
% set(cb, 'FontName', 'cmr10', 'FontSize', 24)
% axis([0 N 0 10])
box on
print svdZ_gamma2p2 -dpng -r300
