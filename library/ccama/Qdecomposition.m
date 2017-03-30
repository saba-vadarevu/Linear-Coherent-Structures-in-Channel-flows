function [B H S] = Qdecomposition(Q)
% Get a decomposition Q = S + S' with S of minimum rank.
%
%              min{rank(S)| Q = S + S'} = max{pi(Q),nu(Q)}
%
% INPUT:   Hermitian matrix Q
% OUTPUT:  S = BH'
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Q = (Q + Q')/2;
[n, m] = size(Q);

% eigenvalues with absolute values less than threshold are considered zero
threshold = 1e-12*norm(Q);

% eigenvalue decomposition Q = V D V';
[V, D] = eig(Q);

% determine the signature of Q, i.e., (piQ, nuQ, deltaQ)
piQ = sum(diag(D) > threshold);
nuQ = sum(diag(D) < -threshold);
deltaQ = n - piQ - nuQ;

% make the diagonal entries to be 1, -1 or 0
D1a = diag(diag(D) > threshold);
Da = D*D1a;
D1b = -diag(diag(D) < -threshold);
Db = D*D1b;
D1 = D1a + D1b;
Dc = Da + Db + (eye(n)-abs(D1));
V1 = V*sqrt(Dc);

% rearrange the order of the diagonal elements
indexvec = [1:n]';
piQindex = compress(indexvec.*diag(D1a));
nuQindex = compress(indexvec.*-diag(D1b));
deltaQindex = compress(indexvec.*diag(eye(n)-abs(D1)));
reorderindex = [piQindex;nuQindex;deltaQindex];

V2 = zeros(n,n);
for i = 1:n
    V2(:,i) = V1(:,reorderindex(i));
end
D2vec = [ones(1,piQ) -ones(1,nuQ) zeros(1,deltaQ)];
D2 = diag(D2vec);

% compute B, H and S matrices
D2 = 2*D2;
V2 = V2*sqrt(1/2);
if piQ <= nuQ
    qQ = nuQ-piQ;
    Bhat = [eye(piQ) zeros(piQ,qQ);
            eye(piQ) zeros(piQ,qQ);
             zeros(qQ,piQ) eye(qQ);
                zeros(deltaQ,nuQ)];
    Hhat = [eye(piQ) zeros(piQ,qQ);
           -eye(piQ) zeros(piQ,qQ);
            zeros(qQ,piQ) -eye(qQ);
                zeros(deltaQ,nuQ)];
else
    qQ = piQ - nuQ;
    Bhat = [eye(qQ) zeros(qQ,nuQ);
           zeros(nuQ,qQ) eye(nuQ);
           zeros(nuQ,qQ) eye(nuQ);
               zeros(deltaQ,piQ)];
    Hhat = [eye(qQ) zeros(qQ,nuQ);
           zeros(nuQ,qQ) eye(nuQ);
           zeros(nuQ,qQ) -eye(nuQ);
           zeros(deltaQ,piQ)];
end

B = V2*Bhat;
H = V2*Hhat;
S = B*H';

end
