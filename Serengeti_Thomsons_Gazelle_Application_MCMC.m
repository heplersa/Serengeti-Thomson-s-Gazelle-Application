
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%% MCMC for spatio-temporal occupancy model for the manuscript
%%%%% "Identifying Drivers of Spatial Variation in Occupancy with Limited 
%%%%% Replication Camera Trap Data" by Hepler, Erhardt, and Anderson
%%%%% This code was run on Matlab 2015a
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Prior to running this code, use the data in Serengeti_Thomsons_Gazelle_Application_Data.csv to
%%% create a .mat file named ApplicationData.mat that has stored the design matrices 'X' and 'W', the
%%% vector of detections 'Tommies', a vector TID of the time period, 
%%% and the same-time neighborhood matrix 'A',
%%% the lagged neighborhood matrix 'Alag', the temporal neighborhood 'At'

%%% load the data
load ApplicationData.mat; 
species1 = 'Tommies'; 
y = eval(lower(species1)); 

n = length(y); %%% total number of observations

A=sparse(A); %%% within season binary adjacency matrix
Alag=sparse(Alag); %%% matrix used for the space-time interactions that has "1" entries for lagged neighbors
At=sparse(At); %%% temporal adjacency matrix; entries 1 if same location previous season

%%

MCS = 200000;  %%%% Total number of iterations of the chain
MCB = 20000;     %%%% Number of burn-in iterations
MCM = 5;     %%%% how much to thin the chain

bp = length(W(1,:));   %%% number of covariates in detection model
ap = length(X(1,:));   %%% number of covariates in occupancy model

sigb = 5;    %%%% standard deviation of 5 for beta prior 
siga = 5;    %%%% standard deviation of 10 for alpha prior
deltsd = 3;     %%% standard deviation of delta prior

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%% Set Initial Values for Markov Chain

z = y;    %%%%% use the observed indicators as the initial value for z
delt = 0.5;
deltT = 2;
deltS = .3;
alpha = zeros(ap,1);
beta = zeros(bp,1);

%%%% Note we only need to update the z's where the corresponding y=0
%%%% If y=1, then z has to be 1. This is already true from the initial
%%%% condition, so those z's never need to be updated
nyz = n-sum(y); %%%% how many of the y's are 0
zInd = find(y==0);   %%% identify which rows have y=0

muvec = exp(X*alpha)./(1+exp(X*alpha));

Ind1 = find(TID==1); %%%TID = indicator for which season
IndT = find(TID>1);
T=max(TID); %%% total number of seasons

logitpsi = zeros(n,1);
logitpsi(Ind1) = X(Ind1,:)*alpha+delt*A(Ind1,Ind1)*(z(Ind1)-muvec(Ind1));
logitpsi(IndT) = X(IndT,:)*alpha+deltT*At(IndT,:)*(z-muvec)+deltS*Alag(IndT,:)*(z-muvec);
psi=exp(logitpsi)./(1+exp(logitpsi));
logitp = W*beta;
p=exp(logitp)./(1+exp(logitp));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%% Set up MCMC%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%% Construct empty vectors for MCMC output
Zout = zeros(n,(MCS-MCB)/MCM);
betaout = zeros(bp,(MCS-MCB)/MCM);
alphaout = zeros(ap,(MCS-MCB)/MCM);
deltout = zeros(3,(MCS-MCB)/MCM);
betaburn = zeros(bp,MCB/MCM);
alphaburn = zeros(ap,MCB/MCM);
deltburn = zeros(3,MCB/MCM);


%%%% keep track of acceptance rates
acceptb = zeros(bp,1);
accepta = zeros(ap,1);
acceptd = zeros(2,1);

%%% create vectors to store the variances to use for the adaptive MCMC
varalpha = ones(ap,1);
varbeta = ones(bp,1);
vardelt = 1;
covdeltTS = eye(2);

%%% scalars used to adapt the step size
ad = zeros(2,1);    
aa = zeros(ap,1);
ab = zeros(bp,1);

%%% initial scalars used to control the proposal variances
Af = 1;
As = 1.5; 
Bf = .5;
Bs = 2;
Df = 1;
Ds = 1.5;

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%% run the MCMC algorithm %%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for m=1:MCS

    %%%%% Update each of the z's individually  
    %%% update z in first T-1 time periods
    for t=1:(T-1)
        ss0 = find(TID==t & y<1);
        IndX = find(TID==(t+1));
        nn = length(ss0);
        for i=1:nn
            j=ss0(i);
            ztemp = z;
            ztemp(j)=0;
            logitpsitemp = X(IndX,:)*alpha+deltT*At(IndX,:)*(ztemp-muvec)+deltS*Alag(IndX,:)*(ztemp-muvec);
            psitemp = exp(logitpsitemp)./(1+exp(logitpsitemp));            
            lf0 = ztemp(IndX)'*logitpsitemp+ones(length(IndX),1)'*log(1-psitemp);
            ztemp(j)=1;
            logitpsitemp = X(IndX,:)*alpha+deltT*At(IndX,:)*(ztemp-muvec)+deltS*Alag(IndX,:)*(ztemp-muvec);
            psitemp = exp(logitpsitemp)./(1+exp(logitpsitemp));
            lf1 = ztemp(IndX)'*logitpsitemp+ones(length(IndX),1)'*log(1-psitemp);
            U1=rand;
            z(j)=(log(U1)<( log(1-p(j))+log(psi(j))+lf1-log( (1-p(j))*psi(j)*exp(lf1)+(1-psi(j))*exp(lf0))));        
            logitpsi(Ind1) = X(Ind1,:)*alpha+delt*A(Ind1,:)*(z-muvec);
            logitpsi(IndT) = X(IndT,:)*alpha+deltT*At(IndT,:)*(z-muvec)+deltS*Alag(IndT,:)*(z-muvec);
            psi = exp(logitpsi)./(1+exp(logitpsi)); 
        end 
    end

    %%% update z in last time period
    ss0 = find(TID==T & y<1);
    nn = length(ss0);
    for i=1:nn
        j=ss0(i);
        U1=rand;
        z(j)=(log(U1)<( log(1-p(j))+log(psi(j))-log((1-p(j))*psi(j)+(1-psi(j)))));      
        logitpsi(Ind1) = X(Ind1,:)*alpha+delt*A(Ind1,:)*(z-muvec);
        logitpsi(IndT) = X(IndT,:)*alpha+deltT*At(IndT,:)*(z-muvec)+deltS*Alag(IndT,:)*(z-muvec);
        psi = exp(logitpsi)./(1+exp(logitpsi)); 
    end    

    %%%%% Update each beta individually with Metropolis Hastings step
    for k=1:bp           
        betaold = beta;        
        MLLBbeta = 1/(2*sigb^2)*(beta')*beta-z'*(y.*(W*beta)-log(1+exp(W*beta))); %%% minus the unnormalized cond'l posterior based on the current value of beta
        %%%% for first few iterations, update using normal random walk with
        %%%% small step size. Then adjust the step size based on the
        %%%% variance of the draws and the acceptance rate
        if m<=50 || rand<0.05
            beta(k) = mvnrnd(betaold(k), (Bf)^2/bp)';
        else 
            beta(k) = mvnrnd(betaold(k), exp(ab(k))*(Bs)^2/bp*varbeta(k))';
        end    
        %%%% decide whether to accept or reject the proposed vector of beta
        MLLAbeta = 1/(2*sigb^2)*(beta')*beta-z'*(y.*(W*beta)-log(1+exp(W*beta)));
        V=rand;
        if log(V)<(MLLBbeta-MLLAbeta) 
            MLLBbeta=MLLAbeta;
            acceptb(k)=acceptb(k)+1;
            p=exp(W*beta)./(1+exp(W*beta));
        else
            beta(k)=betaold(k);
        end            
    end                

    %%%%% Update alpha using pseudolikelihood approximation for [Z1|theta]
    %%%%% alternatively could use the exact sample algorithm 
    for k=1:ap
        MLLBalpha = 1/(2*siga^2)*(alpha)'*(alpha)-z'*logitpsi-ones(n,1)'*log(1-psi);
        alphaold = alpha;
        muvecold = muvec;
        logitpsiold = logitpsi;
        psiold = psi;
        if m<=50 || rand<0.05
            alpha(k) = normrnd(alphaold(k), (Af)^2);
        else
            alpha(k) = normrnd(alphaold(k), exp(aa(k))*(As)^2*varalpha(k));
        end
        muvec = exp(X*alpha)./(1+exp(X*alpha));
        logitpsi(Ind1) = X(Ind1,:)*alpha+delt*A(Ind1,Ind1)*(z(Ind1)-muvec(Ind1));
        logitpsi(IndT) = X(IndT,:)*alpha+deltT*At(IndT,:)*(z-muvec)+deltS*Alag(IndT,:)*(z-muvec);
        psi = exp(logitpsi)./(1+exp(logitpsi));
        MLLAalpha = 1/(2*siga^2)*(alpha)'*(alpha)-z'*logitpsi-ones(n,1)'*log(1-psi);
        V=rand;
        if log(V)<(MLLBalpha-MLLAalpha) 
            accepta(k)=accepta(k)+1;
        else
            alpha=alphaold;
            muvec = muvecold;
            logitpsi = logitpsiold;
            psi = psiold;
        end    
    end

    %%% update delt using pseudolikelihood approximation for [Z1|theta]
    LLBdelt = -1/(2*deltsd^2)*(delt'*delt)+z'*logitpsi+ones(n,1)'*log(1-psi);
    deltold = delt;
    logitpsiold = logitpsi;
    psiold = psi;
    if m<=50 || rand<0.05 
        delt = mvnrnd(deltold, (Df)^2/2)';
    else
        delt = mvnrnd(deltold, exp(ad(1))*(Ds)^2/2*vardelt)';
    end
    logitpsi(Ind1) = X(Ind1,:)*alpha+delt*A(Ind1,Ind1)*(z(Ind1)-muvec(Ind1));
    psi = exp(logitpsi)./(1+exp(logitpsi));      
    LLAdelt = -1/(2*deltsd^2)*(delt'*delt)+z'*logitpsi+ones(n,1)'*log(1-psi);
    V=rand(1);
    if log(V)<(LLAdelt-LLBdelt) && delt>0 && delt<3 %%% prior for delt is truncated normal over (0,3) 
        acceptd(1)=acceptd(1)+1;
    else
        delt = deltold;
        logitpsi = logitpsiold;
        psi = psiold;
    end     

    %%%% update deltT and deltS jointly
    LLBdelt = -1/(2*deltsd^2)*(deltT'*deltT)-1/(2*deltsd^2)*(deltS'*deltS)+z'*logitpsi+ones(n,1)'*log(1-psi);
    deltTold = deltT;
    deltSold = deltS;
    logitpsiold = logitpsi;
    psiold = psi;
    if m<=50 || rand<0.05 
        deltTS = mvnrnd([deltTold, deltSold], (Df)^2/2*eye(2))';
    else
        deltTS = mvnrnd([deltTold, deltSold], exp(ad(2))*(Ds)^2/2*covdeltTS)';
    end
    deltT = deltTS(1);
    deltS = deltTS(2);
    logitpsi(IndT) = X(IndT,:)*alpha+deltT*At(IndT,:)*(z-muvec)+deltS*Alag(IndT,:)*(z-muvec);
    psi = exp(logitpsi)./(1+exp(logitpsi));        
    LLAdelt = -1/(2*deltsd^2)*(deltT'*deltT)-1/(2*deltsd^2)*(deltS'*deltS)+z'*logitpsi+ones(n,1)'*log(1-psi);
    V=rand(1);
    if log(V)<(LLAdelt-LLBdelt) 
        acceptd(2)=acceptd(2)+1;
    else
        deltT = deltTold;
        deltS = deltSold;
        logitpsi = logitpsiold;
        psi = psiold;
    end     
 
    %%% adjust step size every 100th iteration based on acceptance rates
    if ~mod(m,100)         
        for k=1:bp
            if acceptb(k)/m > .6
                ab(k) = ab(k) + min(0.01, (m/10)^(-1/2));
            elseif acceptb(k)/m < .1
                ab(k) = ab(k) - min(0.01, (m/10)^(-1/2));
            end
        end
        for k=1:ap
            if accepta(k)/m > .6
                aa(k) = aa(k) + min(0.01, (m/10)^(-1/2));
            elseif accepta(k)/m < .1
                aa(k) = aa(k) - min(0.01, (m/10)^(-1/2));
            end     
        end
        for k=1:2
            if acceptd(k)/m > .6
                ad(k) = ad(k) + min(0.01, (m/10)^(-1/2));
            elseif acceptd(k)/m < .1
                ad(k) = ad(k) - min(0.01, (m/10)^(-1/2));
            end
        end
    end       
    
    %%%% store output for every MCMth iteration and compute variances to
    %%%% use for adapting
    if ~mod(m,MCM)   
        if m<=MCB     
            betaburn(:,m/MCM)=beta;
            alphaburn(:,m/MCM)=alpha;
            varbeta = var(betaburn(:,(1:m/MCM))')';
            varalpha = var(alphaburn(:,(1:m/MCM))')';
            deltburn(:,m/MCM) = [delt;deltT;deltS];
            vardelt = var(deltburn(1,(1:m/MCM))');
            covdeltTS = cov(deltburn(2:3,(1:m/MCM))');
        elseif m>MCB 
            Zout(:,(m-MCB)/MCM) = z;
            alphaout(:,(m-MCB)/MCM) = alpha;
            betaout(:,(m-MCB)/MCM) = beta;
            deltout(:,(m-MCB)/MCM) = [delt;deltT;deltS];
        end
    end
        
end

filename1 = 'MCMCoutput';
save(filename1,'alphaout','betaout','deltout','Zout');

