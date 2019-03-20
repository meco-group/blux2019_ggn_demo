% This example compares GGN and SCP to solve an optimization problem:
%
%  min   phi(w)
%   w
%      s.t.  G(w)=0
%
%  * w in R^nw
%  * w contains both parameters-to-be-estimated and a state trajectory
%  * G: gap-closing constraints from multiple-shooting with a system S
%  * S: a nonlinear discrete system map (state x parameter)->(state)
%  * y: a given matrix of (noisy) state measurements
%  * phi: a convex map: Huber-like
%         sum_i sqrt(sigma^2+E_i(w)^2)
%
% See: benelux.casadi.org

close all
clc
import casadi.*

load('y.mat');
figure
hold on
plot(y(1,:),'ro')
plot(y(2,:),'bo')
hold off
%% Set-up a nonlinear fitting problem
Ns    = 100; % Number of simulation steps (= number of observations)
sigma = 0.1; % Huber tuning term

% Unknown parameters of dynamic system
alpha = MX.sym('alpha');
beta  = MX.sym('beta');
gamma = MX.sym('gamma');
delta = MX.sym('delta');
p     = [alpha;beta;gamma;delta]; % p in R^4

% Discrete-time nonlinear dynamic system (predator-prey)
x      = MX.sym('x',2); % symbolic state, in R^2
x_next = [(x(1)-alpha*x(1)*x(2))/(1+gamma*x(1));
          (x(2)+ beta*x(1)*x(2))/(1+delta*x(2))];

% System dynamics: R^2 (state) x R^4 (parameter) -> R^2 (state at next)
S = Function('S',{x,p},{x_next})

X = MX.sym('X',2,Ns+1); % Symbolic state trajectory, in R^(2 x Ns+1)

% Decision variable structure w in R^nw : nw = 4+2(Ns+1)
w = [p;X(:)];

size(y) % Noisy state observations R^(2 x Ns)
% Error between state and observation: R^nw -> R^(2Ns)
err = X(:,2:Ns+1)-y;
E = Function('E',{w},{ err(:) });

% Express multiple shooting gaps
k = 1:Ns;
mshooting_gaps = X(:,k+1)-S(X(:,k),p);
size(mshooting_gaps) % Gaps are in R^(2 x Ns)

% Constraint: R^nw -> R^(2Ns)
G = Function('G',{w},{ mshooting_gaps(:) })

figure
spy(jacobian(G(w),w))
title('Constraint Jacobian: R^{nw} -> R^{2Ns x nw}')
JG = Function('JG',{w},{ jacobian(G(w),w) });

%% Initial w

% We have a vague idea of the values of p
pinit = [0.03;0.25;0.35;0.05];

% We can initialize with the measurements y!
winit = [pinit;y(:,1);y(:)];

%% SCP approach

% Jacobian of error Function: R^nw -> R^(2Ns x nw)
JE = Function('JE',{w},{ jacobian(E(w),w) });

wk = winit;
for k=1:10
    opti = Opti('conic');

    % Decision variables of conic problem
    dw = opti.variable(numel(w));
    s  = opti.variable(2*Ns,1);   % Slack
    
    % Objective: sum of slacks
    opti.minimize(sum(s));
    
    % Second-order cone constraints,
    %  arising from objective   sum_i sqrt(sigma^2+E_i(w)^2)
    E_lin = E(wk)+JE(wk)*dw;
    for i=1:2*Ns
        opti.subject_to( norm([sigma;E_lin(i)]) <= s(i) );
    end

    % Linear equality constraints (cfr. gap-closing multiple shooting)
    G_lin = G(wk)+JG(wk)*dw;
    opti.subject_to( G_lin==0 );

    % Choose a conic solver
    options = struct;
    options.superscs.max_iters = 1e5;
    options.superscs.verbose   = 0;
    options.superscs.eps       = 1e-9;
    opti.solver('superscs',options);
    
    % Solve
    sol = opti.solve();
    dw  = sol.value(dw);
    
    % Take a full step
    wk = wk+dw;
    
    fprintf('SCP it %d: ||dw|| %e\n',k, norm(dw));
end

p_opt_SCP = wk(1:4);

%% GGN approach

% Convex map: R^nw -> R
phi  = Function('phi',{w},{ sum(sqrt(sigma^2+E(w).^2))/(2*Ns) });

% Jacobian of phi: R^nw -> R^(1 x nw)
Jphi = Function('Jphi',{w},{ jacobian(phi(w),w) }); % redact

figure
spy(hessian(phi(w),w))
title('Hessian of phi: R^{nw} -> R^{nw x nw}')
Hphi = Function('Hphi',{w},{ hessian(phi(w),w) }); % redact

wk = winit;
for k=1:10
    opti = Opti('conic');

    % Decision variables of quadratic problem
    dw = opti.variable(numel(w));

    % Quadratic approximation to objective
    obj_lin = phi(wk)+Jphi(wk)*dw+1/2*dw'*Hphi(wk)*dw;  % redact
    opti.minimize(obj_lin); % redact

    % Linear equality constraints (cfr. multiple shooting)
    G_lin = G(wk)+JG(wk)*dw;
    opti.subject_to( G_lin==0 );

    % Choose a QP solver
    options = struct;
    options.print_iter   = false;
    options.print_header = false;
    opti.solver('qrqp',options);
    
    % Solve
    sol = opti.solve();
    dw  = sol.value(dw);
    
    % Take a full step
    wk = wk+dw;
    
    fprintf('GGN it %d: ||dw|| %e\n',k, norm(dw));
end

p_opt_GGN = wk(1:4);

%% Comparison

disp('Difference between p_opt_GGN and p_opt_SCP')
p_opt_GGN-p_opt_SCP

%% Plotting
figure
hold on
Sim = S.mapaccum(100);
plot(y(1,:),'ro')
plot(y(2,:),'bo')
sim = full(Sim([1;1],p_opt_GGN)); 
plot(sim(1,:),'r','linewidth',3);
plot(sim(2,:),'b','linewidth',3);
