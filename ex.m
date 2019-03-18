% Solve an unconstrained convex-over-nonlinear problem
%
%  min   Phi(F(w))
%   w
%
%  The example considers a parameter estimation problem with
%    - F    defined by a simulation of a nonlinear system
%    - Phi  defined to be Huber-like
%
% See: blux.casadi.org

import casadi.*

%% Set-up a nonlinear fitting problem F(w)
% Unknown parmeters of dynamic system
a     = MX.sym('a');
b     = MX.sym('b');
alpha = MX.sym('alpha');
beta  = MX.sym('beta');
gamma = MX.sym('gamma');
delta = MX.sym('delta');
% w in R^6
w = [a;b;alpha;beta;gamma;delta];

% Discrete-time nonlinear dynamic system (predator-prey)
s = MX.sym('s',2); % s in R^2
s_next = [(a*s(1)-alpha*s(1)*s(2))/(1+gamma*s(1));
          (b*s(2)+beta*s(1)*s(2))/(1+delta*s(2))];

% Function mapping from state (R^2) and parameter (R^6)
%   to state after a timestep (R^2)
m = Function('m',{s,w},{s_next})

% Simulate over 100 time steps with symbolic w
s = [10;0.1];
y_sim = {};
for i=1:100
    s = m(s,w);
    y_sim{end+1} = s;
end
y_sim = [y_sim{:}];

% Nonlinear map F: R^6 (parameters) -> R^200 (error between simulation and data)
load('y.mat');
error = (y_sim-y).*repmat([0.1;10],1,100);
F = Function('F',{w},{error(:)});

%% Choosing a convex metric to optimize over

% Convex map Phi: R^200 (error vector) -> R (scalar matric)
E = MX.sym('E',200);
Phi = Function('Phi',{E},{sum(sqrt(1^2+E.^2))/200})
%Phi = Function('Phi',{E},{E'*E})

% Total objective: R^6 -> R
f = Function('f',{w},{Phi(F(w))});

%% Solve using GGN

% Gradient of f wrt w: R^6 -> R^6
f_grad = Function('f',{w},{gradient(f(w),w)});

% Jacobian of F wrt w: R^6 -> R^(200x6)
J    = Function('J',{w},{jacobian(F(w),w)})
% Hessian of Phi map: R^200 -> R^(200x200)
HPhi = Function('HPhi',{E},{hessian(Phi(E),E)})
% GGN Hessian: R^6 -> R^(6x6)
BGGN = Function('BGGN',{w},{J(w)'*HPhi(F(w))*J(w)});

w_exact = [1.4;0.80;0.3;0.02;0.04;0.02];
w = w_exact*1.01;
%w = [1.43;0.99;0.22;0.022;0.022;0.011];
maxit = 20;
for i=1:maxit
    dw = full(-BGGN(w)\f_grad(w));
    fprintf('it %d: ||dw|| %e\n',i,norm(dw));
    w = w + dw;
end