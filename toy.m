close all
clc

import casadi.*

w = MX.sym('w');
x = MX.sym('x');

m = Function('m',{x,w},{sin(w+x)+0.75*(w+x)});

eps = 0.5;
xe = [-eps 0 eps]';
ye = [0 0 1]';

xs = linspace(-pi,2*pi,1000);

figure
hold on
plot(xs,full(m(xs,0)));
plot(xe,ye,'o');
title('m(x,0)')
%%
delta = 0.1;

F = Function('F',{w},{m(xe,w)-ye});
J = Function('J',{w},{jacobian(F(w),w)});


E = MX.sym('E',3);
Phi = Function('Phi',{E},{sum(sqrt(delta^2+E.^2))/3});

f = Function('f',{w},{Phi(F(w))});

f_grad = Function('f',{w},{gradient(f(w),w)});

HPhi = Function('HPhi',{E},{hessian(Phi(E),E)});
BGGN = Function('BGGN',{w},{J(w)'*HPhi(F(w))*J(w)});

alpha = Function('alpha',{w},{norm_fro(hessian(f(w),w)-BGGN(w))/norm_fro(BGGN(w))});

opti = Opti()
W = opti.variable();
opti.minimize(f(W));
opti.set_initial(W,4);
opti.solver('ipopt');
sol = opti.solve();
wbad = sol.value(W);

ws = linspace(-pi,2*pi,1000);

f_sampled = full(f(ws));
figure
scale_f = norm(full(f(ws)),'inf');
scale_alpha = norm(full(alpha(ws)),'inf');
scale_alpha = 1;
semilogy(ws,full(alpha(ws))/scale_alpha);
hold on
semilogy(ws,full(f(ws))/scale_f);
semilogy(wbad,full(f(wbad))/scale_f,'o');
semilogy(wbad,full(alpha(wbad))/scale_alpha,'o');
legend('\alpha(w)','f(w)')
title('f(w)')
%%

wk = 0.01;
for i=1:10
    dw = full(-BGGN(wk)\f_grad(wk));
    fprintf('it %d: ||dw|| %e\n',i,norm(dw));
    wk = wk + dw;
end

wk_good = wk

wk_bad = wbad
%%
ymr = 2*m(xe,wk_good)-ye;

F = Function('F',{w},{m(xe,w)-ymr});
J = Function('J',{w},{jacobian(F(w),w)});

f = Function('f',{w},{Phi(F(w))});

HPhi = Function('HPhi',{E},{hessian(Phi(E),E)});
BGGN = Function('BGGN',{w},{J(w)'*HPhi(F(w))*J(w)});

alpha = Function('alpha',{w},{norm_fro(hessian(f(w),w)-BGGN(w))/norm_fro(BGGN(w))});

f_sampled_good = full(f(ws));

%%
ymr = 2*m(xe,wk_bad)-ye;

F = Function('F',{w},{m(xe,w)-ymr});
J = Function('J',{w},{jacobian(F(w),w)});

f = Function('f',{w},{Phi(F(w))});


HPhi = Function('HPhi',{E},{hessian(Phi(E),E)});
BGGN = Function('BGGN',{w},{J(w)'*HPhi(F(w))*J(w)});

alpha = Function('alpha',{w},{norm_fro(hessian(f(w),w)-BGGN(w))/norm_fro(BGGN(w))});

f_sampled_bad = full(f(ws));

%%
figure
hold on
plot(ws,f_sampled)
plot(ws,f_sampled_good)
plot(ws,f_sampled_bad)

legend('f(w)','f_{mirror,good}(w)','f_{mirror,bad}(w)')
