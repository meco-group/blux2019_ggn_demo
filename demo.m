import casadi.*
opti = Opti();

w = opti.variable();
f = Function('f',{w},{sin(w)^4 + (exp(w)-2)^2});

opti.minimize(f(w));

opti.solver('ipopt');
sol=opti.solve();
%%
wstar=sol.value(w)
fstar=sol.value(f(w))
sol.value(gradient(f(w),w))

ws=[-3:0.01:2];
plot(ws,full(f(ws))); hold on; 
plot(wstar, fstar,'*r'); hold off;
%%
N = 20;

F = @(x,u) [(1-x(2)^2)*x(1) - x(2) + u; x(1)];

opti = casadi.Opti();

x = opti.variable(2,N+1); % Decision variables for state trajetcory
u = opti.variable(1,N);   % Decision variables for control trajectory

opti.minimize(sumsqr(x)+sumsqr(u)); % Sum-of-squares objective

for k=1:N
  opti.subject_to(x(:,k+1)==F(x(:,k),u(:,k)));
end
opti.subject_to(-1<=u<=1);           % Bounds on control
opti.subject_to(x(:,1)==[0;1]);      % Initial condition

opti.solver('ipopt');
sol = opti.solve();
spy(jacobian(opti.g,opti.x))
