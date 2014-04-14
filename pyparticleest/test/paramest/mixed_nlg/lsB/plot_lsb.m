load test_lsb.mat
t=1:100;

vals_xi = reshape(vals(1,:,1:end-1), [], 100);
vals_theta = reshape(vals(2,:,1:end-1), [], 100);
svals_xi = reshape(svals(1,:,1:end-1), [], 100);
svals_theta = reshape(svals(2,:,1:end-1), [], 100);
C=[[ 0.0, 0.04, 0.044, 0.08],];
theta = 25 + C*z;

%xi
figure(1)
clf
hold on
plot(t,svals_xi,'-', 'Color', [0.65 0.65 0.65],'LineWidth',1)
plot(t,vals_xi, 'x', 'Color', [0.5 0.5 0.5], 'MarkerSize',2.5)
plot(t,e,'--', 'Color', [0.0 0.0 0.0])

%theta
figure(2)
clf
hold on
plot(t,svals_theta,'-', 'Color', [0.65 0.65 0.65],'LineWidth',1)
plot(t,vals_theta, '.', 'Color', [0.5 0.5 0.5], 'MarkerSize',2.0)
plot(t,theta,'--', 'Color', [0.0 0.0 0.0])