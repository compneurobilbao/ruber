function y = oneGauss(beta,x);
y=beta(1)*exp(-(x-beta(2)).^2/abs(beta(3)));
