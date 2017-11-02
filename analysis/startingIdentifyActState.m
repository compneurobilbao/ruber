% fs = 500
%    lowcut = 0.05
%    highcut = 70
%    output = opj(output_path, 'filtered', file)
%
%    lowcut = 0.5
%    highcut = 3
%    output = opj(output_path, 'delta', file)
%
%    lowcut = 3
%    highcut = 7
%    output = opj(output_path, 'theta', file)
%
%    lowcut = 7
%    highcut = 13
%    output = opj(output_path, 'alpha', file)
%
%    lowcut = 13
%    highcut = 30
%    output = opj(output_path, 'beta', file)
%
%    lowcut = 30
%    highcut = 70
%    output = opj(output_path, 'gamma', file)
%
%    lowcut = 70
%    highcut = 249
%    output = opj(output_path, 'gamma_high', file)


%% define original signal
si2anOr=data(:,10);
%% perform bandpass filter
lp=30;
hp=70;
frSam=500;
numberCyclesInConv=4;
si2an=bandPassButterFilter(si2anOr,lp,hp,frSam);
si2anFilt=si2an;
%% we do convolution on energy signal
ptConv=ceil(numberCyclesInConv*frSam/lp);
si2anEnergy=conv(si2an.*si2an,ones(1,ptConv),'valid');
si2an=si2anEnergy;
%% this would be for hilbert
%     si2anEnergy=abs(hilbert(si2an));
%% plot original signal and filtered one
f1=figure;
plot(si2anOr)
hold on
plot(si2anFilt,'r-')
%% plot the energy or hilbert or whatelse; choose if in log10 space or not
f2=figure;
hold on
plot(log10(si2anEnergy))


%% do gaussian fit; decide in next line if in log10 space or not
si2an=log10(si2an);
s=max(si2an)-min(si2an);
if size(s,1)>1
    [y,X]=hist(si2an,[min(si2an):max(si2an)]);
else
    [y,X]=hist(si2an,[min(si2an):s/500 :max(si2an)]);
end
X=X((y>max(y)*0.005));
y=y((y>max(y)*0.005));
figure
bar(X,y)
[beta] = nlinfit(X,y,'oneGauss',[max(y) mean(si2an) 2*(std(si2an))]);
me=beta(2);
st=sqrt(beta(3)/2);
hold on
plot(X,oneGauss(beta,X),'r-')

%% put the threshold on the energy signal or hilbert or whatelse
figure(f2)
plot([ 1 length(si2an)],(me+3*st)*[1 1 ],'k-')
plot([ 1 length(si2an)],(me-2*st)*[1 1 ],'k-')


