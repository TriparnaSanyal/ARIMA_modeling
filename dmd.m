clear all;
xi=linspace(-pi,pi,64);
t=linspace(0,5,1000);
dt=t(2)-t(1);
[Xgrid,T]=meshgrid(xi,t);
%f1=10.*tanh(Xgrid/2)./cosh(Xgrid/2).*exp((0.1+2.8*j)*T);%sech(Xgrid+3).*(exp(j*2.3*T));%10.*tanh(Xgrid/2)./cosh(Xgrid/2).*exp((0.1+2.8*j)*T);
%plot(real(f1(1,:)));
%f2=(20-Xgrid.^2/5).*exp((-0.05+2.3*j)*T);%0;%);%sech(Xgrid).*tanh(Xgrid).*(2*exp(1j*2.8*T));%(20-Xgrid.^2/5).*exp((-0.05+2.3*j)*T);%0;%);
%f3=Xgrid.*exp(0.6*j*T);
%f=f1+f2+f3;
%X=f';
%X = readmatrix('file_series\bigfile.dat','NumHeaderLines',1);
%%size=[1 inf]
%X=fscanf(fileID,filespec,size);
%fclose(fileID);

%plot(real(X1(:,1)),'xr')
%hold on;
X=[];
d = 'C:\Users\Nairita Pal\OneDrive\Documents\MATLAB';
filePattern = fullfile(d, '*.txt');
file = dir(filePattern);
for k = 1 : numel(file)
baseFileName = file(k).name;
fullFileName = fullfile(d, baseFileName);
F = readtable(fullFileName);
X(:,k) = table2array(F);
end
%plot(X(:,999))
X1=X(:,1:end-1);
X2=X(:,2:end);
[U,S,V]=svd(X1,'econ');
svds(X1,20)
r=14;
%plot(U(:,:))
%plot(real(V(:,1:r)))
Ur= U(:,1:r);
Sr= S(1:r,1:r);
Vr= V(:,1:r);
%plot(real(Ur(:,:)));
X_check=U*S*V';
%plot(real(X_check(:,3)),'b')
Atilde=Ur'*X2*Vr/Sr;
[W,D]=eig(Atilde);
phi = X2*Vr/Sr*W;
max(real(phi),[],"all")
lambda=diag(D);
omega=log(lambda)/dt; %continuous time eigenvalues
x1=X1(:,1);
b=phi\x1;
time_dynamics=zeros(r,length(t));
for iter =1:length(t)
time_dynamics(:,iter)=(b.*exp(omega*t(iter)));
end
Xdmd = phi*time_dynamics;
hold on;
plot(real(Xdmd(:,:)))
%plot(xi,real(phi(:,1)))
%plot(real(phi(:,1)))
%plot(xi,real(phi(:,:)))
%plot(xi,real(phi(,1)/0.2619),'x',xi,real(phi(:,2)/0.1148),'x',xi,real(phi(:,3)/0.2031),'x')
%plot(xi,real(f1(1,:)),'b',xi,real(f2(1,:)),'r',xi,real(f3(1,:)));
%plot(xi,real(f1(1,:)),Linewidth=2)
%plot(xi,(real(phi(:,2))),'--',Linewidth=2)
%plot(xi,(real(phi(:,1))),'--',xi,(real(phi(:,2))),'-x',xi,(real(phi(:,3))),'-o')
%hold on;
%max(f1(1,:))
%max(phi(:,2))
%plot(xi,real(f1(1,:)/4.9933)),'b',xi,real(f2(1,:)/19.9917),'r',xi,real(f3(1,:)/10));%%normalization
% plot(real(omega),imag(omega),'o');
