% By Hery A Mwenegoha (C) 2018
% Unscented Kalman Filter Application 
% Based on the paper by Simon Julier and Jeffrey Uhlmann
% S.J.Julier., J.K.Uhlmann., "A New Extension of the Kalman Filter to
% Nonlinear Systems", The Robotics Research Group, The University of Oxford

% Problem:
% We want to accurately estimate the states of a vehicle that enters the
% atmosphere at high altitude and at very high speed. The vehicle is
% tracked by a radar measuring range and bearing at 10Hz. Strong nonlinear
% forces act on the vehicle including aerodynamic drag, gravity and
% buffeting terms giving a ballistic trajectory at higher altitudes. As
% the density increases drag increases rapidly decelerating the vehicle
% until its trajectory is almost vertical (from a 2D perspective).

% What we Know:
% Drag coefficients are/is known only to a certain confidence interval,
% best if we can estimate this. Some people would call this a dual
% estimation problem where we are interested in estimating both the states
% and parameters (weights) of our system (neural network etc).
% Nonlinear forces will have both temporal and spatial dependencies on our
% states.
% Assume a 2D trajectory for this matter, to make our lives easier. Once
% you have become a fully fledged rocket scientist you can take on a more
% challenging trajectory.
% And Oh, we are using ENU coordinate system. 
% 10Hz range and bearing measuremnets using tracker at location (xr,yr)
% NB: xr - vert.pos yr-hor.pos
% w1 ~ N(0,1m) 
% w2 ~ N(0,17mrad)

% States:
% x1 and x2 - position [m]   x1 - vert.pos, x2 - hor.pos
% x3 and x4 - velocity [m/s] x3 - vert.vel, x4 - hor.vel
% D         - Drag     [N]
% G         - Gravity  [m/s2]
% x5        - aerodynamic property [unit]
% v         - process noise terms []
% B         - ballistic coefficient []

%
% Re-entry object being tracked from the ground
%
Ro  = 6374;        % approximate earth radius at tracker pos 
Bo  = 0.59783;     % paper wrongly uses -ve ballistic coefficient
Ho  = 13.406;      % height parameter
Gmo = 3.9860e5;    % [km3 kg-1 s-2]
xr  = 6374;        % Radar location
yr  = 0;           % Radar Location

%
% Sample Time
%
dt     = 0.05;

%
% State space
%
syms x1 x2 x3 x4 x5 Tor_s v1 v2 v3
x      = [x1 x2 x3 x4 x5].';
R      = sqrt(x1^2 + x2^2);
V      = sqrt(x3^2 + x4^2);
B      =  Bo * exp(x5); 
D      = -B  * exp((Ro-R)/Ho) * V;
G      = -Gmo/R^3;
x1dot  = x3;
x2dot  = x4;
x3dot  = D*x3 + G*x1 + v1;
x4dot  = D*x4 + G*x2 + v2;
x5dot  = v3; 
fd     = [x1dot; x2dot; x3dot; x4dot; x5dot];
F      = x + fd * Tor_s;


fx =@(x1,x2,x3,x4,x5,v1,v2,v3, Tor_s)...
   [x1 + Tor_s*x3;
    x2 + Tor_s*x4;
    x3 - Tor_s*((398600*x1)/(x1^2 + x2^2)^(3/2) - v1 + (59783*x3*exp(3187000/6703 - (500*(x1^2 + x2^2)^(1/2))/6703)*exp(x5)*(x3^2 + x4^2)^(1/2))/100000);
    x4 - Tor_s*((398600*x2)/(x1^2 + x2^2)^(3/2) - v2 + (59783*x4*exp(3187000/6703 - (500*(x1^2 + x2^2)^(1/2))/6703)*exp(x5)*(x3^2 + x4^2)^(1/2))/100000);
    x5 + Tor_s*v3];
F_  = @(x)fx(x(1),x(2),x(3),x(4),x(5),0,0,0,dt);
F_K = @(x,v)fx(x(1),x(2),x(3),x(4),x(5),v(1),v(2),v(3),dt);
%}

%
% Measurement
%
h       = [sqrt((x1-xr)^2 + (x2-yr)^2); atan((x2-yr)/(x1-xr))];
hx      = @(x) [sqrt((x(1)-xr)^2 + (x(2)-yr)^2); atan((x(2)-yr)/(x(1)-xr))];

Phi     = jacobian(F,x);
Phix    = @(x1, x2, x3, x4, x5, Tor_s)...
[                                                                                                                                                                                                    1,                                                                                                                                                                                                    0,                                                                                                                                                                                                            Tor_s,                                                                                                                                                                                                                0,                                                                                                      0;
                                                                                                                                                                                                     0,                                                                                                                                                                                                    1,                                                                                                                                                                                                                0,                                                                                                                                                                                                            Tor_s,                                                                                                       0;
 Tor_s*((1195800*x1^2)/(x1^2 + x2^2)^(5/2) - 398600/(x1^2 + x2^2)^(3/2) + (59783*x1*x3*exp(3187000/6703 - (500*(x1^2 + x2^2)^(1/2))/6703)*exp(x5)*(x3^2 + x4^2)^(1/2))/(1340600*(x1^2 + x2^2)^(1/2))) ,                            Tor_s*((1195800*x1*x2)/(x1^2 + x2^2)^(5/2) + (59783*x2*x3*exp(3187000/6703 - (500*(x1^2 + x2^2)^(1/2))/6703)*exp(x5)*(x3^2 + x4^2)^(1/2))/(1340600*(x1^2 + x2^2)^(1/2))) , 1 - Tor_s*((59783*exp(3187000/6703 - (500*(x1^2 + x2^2)^(1/2))/6703)*exp(x5)*(x3^2 + x4^2)^(1/2))/100000 + (59783*x3^2*exp(3187000/6703 - (500*(x1^2 + x2^2)^(1/2))/6703)*exp(x5))/(100000*(x3^2 + x4^2)^(1/2))),                                                                                                     -(59783*Tor_s*x3*x4*exp(3187000/6703 - (500*(x1^2 + x2^2)^(1/2))/6703)*exp(x5))/(100000*(x3^2 + x4^2)^(1/2)), -(59783*Tor_s*x3*exp(3187000/6703 - (500*(x1^2 + x2^2)^(1/2))/6703)*exp(x5)*(x3^2 + x4^2)^(1/2))/100000;
                              Tor_s*((1195800*x1*x2)/(x1^2 + x2^2)^(5/2) + (59783*x1*x4*exp(3187000/6703 - (500*(x1^2 + x2^2)^(1/2))/6703)*exp(x5)*(x3^2 + x4^2)^(1/2))/(1340600*(x1^2 + x2^2)^(1/2))), Tor_s*((1195800*x2^2)/(x1^2 + x2^2)^(5/2) - 398600/(x1^2 + x2^2)^(3/2) + (59783*x2*x4*exp(3187000/6703 - (500*(x1^2 + x2^2)^(1/2))/6703)*exp(x5)*(x3^2 + x4^2)^(1/2))/(1340600*(x1^2 + x2^2)^(1/2))),                                                                                                     -(59783*Tor_s*x3*x4*exp(3187000/6703 - (500*(x1^2 + x2^2)^(1/2))/6703)*exp(x5))/(100000*(x3^2 + x4^2)^(1/2)), 1 - Tor_s*((59783*exp(3187000/6703 - (500*(x1^2 + x2^2)^(1/2))/6703)*exp(x5)*(x3^2 + x4^2)^(1/2))/100000 + (59783*x4^2*exp(3187000/6703 - (500*(x1^2 + x2^2)^(1/2))/6703)*exp(x5))/(100000*(x3^2 + x4^2)^(1/2))), -(59783*Tor_s*x4*exp(3187000/6703 - (500*(x1^2 + x2^2)^(1/2))/6703)*exp(x5)*(x3^2 + x4^2)^(1/2))/100000;
                                                                                                                                                                      0,                                                                                                                                                                                                    0,                                                                                                                                                                                                                0,                                                                                                                                                                                                                0,                                                                                                       1];
Phi_    = @(x)Phix(x(1), x(2), x(3), x(4), x(5), dt);

                                                                                                                                                                  
H  = jacobian(h,x);
Hx = @(x1, x2, x3, x4, x5, Tor_s)...
        [ (2*x1 - 12748)/(2*((x1 - 6374)^2 + x2^2)^(1/2)),          x2/((x1 - 6374)^2 + x2^2)^(1/2), 0, 0, 0
            -x2/((x2^2/(x1 - 6374)^2 + 1)*(x1 - 6374)^2), 1/((x2^2/(x1 - 6374)^2 + 1)*(x1 - 6374)), 0, 0, 0];

H_ = @(x)Hx(x(1), x(2), x(3), x(4), x(5), dt);

% True intial values 
xo = [6500.4;349.14;-1.8093;-6.7967;0.6932];
Po = [10^-6 0 0 0 0;0 10^-6 0 0 0;0 0 10^-6 0 0;0 0 0 10^-6 0;0 0 0 0 1];

%
% Noise
%
Q  = [...
      0 0         0                 0 0;
      0 0         0                 0 0;
      0 0 2.4064e-5                 0 0;
      0 0         0         2.4064e-5 0;
      0 0         0                 0 0];
  
R   = [...
      1e-3        0;
      0       17e-3];
  
Qk_ = int(Phi * Q * Phi.', Tor_s);
Qk_ = int(Qk_, Tor_s);
Qk_ =@(x1,x2,x3,x4,x5, Tor_s)...
[                                                                                                                                                                                                                                                                                                               (3551219595117973*Tor_s^4)/1770887431076116955136,                                                                                                                                                                                                                                                                                                                                                               0,                                                                                                                                                                                                                                                                                                                                               (3551219595117973*Tor_s^3)/885443715538058477568 - (212302561054937779859*Tor_s^4*x3^2*exp(-(500*(x1^2 + x2^2)^(1/2))/6703)*exp(3187000/6703)*exp(x5))/(88544371553805847756800000*(x3^2 + x4^2)^(1/2)) - (212302561054937779859*Tor_s^4*x4^2*exp(-(500*(x1^2 + x2^2)^(1/2))/6703)*exp(3187000/6703)*exp(x5))/(177088743107611695513600000*(x3^2 + x4^2)^(1/2)),                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       -(212302561054937779859*Tor_s^4*x3*x4*exp(-(500*(x1^2 + x2^2)^(1/2))/6703)*exp(3187000/6703)*exp(x5))/(177088743107611695513600000*(x3^2 + x4^2)^(1/2)),0;
                                                                                                                                                                                                                                                                                                                                                               0,                                                                                                                                                                                                                                                                                                               (3551219595117973*Tor_s^4)/1770887431076116955136,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       -(212302561054937779859*Tor_s^4*x3*x4*exp(-(500*(x1^2 + x2^2)^(1/2))/6703)*exp(3187000/6703)*exp(x5))/(177088743107611695513600000*(x3^2 + x4^2)^(1/2)),                                                                                                                                                                                                                                                                                                                                               (3551219595117973*Tor_s^3)/885443715538058477568 - (212302561054937779859*Tor_s^4*x3^2*exp(-(500*(x1^2 + x2^2)^(1/2))/6703)*exp(3187000/6703)*exp(x5))/(177088743107611695513600000*(x3^2 + x4^2)^(1/2)) - (212302561054937779859*Tor_s^4*x4^2*exp(-(500*(x1^2 + x2^2)^(1/2))/6703)*exp(3187000/6703)*exp(x5))/(88544371553805847756800000*(x3^2 + x4^2)^(1/2)), 0;
 (3551219595117973*Tor_s^3)/885443715538058477568 - (212302561054937779859*Tor_s^4*x3^2*exp(-(500*(x1^2 + x2^2)^(1/2))/6703)*exp(3187000/6703)*exp(x5))/(88544371553805847756800000*(x3^2 + x4^2)^(1/2)) - (212302561054937779859*Tor_s^4*x4^2*exp(-(500*(x1^2 + x2^2)^(1/2))/6703)*exp(3187000/6703)*exp(x5))/(177088743107611695513600000*(x3^2 + x4^2)^(1/2)),                                                                                                                                                                                                         -(212302561054937779859*Tor_s^4*x3*x4*exp(-(500*(x1^2 + x2^2)^(1/2))/6703)*exp(3187000/6703)*exp(x5))/(177088743107611695513600000*(x3^2 + x4^2)^(1/2)), (exp(-(1000*(x1^2 + x2^2)^(1/2))/6703)*(213073175707078380000000000*Tor_s^2*x3^2*exp((1000*(x1^2 + x2^2)^(1/2))/6703) + 213073175707078380000000000*Tor_s^2*x4^2*exp((1000*(x1^2 + x2^2)^(1/2))/6703) + 50768336030189381173242388*Tor_s^4*x3^4*exp(2*x5 + 6374000/6703) + 12692084007547345293310597*Tor_s^4*x4^4*exp(2*x5 + 6374000/6703) - 84921024421975111943600000*Tor_s^3*exp(x5 + (500*(x1^2 + x2^2)^(1/2))/6703 + 3187000/6703)*(x3^2 + x4^2)^(3/2) + 63460420037736726466552985*Tor_s^4*x3^2*x4^2*exp(2*x5 + 6374000/6703) - 84921024421975111943600000*Tor_s^3*x3^2*exp(x5 + (500*(x1^2 + x2^2)^(1/2))/6703 + 3187000/6703)*(x3^2 + x4^2)^(1/2)))/(17708874310761169551360000000000*(x3^2 + x4^2)),                                                                                         (12692084007547345293310597*Tor_s^4*x3*x4^3*exp(-(1000*(x1^2 + x2^2)^(1/2))/6703)*exp(2*x5)*exp(6374000/6703))/(4*(1475739525896764129280000000000*x3^2 + 1475739525896764129280000000000*x4^2)) + (12692084007547345293310597*Tor_s^4*x3^3*x4*exp(-(1000*(x1^2 + x2^2)^(1/2))/6703)*exp(2*x5)*exp(6374000/6703))/(4*(1475739525896764129280000000000*x3^2 + 1475739525896764129280000000000*x4^2)) - (21230256105493777985900000*Tor_s^3*x3*x4*exp(-(500*(x1^2 + x2^2)^(1/2))/6703)*exp(3187000/6703)*exp(x5)*(x3^2 + x4^2)^(1/2))/(3*(1475739525896764129280000000000*x3^2 + 1475739525896764129280000000000*x4^2)), 0;
                                                                                                                                                                                                         -(212302561054937779859*Tor_s^4*x3*x4*exp(-(500*(x1^2 + x2^2)^(1/2))/6703)*exp(3187000/6703)*exp(x5))/(177088743107611695513600000*(x3^2 + x4^2)^(1/2)), (3551219595117973*Tor_s^3)/885443715538058477568 - (212302561054937779859*Tor_s^4*x3^2*exp(-(500*(x1^2 + x2^2)^(1/2))/6703)*exp(3187000/6703)*exp(x5))/(177088743107611695513600000*(x3^2 + x4^2)^(1/2)) - (212302561054937779859*Tor_s^4*x4^2*exp(-(500*(x1^2 + x2^2)^(1/2))/6703)*exp(3187000/6703)*exp(x5))/(88544371553805847756800000*(x3^2 + x4^2)^(1/2)),                                                                                         (12692084007547345293310597*Tor_s^4*x3*x4^3*exp(-(1000*(x1^2 + x2^2)^(1/2))/6703)*exp(2*x5)*exp(6374000/6703))/(4*(1475739525896764129280000000000*x3^2 + 1475739525896764129280000000000*x4^2)) + (12692084007547345293310597*Tor_s^4*x3^3*x4*exp(-(1000*(x1^2 + x2^2)^(1/2))/6703)*exp(2*x5)*exp(6374000/6703))/(4*(1475739525896764129280000000000*x3^2 + 1475739525896764129280000000000*x4^2)) - (21230256105493777985900000*Tor_s^3*x3*x4*exp(-(500*(x1^2 + x2^2)^(1/2))/6703)*exp(3187000/6703)*exp(x5)*(x3^2 + x4^2)^(1/2))/(3*(1475739525896764129280000000000*x3^2 + 1475739525896764129280000000000*x4^2)), (exp(-(1000*(x1^2 + x2^2)^(1/2))/6703)*(213073175707078380000000000*Tor_s^2*x3^2*exp((1000*(x1^2 + x2^2)^(1/2))/6703) + 213073175707078380000000000*Tor_s^2*x4^2*exp((1000*(x1^2 + x2^2)^(1/2))/6703) + 12692084007547345293310597*Tor_s^4*x3^4*exp(2*x5 + 6374000/6703) + 50768336030189381173242388*Tor_s^4*x4^4*exp(2*x5 + 6374000/6703) - 84921024421975111943600000*Tor_s^3*exp(x5 + (500*(x1^2 + x2^2)^(1/2))/6703 + 3187000/6703)*(x3^2 + x4^2)^(3/2) + 63460420037736726466552985*Tor_s^4*x3^2*x4^2*exp(2*x5 + 6374000/6703) - 84921024421975111943600000*Tor_s^3*x4^2*exp(x5 + (500*(x1^2 + x2^2)^(1/2))/6703 + 3187000/6703)*(x3^2 + x4^2)^(1/2)))/(17708874310761169551360000000000*(x3^2 + x4^2)), 0;
                                                                                                                                                                                                                                                                                                                                                               0,                                                                                                                                                                                                                                                                                                                                                               0,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             0,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             0, 0];
 

QK = @(x)Qk_(x(1),x(2),x(3),x(4),x(5), dt);

  
Qv  = sqrt(Q);
Rv  = chol(R);


% Time
Tmax   = 200;
time   = 0:dt:Tmax; 
Len    = length(time);  
Nmonte = 100;            % monte-carlo simulation runs


% record data
xrecord         = zeros(Nmonte,Len,5);
xrecord2        = zeros(Nmonte,Len,5);
dxrecord        = zeros(Nmonte,Len,5);
SDrecord        = zeros(Nmonte,Len,5);

% Time Variables
rng('default')
TimeLapse       = 0;
Interval        = 0;
for monte_epoch=1:Nmonte
    fprintf('Monte: %1u, Interval: %1.2f s, TimeLapse: %1.2f s \n', monte_epoch,Interval, TimeLapse);
    tic
    
    % Best Guess initial values
    Pohat = [...
             10^-6     0     0     0 0;
             0     10^-6     0     0 0;
             0         0 10^-6     0 0;
             0         0     0 10^-6 0;
             0         0     0     0 1];
    xohat = [6500.4; 349.14; -1.8093; -6.7967; 0];
   
    xrecord(monte_epoch, 1,:)  = xo.';
    xrecord2(monte_epoch,1,:)  = xohat.';
    dxrecord(monte_epoch,1,:)  = (xo - xohat).';

    for i=1:5
    SDrecord(monte_epoch,1,i) = sqrt(Pohat(i,i));
    end
    x     = xo;
    xk    = xohat;
    count = 0;
    
    % buffeting accelerations
    v1 = 0; v2 = 0; v3 = 0;
    
for epoch=1:Len
    Tor_s = dt;
    count = count + 1;
        
    %
    % 10Hz measuremnet
    %
    correct = false;
    predict = true;
    if count == 2
    count   = 0;
    y       = hx(x) + Rv * randn(2,1);
    correct = true;
    end   


    %
    % EKF
    %
    if correct
        yk = hx(xk);
        HL = H_(xk);%HL=eval(H);[yk, HL]   = jaccsd(hx,xk);
        Kk    = Pohat*HL'/(HL*Pohat*HL' + R);
        xk    = xk + Kk * (y-yk);          
        Pohat = Pohat - Kk*HL*Pohat;
        Pohat = 1/2*(Pohat + Pohat.');
    end    
       
    %
    % records
    %
    dx                                = sqrt((x - xk).^2);
    dxrecord(monte_epoch, epoch,:)    = dx.';
    xrecord2(monte_epoch, epoch,:)    = xk.'; 
    xrecord(monte_epoch,epoch,:)      = x.';
    for i=1:5
        SDrecord(monte_epoch,epoch,i) = sqrt(Pohat(i,i));
    end
    
    %
    % EKF Predict
    %
    if predict  
        PhiL  = Phi_(xk);       % PhiL = eval(Phi) %[xk,PhiL]=jaccsd(F_,xk);%
        Pohat = PhiL * Pohat  * PhiL.' +  Q*Tor_s*Tor_s;%
        Pohat = 1/2*(Pohat + Pohat.');    
    end 
    
    
    % 
    % next iteration v=zeros(3,1);v(1)= Qv(3,3); v2 = Qv(4,4); v3 = 0;
    %
    %v=zeros(3,1);v(1)= Qv(3,3)*randn; v2 = Qv(4,4)*randn; v3 = 0;
    
    x  = F_K(x,v) + Qv*0.18*randn(5,1); %x = eval(F);
    xk = F_(xk);         %x_2=eval(F);

end
Interval  = toc;
TimeLapse = TimeLapse + Interval;
end

%%
%plot(xrecord2(:,1))
figure(1);
subplot(211);
plot(xrecord(monte_epoch,:,1), xrecord(monte_epoch,:,2), 'Color', 'r');
hold on;
plot(xrecord2(:,:,1), xrecord2(:,:,2), 'Color', [0,0,0] + 0.3);
title('Re-entry Position');
xlabel('x_1 [km]');
ylabel('x_2 [km]');
legend('True', 'Realisations');

subplot(212);
if monte_epoch > 1
plot(time, sqrt(sum(dxrecord(1:monte_epoch,:,1).^2)/monte_epoch).^2);
hold on
plot(time, sqrt(sum(SDrecord(1:monte_epoch,:,1).^2)/monte_epoch).^2);
else
plot(time, sqrt(dxrecord(1:monte_epoch,:,1).^2));
hold on
plot(time, sqrt(SDrecord(1:monte_epoch,:,1).^2));    
end
hold off;
title('Standard Deviation');
xlabel('Time [s]');
ylabel('Position standard deviation [km]');
legend('MSE', '1 \sigma');


%% Variance Logarithmic Scale Plots
figure(2);
subplot(131);
plot(time, sqrt(sum(dxrecord(1:monte_epoch,:,1).^2)/monte_epoch).^2);
hold on
plot(time, sqrt(sum(SDrecord(1:monte_epoch,:,1).^2)/monte_epoch).^2);
hold off
set(gca, 'YScale', 'log');
title('Mean squared error and x1 variance');
xlabel('Time [s]');
ylabel('Error [km^2]');
legend('MSE', '1 \sigma prediction');

subplot(132);
plot(time, sqrt(sum(dxrecord(1:monte_epoch,:,3).^2)/monte_epoch).^2);
hold on
plot(time, sqrt(sum(SDrecord(1:monte_epoch,:,3).^2)/monte_epoch).^2);
hold off
set(gca, 'YScale', 'log');
title('Mean squared error and x3 variance');
xlabel('Time [s]');
ylabel('Velocity variance (km/s)^2');
legend('MSE', '1 \sigma prediction');

subplot(133);
plot(time, sqrt(sum(dxrecord(1:monte_epoch,:,5).^2)/monte_epoch).^2);
hold on
plot(time, sqrt(sum(SDrecord(1:monte_epoch,:,5).^2)/monte_epoch).^2);
hold off
set(gca, 'YScale', 'log');
title('Mean squared error and x5 variance');
xlabel('Time [s]');
ylabel('Coefficient variance');
legend('MSE', '1 \sigma prediction');
