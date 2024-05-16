T=0.1;
A=[1 T;0 1];
B=zeros(2,2);
C=[1 0];
sys=ss(A,B,C,0,T);

des_pole=0.5;
L=(1-des_pole)/T;

Aaug=[1 T 0;0 1 0;0 L*T 1-L*T];
Baug=zeros(3,2);
Caug=eye(3);
O_sys = ss(Aaug, Baug, Caug, 0 , T,'StateName',{'Position' 'Velocity', 'Velocity estim'});

x0=[0 0.4 0];
initial(O_sys,x0,10.0)

%%

T=0.1;

des_pole=0.1;
L=(1-des_pole)/T;

Aaug=[1 T 0;0 1 0;0 L*T 1-L*T];
Baug=[0;T;0];
Caug=eye(3);
O_sys = ss(Aaug, Baug, Caug, 0 , T,'StateName',{'Position' 'Velocity', 'Velocity estim'});

% Define the time vector
t = 0:T:5;

% Define the sinusoidal input
frequency = 1;  % Set the frequency of the sinusoid
u = sin(2*pi*frequency*t);

% Simulate the response of the observer to the same input
[y, t, x] = lsim(O_sys, u, t);

% Brute force differential
diff = zeros(5/T+1,1);
t_t = T:T:5;
diff(1) = x(1,1)/T;
for time = t_t
    id = int16(time/T+1);
    diff(id) = (x(id,1)-x(id-1,1))/T;
end

% Plot the results
subplot(2, 1, 1);
plot(t, x(:,1), 'b');
title('System position');
xlabel('Time (s)');
ylabel('Position');

subplot(2, 1, 2);
plot(t, x(:, 2), 'b', t, x(:, 3), 'r--', t, diff, 'g--');
title('Velocity Estimate Comparison');
xlabel('Time (s)');
ylabel('Velocity');

legend('System', 'Observer', 'Difference');