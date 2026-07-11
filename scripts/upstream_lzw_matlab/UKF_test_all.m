%% 手动实现UKF的完整代码
clc; clear; close all;

%% 导入数据
load('data_mark.mat');
load('a.mat');
load('b.mat');
load('bestX.mat');
data_mark(3700:3900,:) = [];

% data_aver = data_mark(:,[2:end]);
data_aver = data_mark(:,[2:13]);
bestX=[1.907e-07  0.002	 0.0001  0.01249 1.5272];
%% UKF预测老化参数
% 1. 初始化参数
dt = 1;         % 时间步长
% Q = diag([1e-14, 1e-8, 1e-8]);  % 过程噪声协方差
% Q = diag([1e-20, 1e-14, 1e-12]);  % 过程噪声协方差    调参
% R = 1e-8;       % 测量噪声方差  原始数据
R = 9e-1;       % 测量噪声方差  原始数据  R越大，数据波动越小，越相信开始的数据，但准确度会降低
x = bestX(:,[1:3])';
% P = diag([1e-20, 1e-14, 1e-12]);
% Q = diag([4e-18, 1e-8, 3e-12]);

Q = diag([4e-19, 1e-9, 1e-12]);  % 真实值
% Q = diag([4e-18, 1e-8, 1e-11]); 
P = Q;

% 数据准备
num = size(data_aver, 1);
I = data_aver(1:num,1)/406;
V = data_aver(1:num,2)/170;
T = (data_aver(1:num,8)+data_aver(1:num,9))/2;
inner = bestX(:,[4,5]);

% UKF参数
n = 3;          % 状态维度
alpha = 0.02;   % 扩展系数
beta = 2;       % 分布参数
kappa = 0;      % 缩放参数
lambda = alpha^2*(n + kappa) - n;

% 存储变量初始化
j0_estimates = zeros(1, num);
jh_estimates = zeros(1, num);
R_estimates = zeros(1, num);

%% UKF主循环
for k = 1:num
    % 当前时刻的输入参数
    current_T = T(k,:);
    current_I = I(k);
    current_a = a(k,:);
    current_b = b(k,:);

    % ========== UKF核心算法 ==========
    % (1) 生成Sigma点
    [X, Wm, Wc] = manualSigmaPoints(x, P, lambda, alpha, beta);

    % (2) 时间更新（预测）
    X_pred = X; % 状态转移方程x(k+1)=x(k)
    % X_pred(1,:) = X_pred(1,:) - 1e-10;  % j0 每步微降
    % X_pred(2,:) = X_pred(2,:) - 1e-8;  % j0 每步微降
    x_pred = X_pred * Wm';
    P_pred = (X_pred - x_pred) * diag(Wc) * (X_pred - x_pred)' + Q;

    % (3) 测量更新
    % 生成测量Sigma点
    Z_pred = zeros(1, size(X_pred,2));
    for i = 1:size(X_pred,2)
        Z_pred(i) = IV_model(current_T, X_pred(:,i), current_I,...
                            current_a, current_b, inner);
    end
    z_pred = Z_pred * Wm';

    % 计算协方差
    Pzz = (Z_pred - z_pred) * diag(Wc) * (Z_pred - z_pred)' + R;
    Pxz = (X_pred - x_pred) * diag(Wc) * (Z_pred - z_pred)';

    % 卡尔曼增益
    K = Pxz / Pzz;

    % 状态更新
    x = x_pred + K*(V(k) - z_pred);
    P = P_pred - K*Pzz*K';

    % 存储结果
    j0_estimates(k) = x(1);
    jh_estimates(k) = x(2);
    R_estimates(k) = x(3);
end

%% 独立趋势图绘制（三张独立figure）
% 1. 交换电流密度 j0
figure('Position', [100, 400, 800, 400])
plot(j0_estimates, 'b', 'LineWidth', 2)
xlabel('时间步长', 'FontSize', 12)
ylabel('j0 (A/cm²)', 'FontSize', 12)
title('交换电流密度下降趋势', 'FontSize', 14)
grid on
set(gca, 'FontSize', 11)
% ylim([0.9*min(j0_estimates), 1.1*max(j0_estimates)])

% 2. 渗氢电流 jh
figure('Position', [200, 200, 800, 400])
plot(jh_estimates, 'r', 'LineWidth', 2)
xlabel('时间步长', 'FontSize', 12)
ylabel('jh (A/cm²)', 'FontSize', 12)
title('渗氢电流上升趋势', 'FontSize', 14)
grid on
set(gca, 'FontSize', 11)
% ylim([0.9*min(jh_estimates), 1.1*max(jh_estimates)])

% 3. 欧姆电阻 R
figure('Position', [300, 50, 800, 400])
plot(R_estimates, 'g', 'LineWidth', 2)
xlabel('时间步长', 'FontSize', 12)
ylabel('R (Ω·cm²)', 'FontSize', 12)
title('欧姆电阻上升趋势', 'FontSize', 14)
grid on
set(gca, 'FontSize', 11)
% ylim([0.9*min(R_estimates), 1.1*max(R_estimates)])


%% （后续可视化部分与原始代码相同，此处省略...）
% 4. 绘制老化参数的变化
figure;

% 绘制交换电流密度 (j0) 的变化
subplot(3,1,1);
plot(1:length(j0_estimates), j0_estimates,  'LineWidth', 1);
xlabel('时间步长');
ylabel('交换电流密度 j0');
title('交换电流密度随时间的变化');
grid on;

% 绘制欧姆电阻 (R) 的变化
subplot(3,1,3);
plot(1:length(R_estimates), R_estimates, 'LineWidth', 1);
xlabel('时间步长');
ylabel('欧姆电阻 R');
title('欧姆电阻随时间的变化');
grid on;

% 绘制渗氢电流 (jh) 的变化
subplot(3,1,2);
plot(1:length(jh_estimates), jh_estimates,  'LineWidth', 1);
xlabel('时间步长');
ylabel('渗氢电流 jh');
title('渗氢电流随时间的变化');
grid on;

% 调整图像显示
sgtitle('燃料电池老化参数预测');

%% 把三个老化参数带入电压模型中看结果
% x_est_1 = [j0_estimates; jh_estimates; R_estimates];
x_est = [j0_estimates; jh_estimates; R_estimates];
v_test = zeros(1, length(I));
 for k = 1:length(I)
    v_test(k) = IV_model(T(k,:), x_est(:,k), I(k,:), a(k,:), b(k,:), inner);
 end
 t=  1:length(I);
 figure
 plot(t, v_test, 'r-');
 hold on 
 plot(t, V, 'b-');
 legend('训练集预测值','实际电压值');
 ylabel('电压值(V)');
 xlabel('时间（h）');
ylim([0.5 1.2]);   %Y轴的范围


m = size(V,1) - 500 : size(V,1);
figure
plot(m, v_test(size(V,1) - 500 : size(V,1)), 'r-','LineWidth', 2);
hold on
plot(m, V(size(V,1) - 500 : size(V,1)), 'b-','LineWidth', 2);
 ylabel('电压值(V)');
 xlabel('时间（h）');
ylim([0.5 1.2]);   %Y轴的范围

%% 画出误差分布图
error = zeros(1, length(I)); 
t = 1:size(v_test,2);
for i = 1:length(I)
   error(i) = abs((V(i)- v_test(i)))/V(i);
end
figure
plot(t, error)
title('电压误差变化');

%% 评价指标
% 计算电压预测误差统计指标
abs_error = abs(V - v_test');
rel_error = abs_error ./ V;

% 1. 均方根误差（RMSE）
rmse_voltage = sqrt(mean((V - v_test').^2));

% 2. 平均绝对误差（MAE）
mae_voltage = mean(abs_error);

% 3. 平均绝对百分比误差（MAPE）
mape_voltage = mean(rel_error) * 100; 

% 4. 决定系数（R²）
ss_total = sum((V - mean(V)).^2);
ss_residual = sum((V - v_test').^2);
r2_voltage = 1 - (ss_residual/ss_total);

% 5. 误差分布统计
error_stats = struct(...
    'MaxAbsError', max(abs_error), ...
    'MinAbsError', min(abs_error), ...
    'StdError', std(abs_error)...
);

% 打印结果
fprintf('电压预测性能指标：\n');
fprintf('RMSE = %.4f V\n', rmse_voltage);
fprintf('MAE = %.4f V\n', mae_voltage);
fprintf('MAPE = %.2f%%\n', mape_voltage);
fprintf('R² = %.4f\n', r2_voltage);
fprintf('误差统计：最大=%.4f V, 最小=%.4f V, 标准差=%.4f V\n\n', ...
        error_stats.MaxAbsError, error_stats.MinAbsError, error_stats.StdError);




%% 手动实现Sigma点生成函数
function [X, Wm, Wc] = manualSigmaPoints(x, P, lambda, alpha, beta)
    n = length(x);
    scaling_factor = sqrt(n + lambda);

    % Cholesky分解
    try
        S = chol(P, 'lower');
    catch
        [V,D] = eig(P);
        S = V*sqrt(D);
    end

    % 生成Sigma点
    X = zeros(n, 2*n+1);
    X(:,1) = x;
    for i = 1:n
        X(:,i+1) = x + scaling_factor*S(:,i);
        X(:,i+1+n) = x - scaling_factor*S(:,i);
    end

    % 计算权重
    Wm = zeros(1, 2*n+1);
    Wc = zeros(1, 2*n+1);
    Wm(1) = lambda/(n + lambda);
    Wc(1) = Wm(1) + (1 - alpha^2 + beta);

    for i = 2:2*n+1
        Wm(i) = 1/(2*(n + lambda));
        Wc(i) = 1/(2*(n + lambda));
    end
end



function out = IV_model(T,x,i,a,b,inner)
%I-V电压模型
%% 参数设定
R=8.314;%气体常数J/(mol*K)
F=9.64853399*10^4;%法拉第常数 C*mol=A*s*mol
% v=0.5;%反应系数
% Ef=67000;%单位J/mol
% B=0.05 %单位V 但是不清楚是否拟合的出来
%% 计算损失
T = T + 273.15;

%计算活化电压损失
e_act = (R*T)/F*asinh((x(2)+i)/(2*x(1)));% x(2)为漏电流    x(1)为交换电流密度
%计算欧姆损失
e_om = i * x(3) * 406;  %x(3)为 欧姆电阻
%计算传质损失
% e_con = x(6) * log(x(7)/(x(7) - i - x(4) ));    %x(6)为经验系数  x(7)为极限电流  
% e_con = inner(1) * log(inner(2)/(inner(2) - i - x(2) ));    %x(4)为经验系数  x(5)为极限电流  
e_con = inner(1) * log(inner(2)/(inner(2) - i ));    %x(4)为经验系数  x(5)为极限电流  
%% 计算电压
%能斯特电压方程
E=1.229-(0.85*(10^-3))*(T-298.15)+(4.3085*(10^-5))*T*(log(a)+0.5*log(b));
Eout=E-e_act-e_om-e_con;
out=Eout;
% out=[Eout,e_om];
% out=[Eout,ih,iorr,e,lefta, leftca];
end



%% origin
t = 1:0.5:(length(jh_estimates)/2);
t = t';
% j0_estimates = j0_estimates';
% j0_estimates = j0_estimates * 1e+8;
% jh_estimates = jh_estimates';
% jh_estimates = jh_estimates * 1e+3;
% R_estimates = R_estimates';
% 
% R_estimates = R_estimates * 1e+6;
% t = t';
v_test = v_test';


