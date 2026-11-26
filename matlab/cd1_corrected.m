%% Modelado Cinemático y Dinámico Corregido — Manipulador Serial 3 GDL
% Versión corregida de cd1.m / cd1_live.mlx.
%
% Correcciones aplicadas respecto a cd1.m / cd1_live.mlx:
%   1. Propagación de velocidades (Craig): se incluye la rotación R_{i}^{i-1}.'
%      sobre el término cruzado: v_i = R.' * (v_{i-1} + cross(w_{i-1}, P_{i-1,i})).
%   2. Vectores de centro de masa locales correctos:
%      - Eslabón 1 (marco {1}): r_c1 = [0; 0; l_c1 - l_1] (a altura l_c1 desde {0})
%      - Eslabón 2 (marco {2}): r_c2 = [l_c2; 0; 0]       (antes P12c = [0;0;0])
%      - Eslabón 3 (marco {3}): r_c3 = [l_c3; 0; 0]       (antes P23c = [l_2;0;0])
%   3. Energía potencial y Lagrangiano:
%      U_i = +m_i * g * H_i,  U = U1 + U2 + U3,  L = K - U.
%   4. Matriz de Coriolis C(q, qp) construida mediante símbolos de Christoffel,
%      garantizando que (Mdot - 2*C) sea antisimétrica.
%   5. Funciones locales transl/troty/trotz incluidas (no requiere Toolbox externo).

clc;
clear variables;

%% 1. Definición de Variables Simbólicas
syms q1 q2 q3 real
syms q1p q2p q3p real
syms q1pp q2pp q3pp real
syms l1 l2 l3 lc1 lc2 lc3 g real
syms m1 m2 m3 real
syms Ixx1 Iyy1 Izz1 Ixx2 Iyy2 Izz2 Ixx3 Iyy3 Izz3 real

q   = [q1;  q2;  q3];
qp  = [q1p; q2p; q3p];
qpp = [q1pp; q2pp; q3pp];

%% 2. Cinemática Directa (Convención D-H de cd1.m / cd1_live.mlx)
% En q = [0; 0; 0], el manipulador apunta verticalmente hacia +Z:
%   {0} -> {1}: traslación l1 en +Z y giro q1 alrededor de Z (yaw)
%   {1} -> {2}: rotación fija Ry(pi/2)*Rz(pi) y giro q2 alrededor de Z2 (pitch)
%   {2} -> {3}: traslación l2 en X2 y giro q3 alrededor de Z3 (pitch)
%   {3} -> {4}: traslación l3 en X3 hasta el efector final
S01 = transl_local(0, 0, l1) * trotz_local(q1);
S12 = troty_local(pi/2) * trotz_local(q2 + pi);
S23 = transl_local(l2, 0, 0) * trotz_local(q3);
S34 = transl_local(l3, 0, 0);

S02 = simplify(S01 * S12);
S03 = simplify(S02 * S23);
S04 = simplify(S03 * S34);

p_ee = simplify(S04(1:3, 4));
disp('=== Posición del Efector Final p_ee(q) ===');
pretty(p_ee);

% Jacobiano lineal del efector final: J_v = dp_ee / dq
J_v = simplify(jacobian(p_ee, q));
disp('=== Jacobiano Lineal J_v(q) ===');
pretty(J_v);

%% 3. Propagación de Velocidades (Método de Craig)
R01 = S01(1:3, 1:3);  P01 = S01(1:3, 4);
R12 = S12(1:3, 1:3);  P12 = S12(1:3, 4);
R23 = S23(1:3, 1:3);  P23 = S23(1:3, 4);
R34 = S34(1:3, 1:3);  P34 = S34(1:3, 4);

w00 = sym([0; 0; 0]);
v00 = sym([0; 0; 0]);
z   = sym([0; 0; 1]);

% Velocidades angulares y lineales expresadas en el marco {i}
w11 = simplify(R01.' * w00 + q1p * z);
v11 = simplify(R01.' * (v00 + cross(w00, P01)));

w22 = simplify(R12.' * w11 + q2p * z);
v22 = simplify(R12.' * (v11 + cross(w11, P12)));

w33 = simplify(R23.' * w22 + q3p * z);
v33 = simplify(R23.' * (v22 + cross(w22, P23)));

w44 = simplify(R34.' * w33);
v44 = simplify(R34.' * (v33 + cross(w33, P34)));

% Vectores locales desde el origen del marco {i} al CoM del eslabón i
r_c1 = [0; 0; lc1 - l1];  % CoM del eslabón 1 a altura lc1 desde la base {0}
r_c2 = [lc2; 0; 0];       % CoM del eslabón 2 a distancia lc2 sobre X2
r_c3 = [lc3; 0; 0];       % CoM del eslabón 3 a distancia lc3 sobre X3

% Velocidades lineales de los centros de masa (en cada marco local {i})
vc1 = simplify(v11 + cross(w11, r_c1));
vc2 = simplify(v22 + cross(w22, r_c2));
vc3 = simplify(v33 + cross(w33, r_c3));

%% 4. Energía Cinética y Potencial (Euler-Lagrange)
I1 = diag([Ixx1, Iyy1, Izz1]);
I2 = diag([Ixx2, Iyy2, Izz2]);
I3 = diag([Ixx3, Iyy3, Izz3]);

K1 = simplify((1/2) * m1 * (vc1.' * vc1) + (1/2) * (w11.' * I1 * w11));
K2 = simplify((1/2) * m2 * (vc2.' * vc2) + (1/2) * (w22.' * I2 * w22));
K3 = simplify((1/2) * m3 * (vc3.' * vc3) + (1/2) * (w33.' * I3 * w33));
K  = simplify(K1 + K2 + K3);

% Posiciones de los CoM en el marco inercial {0}
pc1_0 = S01 * [r_c1; 1];
pc2_0 = S02 * [r_c2; 1];
pc3_0 = S03 * [r_c3; 1];

H1 = simplify(pc1_0(3));
H2 = simplify(pc2_0(3));
H3 = simplify(pc3_0(3));

% Energía potencial gravitacional (U = +m*g*h con +Z hacia arriba)
U1 = m1 * g * H1;
U2 = m2 * g * H2;
U3 = m3 * g * H3;
U  = simplify(U1 + U2 + U3);

% Lagrangiano L = K - U
L = simplify(K - U);

%% 5. Ecuaciones de Euler-Lagrange: M(q)*qpp + C(q,qp)*qp + G(q) = tau
tau = sym(zeros(3, 1));
for i = 1:3
    dL_dqpi = diff(L, qp(i));
    ddt_dL_dqpi = jacobian(dL_dqpi, q) * qp + jacobian(dL_dqpi, qp) * qpp;
    tau(i) = simplify(ddt_dL_dqpi - diff(L, q(i)));
end

% Matriz de inercia M(q)
M = simplify(jacobian(tau, qpp));

% Vector de gravedad G(q) = dU/dq
G = simplify(jacobian(U, q).');

% Matriz de Coriolis y centrífuga C(q, qp) por símbolos de Christoffel
C_mat = sym(zeros(3, 3));
for i = 1:3
    for j = 1:3
        cij = sym(0);
        for k = 1:3
            c_ijk = (1/2) * (diff(M(i,j), q(k)) + diff(M(i,k), q(j)) - diff(M(j,k), q(i)));
            cij = cij + c_ijk * qp(k);
        end
        C_mat(i, j) = simplify(cij);
    end
end
C_vec = simplify(C_mat * qp);

% Verificación simbólica: M*qpp + C*qp + G == tau
res_el = simplify(M * qpp + C_vec + G - tau);
assert(isequal(res_el, sym(zeros(3, 1))), 'Error en descomposición M, C, G');

% Verificación de propiedad fundamental: (Mdot - 2*C) es antisimétrica
Mdot = sym(zeros(3, 3));
for i = 1:3
    for j = 1:3
        Mdot(i, j) = jacobian(M(i, j), q) * qp;
    end
end
skew_check = simplify((Mdot - 2 * C_mat) + (Mdot - 2 * C_mat).');
assert(isequal(skew_check, sym(zeros(3, 3))), 'Mdot - 2*C no es antisimétrica');

disp('=== Matriz de Inercia M(q) ===');
pretty(M);
disp('=== Vector de Gravedad G(q) ===');
pretty(G);
disp('=== Vector de Coriolis y Centrífugo C(q, qp)*qp ===');
pretty(C_vec);

%% 6. Evaluación Numérica y Comparación con MuJoCo (si existe mujoco_comparison.mat)
% Parámetros del modelo CAD (Ensamble_SWML.xml / MuJoCo)
params_cad = struct( ...
    'l1', 0.07895, 'l2', 0.23700, 'l3', 0.18000, ...
    'lc1', 0.06707, 'lc2', 0.12050, 'lc3', 0.04539, ...
    'g', 9.81, ...
    'm1', 0.00925, 'm2', 0.09413, 'm3', 0.01179, ...
    'Ixx1', 1.65e-6,  'Iyy1', 3.65e-6,    'Izz1', 4.33e-6, ...
    'Ixx2', 14.56e-6, 'Iyy2', 1010.12e-6, 'Izz2', 1014.22e-6, ...
    'Ixx3', 3.21e-6,  'Iyy3', 39.84e-6,   'Izz3', 42.02e-6);

p_ee_num = subs(p_ee, fieldnames(params_cad), struct2cell(params_cad));
tau_num  = subs(tau,  fieldnames(params_cad), struct2cell(params_cad));

fk_fun  = matlabFunction(p_ee_num, 'Vars', {q1, q2, q3});
tau_fun = matlabFunction(tau_num,  'Vars', {q1, q2, q3, q1p, q2p, q3p, q1pp, q2pp, q3pp});

mat_file = fullfile(fileparts(mfilename('fullpath')), '..', 'mujoco_comparison.mat');
if isfile(mat_file)
    S = load(mat_file);
    N = length(S.t);
    ee_matlab  = zeros(N, 3);
    tau_matlab = zeros(N, 3);
    for k = 1:N
        p_dh = fk_fun(S.q(k,1), S.q(k,2), S.q(k,3));
        % Rotación fija de +90 deg en Z para alinear marco DH con marco MuJoCo
        ee_matlab(k, :) = [-p_dh(2), p_dh(1), p_dh(3)];
        tau_matlab(k, :) = tau_fun( ...
            S.q(k,1), S.q(k,2), S.q(k,3), ...
            S.qd(k,1), S.qd(k,2), S.qd(k,3), ...
            S.qdd(k,1), S.qdd(k,2), S.qdd(k,3)).';
    end
    fk_rmse_mm = sqrt(mean((ee_matlab - S.ee_mujoco).^2, 1)) * 1000;
    tau_rmse   = sqrt(mean((tau_matlab - S.tau_mujoco).^2, 1));
    fprintf('\n=== Comparación MATLAB Corregido vs MuJoCo ===\n');
    fprintf('  FK RMSE [mm]:    [%.4f, %.4f, %.4f]\n', fk_rmse_mm);
    fprintf('  Torque RMSE [Nm]: [%.6f, %.6f, %.6f]\n', tau_rmse);
end

%% Funciones Auxiliares Locales (equivalentes a Robotics Toolbox)
function T = transl_local(x, y, z)
    T = [sym(1), 0, 0, x;
         0, sym(1), 0, y;
         0, 0, sym(1), z;
         0, 0, 0, sym(1)];
end

function T = trotz_local(theta)
    c = cos(theta); s = sin(theta);
    T = [c, -s, 0, 0;
         s,  c, 0, 0;
         0,  0, 1, 0;
         0,  0, 0, 1];
end

function T = troty_local(theta)
    c = cos(theta); s = sin(theta);
    T = [ c, 0, s, 0;
          0, 1, 0, 0;
         -s, 0, c, 0;
          0, 0, 0, 1];
end
