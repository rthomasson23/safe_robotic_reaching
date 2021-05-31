%% clean up workspace 
clear; clc;
%% define symbolic variables
q = sym('q', [6,1], 'real'); 
u = sym('u', [3,1], 'real');

q1 = q(1); q2 = q(2); q3 = q(3); q4 = q(4); q5 = q(5); q6 = q(6); 
u1 = u(1); u2 = u(2); u3 = u(3);

% q = [th1 th2 th3 th1_dot th2_dot th3_dot]'
% u = [tau1 tau2 tau3]'

l = sym('l', [3,1], 'real');
L = sym('L', [3,1], 'real');

l1 = l(1); l2 = l(2); l3 = l(3); 
L1 = L(1); L2 = L(2); L3 = L(3); 

m = sym('m', [3,1], 'real');
Ixx = sym('Ixx', [3,1], 'real');
Iyy = sym('Iyy', [3,1], 'real');
Izz = sym('Izz', [3,1], 'real');

m1 = m(1); m2 = m(2); m3 = m(3); 
Ixx1 = Ixx(1); Ixx2 = Ixx(2); Ixx3 = Ixx(3); 
Iyy1 = Iyy(1); Iyy2 = Iyy(2); Iyy3 = Iyy(3); 
Izz1 = Izz(1); Izz2 = Izz(2); Izz3 = Izz(3); 

%% define M, C, B, and G
% elements of M
[M, C, B] = getSymbolicMatrices();
% v = C * [q4^2; q5^2; q6^2] + B * [q4*q5; q4*q6; q5*q6];

%% define dynamics as qdot = f(q) + g(q)*u
fq = [q(4);
      q(5);
      q(6);
      0;
      0;
      0];
  
% fq(4:end) = M\v;
gqu(4:6,1) = M\u;
gq(4:6,:) = inv(M);

%% define safety constraint(s)
syms obs_x obs_y r d 'real' % location of center of obstacle
[hp, hm, hd] = getSymbolicNCConstraints();
%%
% distal ----------------------------------------------------------------
dhd_dq = [simplify(diff(hd, q(1)));
          simplify(diff(hd, q(2)));
          simplify(diff(hd, q(3)));
          simplify(diff(hd, q(4)));
          simplify(diff(hd, q(5)));
          simplify(diff(hd, q(6)))];

Lfhd = simplify(dot(dhd_dq, fq));
% Lghu = simplify(dot(dh_dq, gqu)); 
Lghd = simplify(dhd_dq'*gq);
Lghd_u = Lghd*u;
% as a sanity check, we see Lghu is 0 because this is an rd2 system

hd_dot = Lfhd + Lghd_u;

% medial ----------------------------------------------------------------
dhm_dq = [simplify(diff(hm, q(1)));
                 simplify(diff(hm, q(2)));
                 simplify(diff(hm, q(3)));
                 simplify(diff(hm, q(4)));
                 simplify(diff(hm, q(5)));
                 simplify(diff(hm, q(6)))];

Lfhm = simplify(dot(dhm_dq, fq));
% Lghu = simplify(dot(dh_dq, gqu)); 
Lghm = simplify(dhm_dq'*gq);
Lghm_u = Lghm*u;
% as a sanity check, we see Lghu is 0 because this is an rd2 system

hm_dot = Lfhm + Lghm_u;


% proximal  ----------------------------------------------------------------
dhp_dq = [simplify(diff(hp, q(1)));
                 simplify(diff(hp, q(2)));
                 simplify(diff(hp, q(3)));
                 simplify(diff(hp, q(4)));
                 simplify(diff(hp, q(5)));
                 simplify(diff(hp, q(6)))];

Lfhp = simplify(dot(dhp_dq, fq));
% Lghu = simplify(dot(dh_dq, gqu)); 
Lghp = simplify(dhp_dq'*gq);
Lghp_u = Lghp*u;
% as a sanity check, we see Lghu is 0 because this is an rd2 system

hp_dot = Lfhp + Lghp_u;

% note, were gonna make alpha1(h(q)) = h(q)

%% define B (CBF for rd2)
% distal -----------------------------------------------------------------
B1_distal = hd_dot + hd;

dB1_distal_dq = [simplify(diff(B1_distal, q(1)));
                 simplify(diff(B1_distal, q(2)));
                 simplify(diff(B1_distal, q(3)));
                 simplify(diff(B1_distal, q(4)));
                 simplify(diff(B1_distal, q(5)));
                 simplify(diff(B1_distal, q(6)))];
     
LfB1_distal = simplify(dot(dB1_distal_dq, fq));
LgB1_distal = simplify(dB1_distal_dq'*gq);

% medial -----------------------------------------------------------------
B1_medial = hm_dot + hm;

dB1_medial_dq = [simplify(diff(B1_medial, q(1)));
                 simplify(diff(B1_medial, q(2)));
                 simplify(diff(B1_medial, q(3)));
                 simplify(diff(B1_medial, q(4)));
                 simplify(diff(B1_medial, q(5)));
                 simplify(diff(B1_medial, q(6)))];
     
LfB1_medial = simplify(dot(dB1_medial_dq, fq));
LgB1_medial = simplify(dB1_medial_dq'*gq);

% proximal -----------------------------------------------------------------
B1_proximal = hp_dot + hp;

dB1_proximal_dq = [simplify(diff(B1_proximal, q(1)));
                   simplify(diff(B1_proximal, q(2)));
                   simplify(diff(B1_proximal, q(3)));
                   simplify(diff(B1_proximal, q(4)));
                   simplify(diff(B1_proximal, q(5)));
                   simplify(diff(B1_proximal, q(6)))];
     
LfB1_proximal = simplify(dot(dB1_proximal_dq, fq));
LgB1_proximal = simplify(dB1_proximal_dq'*gq);
% and as a sanity check, this should be nonzero for our rd2 system

% also, we'll define alpha2(B(q)) = B(q)

% our safety constraint is then:
% LfB + LgB*u + B(q) >= 0

