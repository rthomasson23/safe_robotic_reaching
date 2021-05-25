function [M, C, B] = getSymbolicMatrices()
%% geometric parameters 
q = sym('q', [3,1], 'real');
l = sym('l', [3,1], 'real');
L = sym('L', [3,1], 'real');

q1 = q(1); q2 = q(2); q3 = q(3); 
l1 = l(1); l2 = l(2); l3 = l(3); 
L1 = L(1); L2 = L(2); L3 = L(3); 

syms z 'real'

%% get symbolic jacobians
pc1 = [-l1 * sin(q1); 
       l1 * cos(q1);
       z];
   
pc2 = [-L1 * sin(q1) - l2 * sin(q1 + q2); 
       L1 * cos(q1) + l2 * cos(q1 + q2);
       z];
   
pc3 = [-L1 * sin(q1) - L2 * sin(q1 + q2) - l3 * sin(q1 + q2 + q3); 
       L1 * cos(q1) + L2 * cos(q1 + q2) + l3 * cos(q1 + q2 + q3);
       z];
   
Jv1 = [simplify(diff(pc1, q1)), simplify(diff(pc1, q2)), simplify(diff(pc1, q3))];
Jv2 = [simplify(diff(pc2, q1)), simplify(diff(pc2, q2)), simplify(diff(pc2, q3))];
Jv3 = [simplify(diff(pc3, q1)), simplify(diff(pc3, q2)), simplify(diff(pc3, q3))];

Jw1 = [0, 0, 0;
       0, 0, 0;
       1, 0, 0];
Jw2 = [0, 0, 0;
       0, 0, 0;
       1, 1, 0];
Jw3 = [0, 0, 0;
       0, 0, 0;
       1, 1, 1];

%% mass and inertia parameters
m = sym('m', [3,1], 'real');
Ixx = sym('Ixx', [3,1], 'real');
Iyy = sym('Iyy', [3,1], 'real');
Izz = sym('Izz', [3,1], 'real');

m1 = m(1); m2 = m(2); m3 = m(3); 
Ixx1 = Ixx(1); Ixx2 = Ixx(2); Ixx3 = Ixx(3); 
Iyy1 = Iyy(1); Iyy2 = Iyy(2); Iyy3 = Iyy(3); 
Izz1 = Izz(1); Izz2 = Izz(2); Izz3 = Izz(3); 

I1 = [Ixx1, 0, 0;
      0, Iyy1, 0;
      0, 0, Izz1];

I2 = [Ixx2, 0, 0;
      0, Iyy2, 0;
      0, 0, Izz2];
  
I3 = [Ixx3, 0, 0;
      0, Iyy3, 0;
      0, 0, Izz3];
%% get mass matrix symbolically
M = (m1 * Jv1') * Jv1 + (Jw1' * I1) * Jw1;
M = M + (m2 * Jv2') * Jv2 + (Jw2' * I2) * Jw2;
M = M + (m3 * Jv3') * Jv3 + (Jw3' * I3) * Jw3;

%% get coriolis and centrifugal matrices

for i = [1,2,3]
    for j = [1,2,3]
        for k = [1,2,3]
            mijk = simplify(diff(M(i, j), q(k)));
            mikj = simplify(diff(M(i, k), q(j)));
            mjki = simplify(diff(M(j, k), q(i)));
            
            bijk = 0.5 * (mijk + mikj + mjki);
            
            if j == k
                C(i,j) = bijk;
            end
            if (j == 1) && (k == 2)
                B(i, 1) = 2 * bijk;
            elseif (j == 1) && (k == 3)
                B(i, 2) = 2 * bijk;
            elseif (j == 2) && (k == 3)
                B(i, 3) = 2 * bijk;
            end
        end
    end
end
M = simplify(M);
C = simplify(C);
B = simplify(B);
            
end

