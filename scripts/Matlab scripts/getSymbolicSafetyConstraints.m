function [hp, hm, hd] = getSymbolicSafetyConstraints()
q = sym('q', [3,1], 'real');
l = sym('l', [3,1], 'real');
L = sym('L', [3,1], 'real');

q1 = q(1); q2 = q(2); q3 = q(3); 
l1 = l(1); l2 = l(2); l3 = l(3); 
L1 = L(1); L2 = L(2); L3 = L(3); 

lx = sym('lx', [3,1], 'real');
ly = sym('ly', [3,1], 'real');

syms z 'real'

lx1 = lx(1); lx2 = lx(2); lx3 = lx(3); 
ly1 = ly(1); ly2 = ly(2); ly3 = ly(3); 

%%
pp = [-ly1 * sin(q1) + lx1 * cos(q1); 
       ly1 * cos(q1) + lx1 * sin(q1);
       z];
   
pm = [-L1 * sin(q1) - ly2 * sin(q1 + q2) + lx2 * cos(q1 + q2); 
       L1 * cos(q1) + ly2 * cos(q1 + q2) + lx2 * sin(q1 + q2);
       z];
   
pd = [-L1 * sin(q1) - L2 * sin(q1 + q2) - ly3 * sin(q1 + q2 + q3) + lx3 * cos(q1 + q2 + q3); 
       L1 * cos(q1) + L2 * cos(q1 + q2) + ly3 * cos(q1 + q2 + q3) + lx3 * sin(q1 + q2 + q3);
       z];
   
%% now we need to calculate the safety constraints
syms nx ny 'real' % direction of the contact normal projected on the xy plane
syms pcx pcy 'real' % the location of the contact point in the x y plane 
n = [nx; ny; 0];
pcontact = [pcx; pcy; z];
hp = dot(n, (pp - pcontact));
hm = dot(n, (pm - pcontact));
hd = dot(n, (pd - pcontact));

end

