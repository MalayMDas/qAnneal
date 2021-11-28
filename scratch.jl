1+1

using Revise
ga = [[1 1.0im];[1.0im 1]]
gb = Complex{Float32}.(ga)
gc = Complex{BigFloat}.(ga)

using LinearAlgebra
using GenericLinearAlgebra
hgc = Hermitian(gc)
egc = eigen(hgc)
expgc = exp(hgc)
logexpgc = log(expgc)

using Yao
@const_gate gga = ga
@const_gate ggb::Complex{Float32} = gb
@const_gate ggc::Complex{BigFloat} = gc

Matrix(gga)
@enter Matrix(ggb)

gd = [[1 1.0im];[1.0im 1/sqrt(BigFloat(2))]]
@const_gate ggd::Complex{BigFloat} = gd
Matrix(ggd)[4]

Matrix(H)

qcir = put(3, 2=>ggd)

using YaoPlots
plot(qcir)
qm = Matrix(qcir)



@const_gate Hbig::Complex{BigFloat} = 1/sqrt(BigFloat(2))*[[1 1];[1 -1]]
A(i, j) = control(i, j=>shift(2*BigFloat(π)/(1<<(i-j+1))))
B(n, k) = chain(n, j==k ? put(k=>Hbig) : A(j, k) for j in k:n)
qft(n) = chain(B(n, k) for k in 1:n)

B1(t, n, s, k) = chain(t, j==k ? put(k+s-1=>Hbig) : A(j+s-1, k+s-1) for j in k:n)
qft1(t,n,s) = chain(B1(t, n, s, k) for k in 1:n)       #t=totalqubits, n=nqubit fourier,s=starting qubit
plot(qft1(4,3,2))

plot(chain(5,repeat(H,[1 2 3])))

UT = chain(5,repeat(Hbig,[1 2 3 4 5]),qft1(5,3,1),qft1(5,2,4), swap(1,5),repeat(X,[1 2 3 4 5]),control(3, 4=>X))
UP = chain(3,repeat(Hbig,[1 2 3]),qft1(3,3,1),repeat(X,[1 2 3 ]))
UB = chain(2,repeat(Hbig,[1 2 ]),qft1(2,2,1),repeat(X,[1 2 ]))
 
plot(UT)
plot(UP)
plot(UB)

# =======================================================================

function my(x,T::DataType=Float64)
    return T(x)
end 

struct Point{T}
    x::T
    y::T
end

x=rand(1000,1000);
fill!(x, zero(x[1])); # best way to populate data into array.


# =============================================================================
include("qAnneal.jl")
qAnneal.getConfig("config_4_12_decoh")

setprecision(2048)
n=5;
Ht = qAnneal.HamiltonianBig(n, 2^n, 0);
Hh = Hermitian(Ht);
eig = eigen(Hh);
v = eig.vectors;
psi = qAnneal.EqStateBig(n,v);
psiB = qAnneal.cannonical_state_diag(Ht,psi,0.01)

ns=5
Hs = qAnneal.HamiltonianBig(ns, 2^ns, 0);
Hhs = Hermitian(Hs);
eig = eigen(Hhs);
vs = eig.vectors;
d = qAnneal.densityMatrix(psiB, [5,0], vs);
qAnneal.thermalization(d, eig.values, [5,0])

# .* can be used for outer product-**
setprecision(16384)

T1,s1,del1,b1,psi1 = qAnneal.annealTherm_constdiag(6,2.0,0.1,[6, 0], 0.0, 0,psiB,H,Hs);

eop = exp(Hermitian(-im*Hh))
psiC = eop * psiB;
d = qAnneal.densityMatrix(psiC, [6,0], vs);
qAnneal.thermalization(d, eig.values, [6,0])

# ===========
qAnneal.getConfig("config_diag")

waveFun = ones(Complex{BigFloat},2^n)
psi = qAnneal.normWaveFun(waveFun)
Float64.(abs.(exp(Hermitian(-H))*psi))

T = 0.01   # temperature
Float64.(abs.(exp(Hermitian(-H/2T))*psi))    # Canonical thermal state for view only
psiB = exp(Hermitian(-H/2T))*psi;   # canonical thermal state for calculation

d = qAnneal.densityMatrix(psiB, [6,0], v);
qAnneal.thermalization(d, eig.values, [6,0])

psi1B = exp(Hermitian(Matrix(-Hd/(2*T))))*psi1;
dd = qAnneal.densityMatrix(psi1B, [6,0], I)


psi1f = Diagonal(exp.(-im*dt*eig.values))*psi1B;
psi1f = qAnneal.normWaveFun(psi1f);
psif = v*Diagonal(exp.(-im*dt*eig.values))*v'*psiB;
psif = qAnneal.normWaveFun(psif);

dd = qAnneal.densityMatrix(psi1f, [6,0], I);
qAnneal.thermalization(dd, eig.values, [6,0])
d = qAnneal.densityMatrix(psif, [6,0], v);
qAnneal.thermalization(d, eig.values, [6,0])


T2,s2,del2,b2,psi2f = qAnneal.annealTherm_constdiag(6,2.0,1.0,[6, 0], 0.0, 0,psi1B,Matrix(Hd),Matrix(Hd));
T1,s1,del1,b1,psif = qAnneal.annealTherm_constdiag(6,2.0,1.0,[6, 0], 0.0, 0,psiB,H,H);

Float64.(abs.(psif - v*psi1f));


====================================
psi1f = Diagonal(exp.(-im*dt*eig.values))*psi1B;


# ===================================
using Nemo

ComplexField = AcbField
CC = ComplexField(256)

Harb = CC.(H) #does not work and is non-intuitive


# ========================================

qAnneal.qDisplay_unsorted(Float64.(abs.(psi)),6,10)
p = ArrayReg(bit"00")
q = zero_state(2)
qAnneal.qDisplay_unsorted(state(p),2,4)
qAnneal.qDisplay_unsorted(state(q),2,4)

qAnneal.qDisplay_unsorted(state(ArrayReg(bit"01")),2,4)
qAnneal.qDisplay_unsorted(state(ArrayReg(bit"001")),3,8)


[H[i,i] for i=1:32]

# ====
using GenericSVD        # functionality moved to GenericLinearAlgebra.jl
LinearAlgebra.eigvals(Ut)

using GenericSchur # needed for generic eigenvalue
# ====
logUt64 = log(Ut64)
eig = eigen(Ut64)
logUt64d = eig.vectors * Diagonal(log.(eig.values)) * eig.vectors'
diff = logUt64d - logUt64
sum(Float64.(abs.(diff)))

rut64 = exp(logUt64)
rut64d = exp(logUt64d)

diff = rut64 - Ut64
diff = rut64d - Ut64


eig = eigen(Ut)
logUtd = eig.vectors * Diagonal(log.(eig.values)) * eig.vectors'



eiglog = eigen(logUtd)
vtot = eiglog.vectors
etot = eiglog.values
rUt = vtot*Diagonal(exp.(etot))*vtot'

diff = rUt - Ut

# ======================
p = ArrayReg(bit"00000")
ps = state(p)
UCoupling = chain(5,put(1=>X),control(1, 3=>X),swap(1,5),control(3, 4=>X))
Uc = Matrix(UCoupling)

pf = p |> UCoupling
pfs = state(pf)
pfm = Uc*ps

diff = pfs - pfm
sum(abs.(diff))

qAnneal.qDisplay_unsorted(pfs,5,32)
qAnneal.qDisplay_unsorted(pfm,5,32)
qAnneal.qDisplay_unsorted(ps,5,32) # ps gets updated anytime we run the circuit. So, ps = pf all the time.

p = [1 0;0 1]
q = [0 5;5 0]
kron(p,q)

ds = qAnneal.densityMatrix(ps, [5,0], I);

for p = 1:0
    # print(i,' ',j,' ',p,"\n")
    println(p)
end


sys_dim = 4
env_dim = 2
for i = 1:sys_dim
    for j = 1:sys_dim
      for p = 1:env_dim
        # print(i,' ',j,' ',p,"\n")
        # ρ[i,j] =  ρ[i,j] + conj(psi[(p-1)*sys_dim+i])*psi[(p-1)*sys_dim+j]
        println(i,' ',j," : ",(p-1)*sys_dim+i,' ',(p-1)*sys_dim+j)
      end
    end
  end

Uncbp = kron(Ub,Up);
diff = Unc - Uncbp;
Float64.(abs.(diff))
sum(abs.(diff))

Hbp = kron(Hb,Hp);
diff = Hnc - Hbp;


diff = exp(Hnc) - kron(exp(Hb),exp(Hp))

diff = exp(-im*Hnc) - kron(exp(-im*Hb),exp(-im*Hp))

# ================
# To test that log in Julia returns resuts from -pi to pi
exp(pi*im)
log(exp(1.5*BigFloat(pi)*im))
log(exp(0))


exp(0)
exp(4*pi*im)

exp(4*pi*im)

-0.5*BigFloat(pi)



# ===============

m = @which exp([[1 2];[3 4]])
Base.delete_method(m)

# =============
import .qmath

diff = Up - exp(-im*Hp)
diff = Up - eigHp.vectors*Diagonal(exp.(-im*eigHp.values))*eigHp.vectors'

diff = Up - exp(log(Up))

# ================

diff = Unc - kron(Ub,Up)

diff = im*log(Unc) - kron(im*log(Ub),im*log(Up))

# ========================
# From Annealing

qAnneal.getConfig("config_4_12_decoh")
Ht = qAnneal.HamiltonianBig(5, 2^5, 0);
Hp = qAnneal.HamiltonianBig(3, 2^3, 0);
qAnneal.getConfig("config_4_12_decoh",4)
Hb = qAnneal.HamiltonianBig(2, 2^2, 0);

diff = Ht - kron(Hb,Hp)