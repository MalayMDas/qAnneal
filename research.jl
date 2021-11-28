setprecision(2048)
setprecision(128)
setprecision(16384)
setprecision(65536)

using Revise
using MKL
using LinearAlgebra
BLAS.set_num_threads(8)
using GenericLinearAlgebra
#using GenericSchur
using ComplexBigMatrices
using GenericSchur
include("qAnneal.jl")
using Yao
using YaoPlots
using Plots
using Random
Random.seed!(1234);
using DelimitedFiles
using Base.Threads
using Statistics


@const_gate Hbig::Complex{BigFloat} = 1/sqrt(BigFloat(2))*[[1 1];[1 -1]];
A(i, j) = control(i, j=>shift(2*BigFloat(π)/(1<<(i-j+1))));
B1(t, n, s, k) = chain(t, j==k ? put(k+s-1=>Hbig) : A(j+s-1, k+s-1) for j in k:n);
qft1(t,n,s) = chain(B1(t, n, s, k) for k in 1:n);       #t=totalqubits, n=nqubit fourier,s=starting qubit
YaoPlots.plot(qft1(4,3,2))

UTotal = chain(5,repeat(Hbig,[1 2 3 4 5]),qft1(5,3,1),qft1(5,2,4),repeat(X,[1 2 3 4 5]), swap(1,5),control(3, 4=>X));
UProblem = chain(3,repeat(Hbig,[1 2 3]),qft1(3,3,1),repeat(X,[1 2 3 ]));
UBath = chain(2,repeat(Hbig,[1 2 ]),qft1(2,2,1),repeat(X,[1 2 ]));

UTnoCoupling = chain(5,repeat(Hbig,[1 2 3 4 5]),qft1(5,3,1),qft1(5,2,4),repeat(X,[1 2 3 4 5]));
UCoupling = chain(5,swap(1,5),control(3, 4=>X));

YaoPlots.plot(UTotal)
YaoPlots.plot(UProblem)
YaoPlots.plot(UBath)
YaoPlots.plot(UTnoCoupling)
YaoPlots.plot(UCoupling)



Ut = Matrix(UTotal);
eigt = eigen(Ut);
Ht = Hermitian(im * eigt.vectors * Diagonal(log.(eigt.values)) * eigt.vectors');
eigHt = eigen(Ht);

Up = Matrix(UProblem);
eigp = eigen(Up);
Hp = Hermitian(im * eigp.vectors * Diagonal(log.(eigp.values)) * eigp.vectors');
eigHp = eigen(Hp);

Uc = Matrix(UCoupling);
eigc = eigen(Uc);
Hc = Hermitian(im * eigc.vectors * Diagonal(log.(eigc.values)) * eigc.vectors');
eigHc = eigen(Hc);

Unc = Matrix(UTnoCoupling);
eignc = eigen(Unc);
Hnc = Hermitian(im * eignc.vectors * Diagonal(log.(eignc.values)) * eignc.vectors');
eigHnc = eigen(Hnc);

# Scenario 1 - No Bath
n = 5
T = 0.01
psi = qAnneal.EqStateBig(n,eigHt.vectors);

psiB = qAnneal.cannonical_state_diag(Ht,psi,T)
d = qAnneal.densityMatrix(psiB, [5,0], eigHt.vectors)
qAnneal.thermalization(d, eigHt.values, [5,0])

d = qAnneal.densityMatrixThreaded(psiB, [5,0], eigHt.vectors)

T1,s1,del1,b1,psinb = qAnneal.annealTherm_constdiag(5,50.0,1.0,[5, 0], 0.0, 0,psiB,Ht,Ht,Ut); # T=0.01
T2,s2,del2,b2,psibc = qAnneal.annealTherm_constdiag(5,50.0,1.0,[3, 2], 0.0, 0,psiB,Ht,Hp,Ut);

T=0.1
psiB = qAnneal.cannonical_state_diag(Ht,psi,T)
d = qAnneal.densityMatrix(psiB, [5,0], eigHt.vectors)
T3,s3,del3,b3,psinb = qAnneal.annealTherm_constdiag(5,50.0,1.0,[5, 0], 0.0, 0,psiB,Ht,Ht,Ut);
d = qAnneal.densityMatrix(psiB, [3,2], eigHp.vectors)
T5,s5,del5,b5,psibc = qAnneal.annealTherm_constdiag(5,50.0,1.0,[3, 2], 0.0, 0,psiB,Ht,Hp,Ut);

T=0.001
psiB = qAnneal.cannonical_state_diag(Ht,psi,T)
d = qAnneal.densityMatrix(psiB, [5,0], eigHt.vectors)
T4,s4,del4,b4,psinb = qAnneal.annealTherm_constdiag(5,50.0,1.0,[5, 0], 0.0, 0,psiB,Ht,Ht,Ut);
d = qAnneal.densityMatrix(psiB, [3,2], eigHp.vectors)
T6,s6,del6,b6,psibc = qAnneal.annealTherm_constdiag(5,50.0,1.0,[3, 2], 0.0, 0,psiB,Ht,Hp,Ut);

# All uniform distribution
psiB = ones(Complex{BigFloat},2^n)
psiB = psiB/norm(psiB)
d = qAnneal.densityMatrix(psiB, [5,0], eigHt.vectors)
T7,s7,del7,b7,psinb = qAnneal.annealTherm_constdiag(5,50.0,1.0,[5, 0], 0.0, 0,psiB,Ht,Ht,Ut);
d = qAnneal.densityMatrix(psiB, [3,2], eigHp.vectors)
T8,s8,del8,b8,psibc = qAnneal.annealTherm_constdiag(5,50.0,1.0,[3, 2], 0.0, 0,psiB,Ht,Hp,Ut);


# all zero state
psiB = Complex{BigFloat}.(state(ArrayReg(bit"00000")))
d = qAnneal.densityMatrix(psiB, [5,0], eigHt.vectors)
T9,s9,del9,b9,psinb = qAnneal.annealTherm_constdiag(5,50.0,1.0,[5, 0], 0.0, 0,psiB,Ht,Ht,Ut);
d = qAnneal.densityMatrix(psiB, [3,2], eigHp.vectors)
T10,s10,del10,b10,psibc = qAnneal.annealTherm_constdiag(5,50.0,1.0,[3, 2], 0.0, 0,psiB,Ht,Hp,Ut);

Plots.plot(T1,b1,label="T=0.01")
Plots.plot!(T3,b3,label="T=0.1")
Plots.plot!(T4,b4,label="T=0.001")
Plots.plot!(T7,b7,label="Equal superposition")
Plots.plot!(T9,b9,label="|00000>")
title!("5 qubits - No Bath")
xlabel!("time")
ylabel!("b")


Plots.plot(T2,b2,label="T=0.01")
Plots.plot!(T5,b5,label="T=0.1")
Plots.plot!(T6,b6,label="T=0.001")
#Plots.plot!(T8,b8,label="Equal superposition")
Plots.plot!(T10,b10,label="|00000>")
title!("5 qubits - 2 qubit Bath")
xlabel!("time")
ylabel!("b")

Plots.plot(T1,del1,label="delta")
xlabel!("time")
ylabel!("delta")

Plots.plot(T1,s1,label="sigma")
xlabel!("time")
ylabel!("sigma")

T1,s1,del1,b1,psinb = qAnneal.annealTherm_constdiag(5,5.0,1.0,[5, 0], 0.0, 0,psiB,Ht,Ht,Ut);

# Scenario 2 - system and bath from CTSE




# Scenarion 2 test with no coupling between system and bath
Unc = Matrix(UTnoCoupling);
eignc = eigen(Unc);
Hnc = Hermitian(im * eignc.vectors * Diagonal(log.(eignc.values)) * eignc.vectors');

eigHnc = eigen(Hnc);
psinc = qAnneal.EqStateBig(n,eigHnc.vectors)
psiBnc = qAnneal.cannonical_state_diag(Hnc,psinc,T)
dnc = qAnneal.densityMatrix(psiBnc, [5,0], eigHnc.vectors)
qAnneal.thermalization(dnc, eigHnc.values, [5,0])

eigHp = eigen(Hp)
dncb = qAnneal.densityMatrix(psiBnc, [3,2], eigHp.vectors)
qAnneal.thermalization(dncb, eigHp.values, [3,2])

T2,s2,del2,b2,psibc = qAnneal.annealTherm_constdiag(5,50.0,1.0,[3, 2], 0.0, 0,psiBnc,Hnc,Hp,Unc);

T3,s3,del3,b3,psibc = qAnneal.annealTherm_constdiag(5,50.0,1.0,[3, 2], 0.0, 0,psinc,Hnc,Hp,Unc);  # for not starting at CTSE

#Checking if my order is Incorrect 
Ub = Matrix(UBath)
eigb = eigen(Ub)
Hb = Hermitian(im * eigb.vectors * Diagonal(log.(eigb.values)) * eigb.vectors')
eigHb = eigen(Hb)
dncb = qAnneal.densityMatrix(psiBnc, [2,3], eigHb.vectors)
qAnneal.thermalization(dncb, eigHp.values, [2,3])




UbUp = kron(Ub, Up);
# UpUb = kron(Up, Ub);      # not correct i.e Unc != kron(Up, Ub)
diff = Unc - UbUp
# diff = Unc - UpUb         # Not correct i.e Unc != kron(Up, Ub)
sum(abs.(diff))

HbHp = kron(Hb, Hp);
diff = Hnc - HbHp

heatmap(Float64.(real(Hta)), title="H_total real")
heatmap(Float64.(imag(Hta)), title="H_total imaginery")
heatmap(Float64.(imag(Ut)), title="U_total imaginery")
heatmap(Float64.(real(Ut)), title="U_total real")
heatmap(Float64.(real(Up)), title="U_problem real")
heatmap(Float64.(imag(Up)), title="U_problem imaginery")
heatmap(Float64.(imag(Hpa)), title="H_problem imaginery")
heatmap(Float64.(real(Hpa)), title="H_problem real")

Plots.plot(T6,b6,label="with Bath", title="b at T=0.001")
Plots.plot!(T4,b4,label="No Bath")


psinc = qAnneal.EqStateBig(5,eigHnc.vectors);
psip = qAnneal.EqStateBig(3,eigHp.vectors);
psib = qAnneal.EqStateBig(2,eigHb.vectors);

temp = 0.01
psiBnc = qAnneal.cannonical_state_diag(Hnc,psinc,Temp);
psiBp = qAnneal.cannonical_state_diag(Hp,psip,Temp);
psiBb = qAnneal.cannonical_state_diag(Hb,psib,Temp);

dnc = qAnneal.densityMatrix(psiBnc, [5,0], eigHnc.vectors);
dncs = qAnneal.densityMatrix(psiBnc, [3,2], eigHp.vectors);
dp = qAnneal.densityMatrix(psiBp, [3,0], eigHp.vectors);
db = qAnneal.densityMatrix(psiBb, [2,0], eigHb.vectors);

psiBt = kron(psiBb,psiBp);
dt = qAnneal.densityMatrix(psiBt, [5,0], eigHnc.vectors);

dtpb = qAnneal.densityMatrix(psiBt, [3,2], eigHp.vectors);

qAnneal.thermalization(dnc, eigHnc.values, [5,0])
qAnneal.thermalization(dncs, eigHp.values, [3,2])
qAnneal.thermalization(dp, eigHp.values, [3,0])
qAnneal.thermalization(db, eigHb.values, [2,0])
qAnneal.thermalization(dt, eigHnc.values, [5,0])

qAnneal.thermalization(dtpb, eigHp.values, [3,2])




Float64.(qAnneal.thermalization(dnc, eigHnc.values, [5,0]))
Float64.(qAnneal.thermalization(dncs, eigHp.values, [3,2]))
Float64.(qAnneal.thermalization(dp, eigHp.values, [3,0]))
Float64.(qAnneal.thermalization(db, eigHb.values, [2,0]))
Float64.(qAnneal.thermalization(dt, eigHnc.values, [5,0]))
Float64.(qAnneal.thermalization(dtpb, eigHp.values, [3,2]))


diff = psiBt - psiBnc;
Float64.(abs.(diff))

# ==================================================

psiBnc = qAnneal.cannonical_state_diag(Matrix(Diagonal(eigHnc.values)),psinc,Temp);
psiBp = qAnneal.cannonical_state_diag(Matrix(Diagonal(eigHp.values)),psip,Temp);
psiBb = qAnneal.cannonical_state_diag(Matrix(Diagonal(eigHb.values)),psib,Temp);
psiBt = kron(psiBb,psiBp);

dnc = qAnneal.densityMatrix(psiBnc, [5,0], I);
dncs = qAnneal.densityMatrix(psiBnc, [3,2], I);
dp = qAnneal.densityMatrix(psiBp, [3,0], I);
db = qAnneal.densityMatrix(psiBb, [2,0], I);
dt = qAnneal.densityMatrix(psiBt, [5,0], I);
dtpb = qAnneal.densityMatrix(psiBt, [3,2], I);

histogram(Float64.(abs.(psiBb)),bins=1:5)
histogram(Float64.(abs.(psiBt)),bins=1:33)
# ================================================
# Single runs
psi = qAnneal.randomStateBig(5,eigHnc.vectors);
psiB = qAnneal.cannonical_state_diag(Hnc,psi,T)
dncs = qAnneal.densityMatrix(psiB, [3,2], eigHp.vectors);   
Idel, ib = qAnneal.thermalization(dncs, eigHp.values, [3,2]); 
T1,s1,del1,b1,psibc = qAnneal.annealTherm_constdiag(5,50.0,1.0,[3, 2], 0.0, 0,psiB,Hnc,Hp,Unc);

# ===================================

qAnneal.getConfig("config_4_2_degen_sc015");
Hta = qAnneal.HamiltonianBig(6, 2^6, 0);
eigHta=eigen(Hta);
Uta = eigHta.vectors*Diagonal(exp.(-im*eigHta.values))*eigHta.vectors';
Hpa = qAnneal.HamiltonianBig(4, 2^4, 0);
eigHpa=eigen(Hpa);
T=1;
factor = exp(-Hta/(2*T));
#psi = qAnneal.randomStateBig(6,eigHta.vectors);
psi = qAnneal.EqStateBig(6,eigHta.vectors);
psiB = qAnneal.cannonical_state_fast(Hta,psi,T,factor);

d = qAnneal.densityMatrix_new(psiB, [4,2], eigHta.vectors);  
d2 = qAnneal.densityMatrix(psiB, [4,2], eigHpa.vectors);  
Idel, ib = qAnneal.thermalization(d2, eigHpa.values, [4,2])

# =========================================================
T = 10
psi = qAnneal.randomStateBig(6,I);
psi = ones(Complex{BigFloat},2^6);
psiB = qAnneal.cannonical_state_fast(Hta,psi,T,Diagonal(exp.(-eigHta.values/(2*T))));

d1 = qAnneal.densityMatrix_new(psiB, [4,2], I);   
Idel, ib = qAnneal.thermalization(d1, eigHpa.values, [4,2],1.0e-20)
log(d1[1,1]) - log(d1[3,3])
eigHpa.values[3] - eigHpa.values[1]
(log(d1[1,1]) - log(d1[3,3]))/(eigHpa.values[3] - eigHpa.values[1])

d = qAnneal.densityMatrix_new(psiB, [6,0], I);   
Idel, ib = qAnneal.thermalization(d, eigHta.values, [6,0],1.0e-10)
log(d[1,1]) - log(d[3,3])
eigHta.values[3] - eigHta.values[1]
(log(d[1,1]) - log(d[3,3]))/(eigHta.values[3] - eigHta.values[1])

dlog = log.(d)
p = exp.(-eigHta.values/(2*T));
q = abs2.(p/norm(p))

p = exp.(-eigHpa.values/(2*T));
q = abs2.(p/norm(p))

# ==================================================
# Automated runs
Temperature = [0.001, 0.01, 0.1, 1, 10]
Coupling = [0,1,2]   # 0=No Coupling, 1=Coupling, 3=weak coupling
InitState = [0,1,2]  # 0=CTSE, 1=Equal Superposition, 2=All zeros
Random.seed!(10480);

rC = []
rI = []
rT = []
riσ = []                                      
riδ = []
rib = []
rσ = []                                      
rδ = []
rb = []

for C in Coupling
 Random.seed!(10480);
 if C == 0       # No Coupling
    v = eigHnc.vectors
    e = eigHnc.values
    H = Hnc
    ctext = "No Coupling"
 elseif C == 1   # Coupling
    v = eigHt.vectors
    e = eigHt.values
    H = Ht
    ctext = "Coupling"
 else            # Weak Coupling
    v = eigHt.vectors
    e = eigHt.values
    H = Ht
    ctext = "Weak Coupling"
 end
 @threads for Initial in InitState
  if Initial == 0
   itext = "CTSE" 
   for T in Temperature
    factor = exp(-H/(2*T))
    @threads for i = 1:5         # Numbers to average
     println("Coupling:",C," ,Initial State:",Initial, " ,T=",T," ,Loop #", i)
     if Initial == 0     # CTSE
         psi = qAnneal.randomStateBig(5,v);
         psiB = qAnneal.cannonical_state_fast(H,psi,T,factor)
     end
     
     if C == 0       # No Coupling
         T1,s1,del1,b1,psibc = qAnneal.annealTherm_constdiag(5,250.0,1.0,[3, 2], 0.0, 0,psiB,Hnc,Hp,eigHp.values, eigHp.vectors, Unc);
     elseif C == 1   # Coupling
         T1,s1,del1,b1,psibc = qAnneal.annealTherm_constdiag(5,250.0,1.0,[3, 2], 0.0, 0,psiB,Ht,Hp,eigHp.values, eigHp.vectors,Ut);
     else            # Weak Coupling
         T1,s1,del1,b1,psibc = qAnneal.annealTherm_constdiag(5,250.0,1.0,[3, 2], 0.0, 0,psiB,Hnc,Hp,eigHp.values, eigHp.vectors, Unc,Uc);
     end
     println("Starting plot with T1:",T1[1]," ,s1[1]:",s1[1])
     display(Plots.plot(T1,s1,label="sigma", title="gated 3-2 $ctext at T=$T starting at $itext", xlabel="time", ylabel="sigma"))
     display(Plots.plot(T1,del1,label="delta", title="gated 3-2 $ctext at T=$T starting at $itext", xlabel="time", ylabel="delta"))
     display(Plots.plot(T1,b1,label="b", title="gated 3-2 $ctext at T=$T starting at $itext", xlabel="time", ylabel="b"))
     push!(rC,C)
     push!(rI,Initial)
     push!(rT,T)
     push!(riσ,s1[1])
     push!(riδ,del1[1])
     push!(rib,b1[1])
     push!(rσ,s1[end])
     push!(rδ,del1[end])
     push!(rb,b1[end])
    end
   end
  else
   if Initial == 1 # Equal Superposition
    psiB = ones(Complex{BigFloat},2^5)
    psiB = psiB/norm(psiB) 
    itext = "Equal Superposition" 
   else                # All zeros
    psiB = Complex{BigFloat}.(state(ArrayReg(bit"00000")))
    itext = "All zeros" 
   end
   println("Coupling:",C," ,Initial State:",Initial, " ,T=",T)
   if C == 0       # No Coupling
    T1,s1,del1,b1,psibc = qAnneal.annealTherm_constdiag(5,250.0,1.0,[3, 2], 0.0, 0,psiB,Hnc,Hp,eigHp.values, eigHp.vectors, Unc);
   elseif C == 1   # Coupling
    T1,s1,del1,b1,psibc = qAnneal.annealTherm_constdiag(5,250.0,1.0,[3, 2], 0.0, 0,psiB,Ht,Hp,eigHp.values, eigHp.vectors,Ut);
   else            # Weak Coupling
    T1,s1,del1,b1,psibc = qAnneal.annealTherm_constdiag(5,250.0,1.0,[3, 2], 0.0, 0,psiB,Hnc,Hp,eigHp.values, eigHp.vectors, Unc,Uc);
   end
   println("Starting plot with T1:",T1[1]," ,s1[1]:",s1[1])
   display(Plots.plot(T1,s1,label="sigma", title="gated 3-2 $ctext starting at $itext", xlabel="time", ylabel="sigma"))
   display(Plots.plot(T1,del1,label="delta", title="gated 3-2 $ctext starting at $itext", xlabel="time", ylabel="delta"))
   display(Plots.plot(T1,b1,label="b", title="gated 3-2 $ctext starting at $itext", xlabel="time", ylabel="b"))
   push!(rC,C)
   push!(rI,Initial)
   push!(rT,"NA")
   push!(riσ,s1[1])
   push!(riδ,del1[1])
   push!(rib,b1[1])
   push!(rσ,s1[end])
   push!(rδ,del1[end])
   push!(rb,b1[end])
  end   
 end
end


a = hcat(rC,rI,rT,riσ,riδ,rib,rσ,rδ,rb);
b = Float64.(a);
b = conFloat.(a);


open("Sim2_out.csv", "w") do fil
    writedlm(fil,b,',')
end

for i = 1:2 
    display(Plots.plot(x,y,label="with Bath", title="b at T=$b", xlabel="time", ylabel="decoherence"))
end


# ====================================================
# Annealing

qAnneal.getConfig("config_4_12_decoh");
n = 5;
Random.seed!(97225);
Hta = qAnneal.HamiltonianBig(n, 2^n, 0);
eigHta=eigen(Hta);
Uta = eigHta.vectors*Diagonal(exp.(-im*eigHta.values))*eigHta.vectors'
Hpa = qAnneal.HamiltonianBig(3, 2^3, 0);
eigHpa=eigen(Hpa);

psi = qAnneal.randomStateBig(n,eigHta.vectors);
psiB = qAnneal.cannonical_state_diag(Hta,psi,0.001);


da = qAnneal.densityMatrix(psiB, [5,0], eigHta.vectors);
Float64.(qAnneal.thermalization(da, eigHta.values, [5,0]))

T3,s3,del3,b3,psibc = qAnneal.annealTherm_constdiag(5,500.0,1.0,[3, 2], 0.0, 0,psiB,Hta,Hpa,eigHpa.values,eigHpa.vectors,Uta);

display(Plots.plot(T1,s1,label="sigma", title="Adiabatic at T=0.001 starting at CTSE", xlabel="time", ylabel="sigma"))
display(Plots.plot(T1,del1,label="delta", title="Adiabatic at T=0.001 starting at CTSE", xlabel="time", ylabel="delta"))
display(Plots.plot(T1,b1,label="b", title="Adiabatic at T=0.001 starting at CTSE", xlabel="time", ylabel="b"))
display(Plots.plot(1 ./b1[80:end],del1[80:end],label="delta-1/b", title="Adiabatic at T=0.001 starting at CTSE", xlabel="1/b", ylabel="delta"))
display(Plots.plot(1 ./b1[80:end],s1[80:end],label="sigma-1/b", title="Adiabatic at T=0.001 starting at CTSE", xlabel="1/b", ylabel="sigma"))

psi = qAnneal.randomStateBig(n,eigHta.vectors);
psiB = qAnneal.cannonical_state_diag(Hta,psi,0.01);
T3,s3,del3,b3,psibc = qAnneal.annealTherm_constdiag(5,500.0,1.0,[3, 2], 0.0, 0,psiB,Hta,Hpa,eigHpa.values,eigHpa.vectors,Uta);

display(Plots.plot(T2,s2,label="sigma", title="Adiabatic at T=0.01 starting at CTSE", xlabel="time", ylabel="sigma"))
display(Plots.plot(T2,del2,label="delta", title="Adiabatic at T=0.01 starting at CTSE", xlabel="time", ylabel="delta"))
display(Plots.plot(T2,b2,label="b", title="Adiabatic at T=0.01 starting at CTSE", xlabel="time", ylabel="b"))
display(Plots.plot(1 ./b2[11:end],del2[11:end],label="delta-1/b", title="Adiabatic at T=0.01 starting at CTSE", xlabel="1/b", ylabel="delta"))
display(Plots.plot(1 ./b2[11:end],s2[11:end],label="sigma-1/b", title="Adiabatic at T=0.01 starting at CTSE", xlabel="1/b", ylabel="sigma"))
display(Plots.plot(1 ./b2[15:end],del2[15:end],label="delta-1/b", title="Adiabatic at T=0.01 starting at CTSE", xlabel="1/b", ylabel="delta"))
display(Plots.plot(1 ./b2[15:end],s2[15:end],label="sigma-1/b", title="Adiabatic at T=0.01 starting at CTSE", xlabel="1/b", ylabel="sigma"))

#

psia = qAnneal.EqStateBig(5,eigHta.vectors);
psiBa = qAnneal.cannonical_state_diag(Hta,psia,0.001);

da = qAnneal.densityMatrix(psiBa, [5,0], eigHta.vectors);
Float64.(qAnneal.thermalization(da, eigHta.values, [5,0]))

Hpa = qAnneal.HamiltonianBig(3, 2^3, 0);
eigHpa=eigen(Hpa);
da = qAnneal.densityMatrix(psiBa, [3,2], eigHpa.vectors);
Float64.(qAnneal.thermalization(da, eigHpa.values, [3,2]))


savefig(Plots.plot(T1,b1,label="sigma", title="Adiabatic at T=0.001 starting at CTSE", xlabel="time", ylabel="sigma"),"figures/$n testfig.png")

T1,s1,del1,b1,psibc = qAnneal.annealTherm_constdiag(5,50.0,1.0,[3, 2], 0.0, 0,psiB,Hta,Hpa,eigHpa.values,eigHpa.vectors,Uta);

# ==================================================
# Automated annealing runs

qAnneal.getConfig("config_4_12_decoh");
qAnneal.getConfig("config_4_12_w_bias");
qAnneal.getConfig("config_4_12_w_bias_nc");
qAnneal.getConfig("config_4_12_w_bias_wc");
qAnneal.getConfig("config_4_12_w_bias_sc");
qAnneal.getConfig("config_4_12_w_bias_sc005");
qAnneal.getConfig("config_4_12_w_bias_sc010");
qAnneal.getConfig("config_4_2_w_bias_sc015");
function conFloat(x)
    if isa(x,Number)
        return Float64(x)
    else
        return x
    end
end
n = 6;
Temperature = [1, 10]
Coupling = [0,1,2]   # 0=No Coupling, 1=Coupling, 3=weak coupling
InitState = [0]  # 0=CTSE, 1=Equal Superposition, 2=All zeros
Random.seed!(10480);
pretext = "Degen spin 4-2 wc2"

rC = []
rI = []
rT = []
riσ = []                                      
riδ = []
rib = []
rσ = []                                      
rδ = []
rb = []


Random.seed!(10480);
qAnneal.getConfig("config_4_2_degen_sc015");
Hta = qAnneal.HamiltonianBig(6, 2^6, 0);
eigHta=eigen(Hta);
Uta = eigHta.vectors*Diagonal(exp.(-im*eigHta.values))*eigHta.vectors';
#
#qAnneal.getConfig("config_4_12_w_bias_sc010");
#Htwc = qAnneal.HamiltonianBig(5, 2^5, 0);
#eigHtwc=eigen(Htwc);
#Utwc = eigHtwc.vectors*Diagonal(exp.(-im*eigHtwc.values))*eigHtwc.vectors';
#
#qAnneal.getConfig("config_4_12_w_bias_sc010");
#Htwc1 = qAnneal.HamiltonianBig(5, 2^5, 0);
#eigHtwc1=eigen(Htwc1);
#Utwc1 = eigHtwc1.vectors*Diagonal(exp.(-im*eigHtwc1.values))*eigHtwc1.vectors';
#
#qAnneal.getConfig("config_4_12_w_bias_nc");
#Htnc = qAnneal.HamiltonianBig(5, 2^5, 0);
#eigHtnc=eigen(Htnc);
#Utnc = eigHtnc.vectors*Diagonal(exp.(-im*eigHtnc.values))*eigHtnc.vectors';
Hpa = qAnneal.HamiltonianBig(4, 2^4, 0);
eigHpa=eigen(Hpa);
#

@threads for Initial in InitState
 if Initial == 0
  itext = "CTSE" 
  for T in Temperature
   factor = exp(-Hta/(2*T))
   @threads for i = 1:3         # Numbers to average
    println("Initial State:",Initial, " ,T=",T," ,Loop #", i)
    if Initial == 0     # CTSE
        psi = qAnneal.randomStateBig(6,eigHta.vectors);
        psiB = qAnneal.cannonical_state_fast(Hta,psi,T,factor)
    end
    T1,s1,del1,b1,psibc = qAnneal.annealTherm_constdiag(6,500.0,1.0,[4, 2], 0.0, 0,psiB,Hta,Hpa,eigHpa.values,eigHpa.vectors,[Uta],[250]);
    # println("Starting plot with T1:",T1[1]," ,s1[1]:",s1[1])
    savefig(Plots.plot(T1,s1,label="sigma", title="$pretext at T=$T starting at $itext", xlabel="time", ylabel="sigma"),"figures/$pretext $itext $T $i sigma.png")
    savefig(Plots.plot(T1,del1,label="delta", title="$pretext at T=$T starting at $itext", xlabel="time", ylabel="delta"),"figures/$pretext $itext $T $i delta.png")
    savefig(Plots.plot(T1,b1,label="b", title="$pretext at T=$T starting at $itext", xlabel="time", ylabel="b"),"figures/$pretext $itext $T $i b.png")
    push!(rC,"4-2")
    push!(rI,itext)
    push!(rT,T)
    push!(riσ,s1[1])
    push!(riδ,del1[1])
    push!(rib,b1[1])
    push!(rσ,s1[end])
    push!(rδ,del1[end])
    push!(rb,b1[end])
   end
  end
 else
  if Initial == 1 # Equal Superposition
   psiB = ones(Complex{BigFloat},2^6)
   psiB = psiB/norm(psiB) 
   itext = "Equal Superposition" 
  else                # All zeros
   psiB = Complex{BigFloat}.(state(ArrayReg(bit"000000")))
   itext = "All zeros" 
  end
  # println("Initial State:",Initial, " ,T=",T)
  T1,s1,del1,b1,psibc = qAnneal.annealTherm_constdiag(6,1000.0,1.0,[4, 2], 0.0, 0,psiB,Hta,Hpa,eigHpa.values,eigHpa.vectors,[Uta],[1000]);
  # println("Starting plot with T1:",T1[1]," ,s1[1]:",s1[1])
  savefig(Plots.plot(T1,s1,label="sigma", title="$pretext starting at $itext", xlabel="time", ylabel="sigma"),"figures/$pretext $itext sigma.png")
  savefig(Plots.plot(T1,del1,label="delta", title="$pretext starting at $itext", xlabel="time", ylabel="delta"),"figures/$pretext $itext delta.png")
  savefig(Plots.plot(T1,b1,label="b", title="$pretext at $itext", xlabel="time", ylabel="b"),"figures/$pretext $itext b.png")
  push!(rC,"4-2")
  push!(rI,itext)
  push!(rT,"NA")
  push!(riσ,s1[1])
  push!(riδ,del1[1])
  push!(rib,b1[1])
  push!(rσ,s1[end])
  push!(rδ,del1[end])
  push!(rb,b1[end])
 end   
end



a = hcat(rC,rI,rT,riσ,riδ,rib,rσ,rδ,rb);
b = conFloat.(a);


open("4_2_sim_degen_wc3.csv", "w") do fil
    writedlm(fil,b,',')
end

##########################################################################
# another run

Temperature = [0.001, 0.01, 0.1, 1, 10]
Coupling = [0,1,2]   # 0=No Coupling, 1=Coupling, 3=weak coupling
InitState = [1,2,0]  # 0=CTSE, 1=Equal Superposition, 2=All zeros
Random.seed!(10480);
pretext = "Multi-H spin 3-2 nc"

rC = []
rI = []
rT = []
riσ = []                                      
riδ = []
rib = []
rσ = []                                      
rδ = []
rb = []


Random.seed!(10480);
qAnneal.getConfig("config_4_12_w_bias_sc005");
Hta = qAnneal.HamiltonianBig(5, 2^5, 0);
eigHta=eigen(Hta);
Uta = eigHta.vectors*Diagonal(exp.(-im*eigHta.values))*eigHta.vectors';
#
qAnneal.getConfig("config_4_12_w_bias_sc010");
Htwc = qAnneal.HamiltonianBig(5, 2^5, 0);
eigHtwc=eigen(Htwc);
Utwc = eigHtwc.vectors*Diagonal(exp.(-im*eigHtwc.values))*eigHtwc.vectors';
#
#qAnneal.getConfig("config_4_12_w_bias_sc010");
#Htwc1 = qAnneal.HamiltonianBig(5, 2^5, 0);
#eigHtwc1=eigen(Htwc1);
#Utwc1 = eigHtwc1.vectors*Diagonal(exp.(-im*eigHtwc1.values))*eigHtwc1.vectors';
#
qAnneal.getConfig("config_4_12_w_bias_nc");
Htnc = qAnneal.HamiltonianBig(5, 2^5, 0);
eigHtnc=eigen(Htnc);
Utnc = eigHtnc.vectors*Diagonal(exp.(-im*eigHtnc.values))*eigHtnc.vectors';
Hpa = qAnneal.HamiltonianBig(3, 2^3, 0);
eigHpa=eigen(Hpa);
#

for Initial in InitState
 if Initial == 0
  itext = "CTSE" 
  for T in Temperature
   factor = exp(-Htnc/(2*T))
   for i = 1:3         # Numbers to average
    println("Initial State:",Initial, " ,T=",T," ,Loop #", i)
    if Initial == 0     # CTSE
        psi = qAnneal.randomStateBig(5,eigHtnc.vectors);
        psiB = qAnneal.cannonical_state_fast(Htnc,psi,T,factor)
    end
    T1,s1,del1,b1,psibc = qAnneal.annealTherm_constdiag(5,200.0,1.0,[3, 2], 0.0, 0,psiB,Hta,Hpa,eigHpa.values,eigHpa.vectors,[Uta,Utwc,Utnc],[10,40,150]);
    # println("Starting plot with T1:",T1[1]," ,s1[1]:",s1[1])
    savefig(Plots.plot(T1,s1,label="sigma", title="$pretext at T=$T starting at $itext", xlabel="time", ylabel="sigma"),"figures/$pretext $itext $T $i sigma.png")
    savefig(Plots.plot(T1,del1,label="delta", title="$pretext at T=$T starting at $itext", xlabel="time", ylabel="delta"),"figures/$pretext $itext $T $i delta.png")
    savefig(Plots.plot(T1,b1,label="b", title="$pretext at T=$T starting at $itext", xlabel="time", ylabel="b"),"figures/$pretext $itext $T $i b.png")
    push!(rC,"3-2")
    push!(rI,itext)
    push!(rT,T)
    push!(riσ,s1[1])
    push!(riδ,del1[1])
    push!(rib,b1[1])
    push!(rσ,s1[end])
    push!(rδ,del1[end])
    push!(rb,b1[end])
   end
  end
 else
  if Initial == 1 # Equal Superposition
   psiB = ones(Complex{BigFloat},2^5)
   psiB = psiB/norm(psiB) 
   itext = "Equal Superposition" 
  else                # All zeros
   psiB = Complex{BigFloat}.(state(ArrayReg(bit"00000")))
   itext = "All zeros" 
  end
  # println("Initial State:",Initial, " ,T=",T)
  T1,s1,del1,b1,psibc = qAnneal.annealTherm_constdiag(5,200.0,1.0,[3, 2], 0.0, 0,psiB,Hta,Hpa,eigHpa.values,eigHpa.vectors,[Uta,Utwc,Utnc],[10,40,150]);
  # println("Starting plot with T1:",T1[1]," ,s1[1]:",s1[1])
  savefig(Plots.plot(T1,s1,label="sigma", title="$pretext starting at $itext", xlabel="time", ylabel="sigma"),"figures/$pretext $itext sigma.png")
  savefig(Plots.plot(T1,del1,label="delta", title="$pretext starting at $itext", xlabel="time", ylabel="delta"),"figures/$pretext $itext delta.png")
  savefig(Plots.plot(T1,b1,label="b", title="$pretext at $itext", xlabel="time", ylabel="b"),"figures/$pretext $itext b.png")
  push!(rC,"3-2")
  push!(rI,itext)
  push!(rT,"NA")
  push!(riσ,s1[1])
  push!(riδ,del1[1])
  push!(rib,b1[1])
  push!(rσ,s1[end])
  push!(rδ,del1[end])
  push!(rb,b1[end])
 end   
end



a = hcat(rC,rI,rT,riσ,riδ,rib,rσ,rδ,rb);
b = conFloat.(a);


open("Multi_H_sim_out_can_nc.csv", "w") do fil
    writedlm(fil,b,',')
end
#

# ==================================================
# Automated 8 qubit annealing runs

qAnneal.getConfig("config_4_12_decoh");
n = 8;
Temperature = [0.001, 0.01, 0.1, 1, 10]
Coupling = [0,1,2]   # 0=No Coupling, 1=Coupling, 3=weak coupling
InitState = [0,1,2]  # 0=CTSE, 1=Equal Superposition, 2=All zeros
Random.seed!(10480);

rC = []
rI = []
rT = []
riσ = []                                      
riδ = []
rib = []
rσ = []                                      
rδ = []
rb = []


Random.seed!(10480);
Hta = qAnneal.HamiltonianBig(n, 2^n, 0);
eigHta=eigen(Hta);
Uta = eigHta.vectors*Diagonal(exp.(-im*eigHta.values))*eigHta.vectors'
Hpa = qAnneal.HamiltonianBig(4, 2^4, 0);
eigHpa=eigen(Hpa);

for Initial in InitState
 if Initial == 0
  itext = "CTSE" 
  for T in Temperature
   factor = exp(-Hta/(2*T))
   for i = 1:1         # Numbers to average
    println("Initial State:",Initial, " ,T=",T," ,Loop #", i)
    if Initial == 0     # CTSE
        psi = qAnneal.randomStateBig(5,eigHta.vectors);
        psiB = qAnneal.cannonical_state_fast(Hta,psi,T,factor)
    end
    T1,s1,del1,b1,psibc = qAnneal.annealTherm_constdiag(8,50.0,1.0,[4, 4], 0.0, 0,psiB,Hta,Hpa,eigHpa.values,eigHpa.vectors,Uta);
    # println("Starting plot with T1:",T1[1]," ,s1[1]:",s1[1])
    display(Plots.plot(T1,s1,label="sigma", title="annealing at T=$T starting at $itext", xlabel="time", ylabel="sigma"))
    display(Plots.plot(T1,del1,label="delta", title="annealing at T=$T starting at $itext", xlabel="time", ylabel="delta"))
    display(Plots.plot(T1,b1,label="b", title="annealing at T=$T starting at $itext", xlabel="time", ylabel="b"))
    push!(rI,Initial)
    push!(rT,T)
    push!(riσ,s1[1])
    push!(riδ,del1[1])
    push!(rib,b1[1])
    push!(rσ,s1[end])
    push!(rδ,del1[end])
    push!(rb,b1[end])
   end
  end
 else
  if Initial == 1 # Equal Superposition
   psiB = ones(Complex{BigFloat},2^5)
   psiB = psiB/norm(psiB) 
   itext = "Equal Superposition" 
  else                # All zeros
   psiB = Complex{BigFloat}.(state(ArrayReg(bit"00000")))
   itext = "All zeros" 
  end
  # println("Initial State:",Initial, " ,T=",T)
  T1,s1,del1,b1,psibc = qAnneal.annealTherm_constdiag(8,50.0,1.0,[4, 4], 0.0, 0,psiB,Hta,Hpa,eigHpa.values,eigHpa.vectors,Uta);
  # println("Starting plot with T1:",T1[1]," ,s1[1]:",s1[1])
  display(Plots.plot(T1,s1,label="sigma", title="annealing starting at $itext", xlabel="time", ylabel="sigma"))
  display(Plots.plot(T1,del1,label="delta", title="annealing starting at $itext", xlabel="time", ylabel="delta"))
  display(Plots.plot(T1,b1,label="b", title="annealing at starting at $itext", xlabel="time", ylabel="b"))
  push!(rI,Initial)
  push!(rT,T)
  push!(riσ,s1[1])
  push!(riδ,del1[1])
  push!(rib,b1[1])
  push!(rσ,s1[end])
  push!(rδ,del1[end])
  push!(rb,b1[end])
 end   
end



a = hcat(rC,rI,rT,riσ,riδ,rib,rσ,rδ,rb);
b = Float64.(a);


open("Annealing_sim_out.csv", "w") do fil
    writedlm(fil,b,',')
end

# =================
@threads for i=1:50
    println(i, ' ', threadid())
    @threads for j=1:10
        println("inside ", i,' ',j, ' ', threadid())
    end
end

s1=maximum(q)
s2=minimum(q)
s3=(s1-s2)/10
Plots.plot(p,q,yticks=[s2:s3:s1;],yformatter=y->(y),ylabel="test")

#Both the gates below are same
Matrix(chain(2,control(1, 2=>X),put(2=>Rz(BigFloat(π)/10)),control(1, 2=>X)))
Matrix(chain(2,control(2, 1=>X),put(1=>Rz(BigFloat(π)/10)),control(2, 1=>X)))

qAnneal.partialState(Int(0b11),[2,1],[0.6,0.4])
state(ArrayReg(bit"110"))

# ==============

@const_gate Hbig::Complex{BigFloat} = 1/sqrt(BigFloat(2))*[[1 1];[1 -1]];
A(i, j) = control(i, j=>shift(2*BigFloat(π)/(1<<(i-j+1))));
B1(t, n, s, k) = chain(t, j==k ? put(k+s-1=>Hbig) : A(j+s-1, k+s-1) for j in k:n);
qft1(t,n,s) = chain(B1(t, n, s, k) for k in 1:n);       #t=totalqubits, n=nqubit fourier,s=starting qubit
#YaoPlots.plot(qft1(4,3,2))

# UTotal = chain(7,repeat(Hbig,[1 2 3 4 5 6 7]),qft1(7,4,1),qft1(7,3,5),repeat(X,[1 2 3 4 5 6 7]), control(4, 5=>X),put(5=>Rz(BigFloat(π)/100)),control(4, 5=>X), control(7, 1=>X),put(1=>Rz(BigFloat(π)/100)),control(7, 1=>X));
# UProblem = chain(4,repeat(Hbig,[1 2 3 4]),qft1(4,4,1),repeat(X,[1 2 3 4]));
# UBath = chain(3,repeat(Hbig,[1 2 3]),qft1(3,3,1),repeat(X,[1 2 3]));

# UTotal1 = chain(7,repeat(Hbig,[1 2 3 4 5 6 7]),qft1(7,4,1),qft1(7,3,5),repeat(X,[1 2 3 4 5 6 7]), control(4, 5=>X),put(5=>Rz(π/100)),control(4, 5=>X), control(7, 1=>X),put(1=>Rz(π/100)),control(7, 1=>X));

"""
# wc4 and wc5
UTotal = chain(7,repeat(Hbig,[1 2]),control(1, 2=>X),put(2=>Rz(BigFloat(π)/10)),control(1, 2=>X),repeat(Hbig,[1 2]),                              
                        repeat(Hbig,[2 3]),control(2, 3=>X),put(3=>Rz(BigFloat(π)/15)),control(2, 3=>X), repeat(Hbig,[2 3]),
                        control(2, 3=>X),put(3=>Rz(BigFloat(π)/15)),control(2, 3=>X),                                                        
                        repeat(Hbig,[3 4]),control(3, 4=>X),put(4=>Rz(BigFloat(π)/12)),control(3, 4=>X), repeat(Hbig,[3 4]),
                        repeat(Rz(BigFloat(π)/2),[3 4]), repeat(Hbig,[3 4]),control(3, 4=>X),put(4=>Rz(BigFloat(π)/12)),control(3, 4=>X), repeat(Hbig,[3 4]), repeat(Rz(-BigFloat(π)/2),[3 4]),                 
                        repeat(Hbig,[4 5]),control(4, 5=>X),put(5=>Rz(BigFloat(π)/10)),control(4, 5=>X),repeat(Hbig,[4 5]), 
                        repeat(Hbig,[5 6]),control(5, 6=>X),put(6=>Rz(BigFloat(π)/50)),control(5, 6=>X), repeat(Hbig,[5 6]),
                        repeat(Hbig,[6 7]),control(6, 7=>X),put(7=>Rz(BigFloat(π)/150)),control(6, 7=>X), repeat(Hbig,[6 7]),
                        control(6, 7=>X),put(7=>Rz(BigFloat(π)/50)),control(6, 7=>X),
                        repeat(Hbig,[7 1]),control(7, 1=>X),put(1=>Rz(BigFloat(π)/14)),control(7, 1=>X), repeat(Hbig,[7 1]));

UBath = chain(3,repeat(Hbig,[1 2]),control(1, 2=>X),put(2=>Rz(BigFloat(π)/50)),control(1, 2=>X), repeat(Hbig,[1 2]),
                        repeat(Hbig,[2 3]),control(2, 3=>X),put(3=>Rz(BigFloat(π)/150)),control(2, 3=>X), repeat(Hbig,[2 3]),
                        control(2, 3=>X),put(3=>Rz(BigFloat(π)/50)),control(2, 3=>X));

UProblem = chain(4,repeat(Hbig,[1 2]),control(1, 2=>X),put(2=>Rz(BigFloat(π)/10)),control(1, 2=>X),repeat(Hbig,[1 2]),                             
                        repeat(Hbig,[2 3]),control(2, 3=>X),put(3=>Rz(BigFloat(π)/15)),control(2, 3=>X), repeat(Hbig,[2 3]),
                        control(2, 3=>X),put(3=>Rz(BigFloat(π)/15)),control(2, 3=>X),                                                           
                        repeat(Hbig,[3 4]),control(3, 4=>X),put(4=>Rz(BigFloat(π)/12)),control(3, 4=>X), repeat(Hbig,[3 4]),
                        repeat(Rz(BigFloat(π)/2),[3 4]), repeat(Hbig,[3 4]),control(3, 4=>X),put(4=>Rz(BigFloat(π)/12)),control(3, 4=>X), repeat(Hbig,[3 4]), repeat(Rz(-BigFloat(π)/2),[3 4]));                        

# UTnoCoupling = chain(7,repeat(Hbig,[1 2 3 4 5 6 7]),qft1(7,4,1),qft1(7,3,5),repeat(X,[1 2 3 4 5 6 7]));
UTnoCoupling = chain(7,repeat(Hbig,[1 2]),control(1, 2=>X),put(2=>Rz(BigFloat(π)/10)),control(1, 2=>X),repeat(Hbig,[1 2]),                              
                   repeat(Hbig,[2 3]),control(2, 3=>X),put(3=>Rz(BigFloat(π)/15)),control(2, 3=>X), repeat(Hbig,[2 3]),
                   control(2, 3=>X),put(3=>Rz(BigFloat(π)/15)),control(2, 3=>X),                                                        
                   repeat(Hbig,[3 4]),control(3, 4=>X),put(4=>Rz(BigFloat(π)/12)),control(3, 4=>X), repeat(Hbig,[3 4]),
                   repeat(Rz(BigFloat(π)/2),[3 4]), repeat(Hbig,[3 4]),control(3, 4=>X),put(4=>Rz(BigFloat(π)/12)),control(3, 4=>X), repeat(Hbig,[3 4]), repeat(Rz(-BigFloat(π)/2),[3 4]),                 
                   repeat(Hbig,[5 6]),control(5, 6=>X),put(6=>Rz(BigFloat(π)/50)),control(5, 6=>X), repeat(Hbig,[5 6]),
                   repeat(Hbig,[6 7]),control(6, 7=>X),put(7=>Rz(BigFloat(π)/150)),control(6, 7=>X), repeat(Hbig,[6 7]),
                   control(6, 7=>X),put(7=>Rz(BigFloat(π)/50)),control(6, 7=>X));
# UCoupling = chain(7,control(4, 5=>X),put(5=>Rz(BigFloat(π)/100)),control(4, 5=>X), control(7, 1=>X),put(1=>Rz(BigFloat(π)/100)),control(7, 1=>X));
UCoupling = chain(7,repeat(Hbig,[4 5]),control(4, 5=>X),put(5=>Rz(BigFloat(π)/10)),control(4, 5=>X),repeat(Hbig,[4 5]), 
                  repeat(Hbig,[7 1]),control(7, 1=>X),put(1=>Rz(BigFloat(π)/14)),control(7, 1=>X), repeat(Hbig,[7 1]));

 YaoPlots.plot(UTotal)
 YaoPlots.plot(UProblem)
 YaoPlots.plot(UBath)
 YaoPlots.plot(UTnoCoupling)
 YaoPlots.plot(UCoupling)
"""
# wc6 and wc7
UTotal = chain(7,repeat(Hbig,[1 2]),control(1, 2=>X),put(2=>Rz(BigFloat(π)/2)),control(1, 2=>X),repeat(Hbig,[1 2]),                              
                        repeat(Hbig,[2 3]),control(2, 3=>X),put(3=>Rz(BigFloat(π)/5)),control(2, 3=>X), repeat(Hbig,[2 3]),
                        control(2, 3=>X),put(3=>Rz(BigFloat(π)/5)),control(2, 3=>X),                                                        
                        repeat(Hbig,[3 4]),control(3, 4=>X),put(4=>Rz(BigFloat(π)/2)),control(3, 4=>X), repeat(Hbig,[3 4]),
                        repeat(Rz(BigFloat(π)/2),[3 4]), repeat(Hbig,[3 4]),control(3, 4=>X),put(4=>Rz(BigFloat(π)/2)),control(3, 4=>X), repeat(Hbig,[3 4]), repeat(Rz(-BigFloat(π)/2),[3 4]),                 
                        repeat(Hbig,[4 5]),control(4, 5=>X),put(5=>Rz(BigFloat(π)/5)),control(4, 5=>X),repeat(Hbig,[4 5]), 
                        repeat(Hbig,[5 6]),control(5, 6=>X),put(6=>Rz(BigFloat(π)/3)),control(5, 6=>X), repeat(Hbig,[5 6]),
                        repeat(Hbig,[6 7]),control(6, 7=>X),put(7=>Rz(BigFloat(π)/15)),control(6, 7=>X), repeat(Hbig,[6 7]),
                        control(6, 7=>X),put(7=>Rz(BigFloat(π)/5)),control(6, 7=>X),
                        repeat(Hbig,[7 1]),control(7, 1=>X),put(1=>Rz(BigFloat(π)/4)),control(7, 1=>X), repeat(Hbig,[7 1]));

UBath = chain(3,repeat(Hbig,[1 2]),control(1, 2=>X),put(2=>Rz(BigFloat(π)/3)),control(1, 2=>X), repeat(Hbig,[1 2]),
                        repeat(Hbig,[2 3]),control(2, 3=>X),put(3=>Rz(BigFloat(π)/15)),control(2, 3=>X), repeat(Hbig,[2 3]),
                        control(2, 3=>X),put(3=>Rz(BigFloat(π)/5)),control(2, 3=>X));

UProblem = chain(4,repeat(Hbig,[1 2]),control(1, 2=>X),put(2=>Rz(BigFloat(π)/2)),control(1, 2=>X),repeat(Hbig,[1 2]),                             
                        repeat(Hbig,[2 3]),control(2, 3=>X),put(3=>Rz(BigFloat(π)/5)),control(2, 3=>X), repeat(Hbig,[2 3]),
                        control(2, 3=>X),put(3=>Rz(BigFloat(π)/5)),control(2, 3=>X),                                                           
                        repeat(Hbig,[3 4]),control(3, 4=>X),put(4=>Rz(BigFloat(π)/2)),control(3, 4=>X), repeat(Hbig,[3 4]),
                        repeat(Rz(BigFloat(π)/2),[3 4]), repeat(Hbig,[3 4]),control(3, 4=>X),put(4=>Rz(BigFloat(π)/2)),control(3, 4=>X), repeat(Hbig,[3 4]), repeat(Rz(-BigFloat(π)/2),[3 4]));                        

# UTnoCoupling = chain(7,repeat(Hbig,[1 2 3 4 5 6 7]),qft1(7,4,1),qft1(7,3,5),repeat(X,[1 2 3 4 5 6 7]));
UTnoCoupling = chain(7,repeat(Hbig,[1 2]),control(1, 2=>X),put(2=>Rz(BigFloat(π)/2)),control(1, 2=>X),repeat(Hbig,[1 2]),                              
                   repeat(Hbig,[2 3]),control(2, 3=>X),put(3=>Rz(BigFloat(π)/5)),control(2, 3=>X), repeat(Hbig,[2 3]),
                   control(2, 3=>X),put(3=>Rz(BigFloat(π)/5)),control(2, 3=>X),                                                        
                   repeat(Hbig,[3 4]),control(3, 4=>X),put(4=>Rz(BigFloat(π)/2)),control(3, 4=>X), repeat(Hbig,[3 4]),
                   repeat(Rz(BigFloat(π)/2),[3 4]), repeat(Hbig,[3 4]),control(3, 4=>X),put(4=>Rz(BigFloat(π)/2)),control(3, 4=>X), repeat(Hbig,[3 4]), repeat(Rz(-BigFloat(π)/2),[3 4]),                 
                   repeat(Hbig,[5 6]),control(5, 6=>X),put(6=>Rz(BigFloat(π)/3)),control(5, 6=>X), repeat(Hbig,[5 6]),
                   repeat(Hbig,[6 7]),control(6, 7=>X),put(7=>Rz(BigFloat(π)/15)),control(6, 7=>X), repeat(Hbig,[6 7]),
                   control(6, 7=>X),put(7=>Rz(BigFloat(π)/5)),control(6, 7=>X));
# UCoupling = chain(7,control(4, 5=>X),put(5=>Rz(BigFloat(π)/100)),control(4, 5=>X), control(7, 1=>X),put(1=>Rz(BigFloat(π)/100)),control(7, 1=>X));
UCoupling = chain(7,repeat(Hbig,[4 5]),control(4, 5=>X),put(5=>Rz(BigFloat(π)/5)),control(4, 5=>X),repeat(Hbig,[4 5]), 
                  repeat(Hbig,[7 1]),control(7, 1=>X),put(1=>Rz(BigFloat(π)/4)),control(7, 1=>X), repeat(Hbig,[7 1]));                  


Ut = Matrix(UTotal);
eigt = eigen(Ut);
Ht = Hermitian(im * eigt.vectors * Diagonal(log.(eigt.values)) * eigt.vectors');
eigHt = eigen(Ht);

Up = Matrix(UProblem);
eigp = eigen(Up);
Hp = Hermitian(im * eigp.vectors * Diagonal(log.(eigp.values)) * eigp.vectors');
eigHp = eigen(Hp);

Ub = Matrix(UBath);
eigb = eigen(Ub);
Hb = Hermitian(im * eigb.vectors * Diagonal(log.(eigb.values)) * eigb.vectors');
eigHb = eigen(Hb);

Unc = Matrix(UTnoCoupling);

Uc = Matrix(UCoupling);

function conFloat(x)
    if isa(x,Number)
        return Float64(x)
    else
        return x
    end
end
n = 7;
Temperature = [0.001, 0.01, 0.1, 1, 10]
#Temperature = [0.1, 1]
Coupling = [0,1,2]   # 0=No Coupling, 1=Coupling, 3=weak coupling
#InitState = [0,1,2,3,4,3,4,0, 3, 3 ]  # 0=CTSE, 1=Equal Superposition, 2=All zeros, 3=Random, 4=ENV only in CTSE 
InitState = [4]  # 0=CTSE, 1=Equal Superposition, 2=All zeros, 3=Random, 4=ENV only in CTSE 
#InitState = [0, 1, 2, 3 ]
Random.seed!(10480);
pretext = "Gated 4-3 wc55"

rC   = Vector{Any}(["System Type"])      
rI   = Vector{Any}(["Initial Condition"])
rT   = Vector{Any}(["Temperature"])
riσ  = Vector{Any}(["initial Sigma"]  )                                    
riδ  = Vector{Any}(["Initial Delta"])
rib  = Vector{Any}(["Initial Beta"])
rims = Vector{Any}(["Initial 50 Sigma Mean"])
rivs = Vector{Any}(["Initial 50 Sigma Var"])
rimd = Vector{Any}(["Initial 50 delta Mean"])
rivd = Vector{Any}(["Initial 50 delta Var"])
rimb = Vector{Any}(["Initial 50 b Mean"])
rivb = Vector{Any}(["Initial 50 b Var"])
rσ   = Vector{Any}(["FInal Sigma"]   )                                   
rδ   = Vector{Any}(["Final Delta"])
rb   = Vector{Any}(["Final beta"])
rms  = Vector{Any}(["Final 50 Sigma Mean"])
rvs  = Vector{Any}(["Final 50 Sigma Var"])
rmd  = Vector{Any}(["Final 50 delta Mean"])
rvd  = Vector{Any}(["Final 50 delta Var"])
rmb  = Vector{Any}(["Final 50 b Mean"])
rvb  = Vector{Any}(["Final 50 b Var"])


Random.seed!(10480);
# qAnneal.getConfig("config_4_2_degen_sc015");
# Hta = qAnneal.HamiltonianBig(6, 2^6, 0);
#eigHta=eigen(Hta);
#Uta = eigHta.vectors*Diagonal(exp.(-im*eigHta.values))*eigHta.vectors';
#
#qAnneal.getConfig("config_4_12_w_bias_sc010");
#Htwc = qAnneal.HamiltonianBig(5, 2^5, 0);
#eigHtwc=eigen(Htwc);
#Utwc = eigHtwc.vectors*Diagonal(exp.(-im*eigHtwc.values))*eigHtwc.vectors';
#
#qAnneal.getConfig("config_4_12_w_bias_sc010");
#Htwc1 = qAnneal.HamiltonianBig(5, 2^5, 0);
#eigHtwc1=eigen(Htwc1);
#Utwc1 = eigHtwc1.vectors*Diagonal(exp.(-im*eigHtwc1.values))*eigHtwc1.vectors';
#
#qAnneal.getConfig("config_4_12_w_bias_nc");
#Htnc = qAnneal.HamiltonianBig(5, 2^5, 0);
#eigHtnc=eigen(Htnc);
#Utnc = eigHtnc.vectors*Diagonal(exp.(-im*eigHtnc.values))*eigHtnc.vectors';
#Hpa = qAnneal.HamiltonianBig(4, 2^4, 0);
#eigHpa=eigen(Hpa);
#

j=0;
const lk = ReentrantLock()
for Initial in InitState
 if Initial == 0 || Initial == 4
  for T in Temperature
   if Initial == 0   
    factor = exp(-Ht/(2*T))
   elseif Initial == 4 
    factor = exp(-Hb/(2*T))
   else
    println("incorrect initial state")
   end
   @threads for i = 1:3         # Numbers to average
    println("Initial State:",Initial, " ,T=",T," ,Loop #", i)
    if Initial == 0     # CTSE
        psi = qAnneal.randomStateBig(7,eigHt.vectors);
        psiB = qAnneal.cannonical_state_fast(Ht,psi,T,factor)
        itext = "Full CTSE" 
    elseif Initial == 4
        psib = qAnneal.randomStateBig(3,eigHb.vectors);
        psibB = qAnneal.cannonical_state_fast(Hb,psib,T,factor)
        psiBfull = qAnneal.partialState(Int(0b1010),[4,3],psibB)
        psiB = psiBfull/norm(psiBfull)
        itext = "Env CTSE" 
    else
        println("incorrect initial state")
    end
    T1,s1,del1,b1,psibc = qAnneal.annealTherm_constdiag(7,500.0,1.0,[4, 3], 0.0, 0,psiB,Ht,Hp,eigHp.values,eigHp.vectors,[Ut],[500]);
    #T1,s1,del1,b1,psibc = qAnneal.annealTherm_constdiag(7,6.0,1.0,[4, 3], 0.0, 0,psiB, missing,Hp,eigHp.values,eigHt.vectors,[Ut,Unc,Uc,Unc,Uc,Unc,Uc,Unc,Uc,Unc,Uc,Unc,Uc,Unc],[50,50,1,50,1,50,1,50,1,75,1,75,1,300]);
    # println("Starting plot with T1:",T1[1]," ,s1[1]:",s1[1])
    lock(lk) do
      max=maximum(s1)
      min=minimum(s1)
      delta=(max-min)/10
      savefig(Plots.plot(T1,s1,label="sigma", title="$pretext at T=$T starting at $itext", xlabel="time", yticks=[min:delta:max;],yformatter=y->(y), ylabel="sigma"),"figures/$pretext $itext $T $i sigma.png")
      max=maximum(del1)
      min=minimum(del1)
      delta=(max-min)/10
      savefig(Plots.plot(T1,del1,label="delta", title="$pretext at T=$T starting at $itext", xlabel="time", yticks=[min:delta:max;],yformatter=y->(y), ylabel="delta"),"figures/$pretext $itext $T $i delta.png")
      max=maximum(b1)
      min=minimum(b1)
      delta=(max-min)/10
      savefig(Plots.plot(T1,b1,label="b", title="$pretext at T=$T starting at $itext", xlabel="time", yticks=[min:delta:max;],yformatter=y->(y), ylabel="b"),"figures/$pretext $itext $T $i b.png")
    
      push!(rC,"4-3")
      push!(rI,itext)
      push!(rT,T)
      push!(riσ,s1[1])
      push!(riδ,del1[1])
      push!(rib,b1[1])
      push!(rims,mean(s1[1:50]))
      push!(rivs,var(s1[1:50]))
      push!(rimd,mean(del1[1:50]))
      push!(rivd,var(del1[1:50]))
      push!(rimb,mean(b1[1:50]))
      push!(rivb,var(b1[1:50]))
      push!(rσ,s1[end])
      push!(rδ,del1[end])
      push!(rb,b1[end])
      push!(rms,mean(s1[end-150:end]))
      push!(rvs,var(s1[end-50:end]))
      push!(rmd,mean(del1[end-50:end]))
      push!(rvd,var(del1[end-50:end]))
      push!(rmb,mean(b1[end-50:end]))
      push!(rvb,var(b1[end-50:end]))
    end
   end
  end
 else
  if Initial == 1 # Equal Superposition
   psiB = ones(Complex{BigFloat},2^7)
   psiB = psiB/norm(psiB) 
   itext = "Equal Superposition" 
  elseif Initial == 2                # All zeros
   psiB = Complex{BigFloat}.(state(ArrayReg(bit"0000000")))
   itext = "All zeros" 
  elseif Initial == 3
   psiB = qAnneal.rand(2^7);
   psiB = psiB/norm(psiB);
   global j = j + 1
   itext = "Random $j"
   println("Thread ID: ", threadid())
  else
   println("incorrect initial state")
  end
  # println("Initial State:",Initial, " ,T=",T)
  T1,s1,del1,b1,psibc = qAnneal.annealTherm_constdiag(7,1000.0,1.0,[4, 3], 0.0, 0,psiB,Ht,Hp,eigHp.values,eigHp.vectors,[Ut],[1000]);
  # T1,s1,del1,b1,psibc = qAnneal.annealTherm_constdiag(7,10.0,1.0,[4, 3], 0.0, 0,psiB, missing,Hp,eigHp.values,eigHt.vectors,[Ut,Unc,Uc,Unc,Uc,Unc,Uc,Unc,Uc,Unc,Uc,Unc,Uc,Unc,Uc,Unc],[50,50,1,50,1,50,1,50,1,75,1,75,1,300,1,500]);
  # println("Starting plot with T1:",T1[1]," ,s1[1]:",s1[1])
  lock(lk) do
    max=maximum(s1)
    min=minimum(s1)
    delta=(max-min)/10
    #savefig(Plots.plot(T1,s1,label="sigma", title="$pretext starting at $itext", xlabel="time",yticks=[min:delta:max;],yformatter=y->(y), ylabel="sigma",size=(1200,800)),"figures/$pretext $itext sigma.png")
    savefig(Plots.plot(T1,s1,label="sigma", title="$pretext starting at $itext", xlabel="time",yticks=[min:delta:max;],yformatter=y->(y), ylabel="sigma"),"figures/$pretext $itext sigma.png")
    max=maximum(del1)
    min=minimum(del1)
    delta=(max-min)/10
    savefig(Plots.plot(T1,del1,label="delta", title="$pretext starting at $itext", xlabel="time",yticks=[min:delta:max;],yformatter=y->(y), ylabel="delta"),"figures/$pretext $itext delta.png")
    max=maximum(b1)
    min=minimum(b1)
    delta=(max-min)/10
    savefig(Plots.plot(T1,b1,label="b", title="$pretext at $itext", xlabel="time",yticks=[min:delta:max;],yformatter=y->(y), ylabel="b"),"figures/$pretext $itext b.png")
  
    push!(rC,"4-3")
    push!(rI,itext)
    push!(rT,"NA")
    push!(riσ,s1[1])
    push!(riδ,del1[1])
    push!(rib,b1[1])
    push!(rims,mean(s1[1:25]))
    push!(rivs,var(s1[1:25]))
    push!(rimd,mean(del1[1:25]))
    push!(rivd,var(del1[1:25]))
    push!(rimb,mean(b1[1:25]))
    push!(rivb,var(b1[1:25]))
    push!(rσ,s1[end])
    push!(rδ,del1[end])
    push!(rb,b1[end])
    push!(rms,mean(s1[end-100:end]))
    push!(rvs,var(s1[end-100:end]))
    push!(rmd,mean(del1[end-100:end]))
    push!(rvd,var(del1[end-100:end]))
    push!(rmb,mean(b1[end-100:end]))
    push!(rvb,var(b1[end-100:end]))
  end
 end   
end



a = hcat(rC,rI,rT,riσ,riδ,rib,rims,rivs,rimd,rivd,rimb,rivb,rσ,rδ,rb,rms,rvs,rmd,rvd,rmb,rvb);
b = conFloat.(a);


open("4_3_gated_wc55.csv", "w") do fil
    writedlm(fil,b,',')
end


# ==================================================
# Validating spin systems from paper

qAnneal.getConfig("config_4_8_paper_degen");
Htt = qAnneal.Hamiltonian(12, 2^12, 0);
eigHtt=eigen(Htt);
#Uta = eigHta.vectors*Diagonal(exp.(-im*eigHta.values))*eigHta.vectors'
Hpt = qAnneal.Hamiltonian(4, 2^4, 0);
eigHpt=eigen(Hpt);

qAnneal.getConfig("config_4_8_paper");
Htt = qAnneal.Hamiltonian32(12, 2^12, 0);
eigHtt=eigen(Htt);
#Uta = eigHta.vectors*Diagonal(exp.(-im*eigHta.values))*eigHta.vectors'
Hpt = qAnneal.Hamiltonian32(4, 2^4, 0);
eigHpt=eigen(Hpt);
Hbt = qAnneal.Hamiltonian32(8, 2^8, 0,4);
eigHbt=eigen(Hbt);

factor = exp(-Ht/(2*T))
#psi = qAnneal.randomStateBig(7,eigHt.vectors);
psi = qAnneal.EqStateBig(n,eigHnc.vectors)
psiB = qAnneal.cannonical_state_fast(Ht,psi,T,factor)


qAnneal.Hamiltonian64(4, 2^4, 0)

T = 0.01;
psi = qAnneal.EqStateBig(7,eigHt.vectors);
psiB = qAnneal.cannonical_state_diag(Ht,psi,T);
d = qAnneal.densityMatrix(psiB, [7,0], eigHt.vectors);
qAnneal.thermalization(d, eigHt.values, [7,0])

psi = qAnneal.EqStateBig(7,eigHt.vectors);
psiB = qAnneal.cannonical_state_diag(Ht,psi,T);
d = qAnneal.densityMatrix(psiB, [4,3], eigHt.vectors);
qAnneal.thermalization(d, eigHp.values, [4,3])

psi = qAnneal.EqStateBig(7,eigHnc.vectors);
psiB = qAnneal.cannonical_state_diag(Hnc,psi,T);
d = qAnneal.densityMatrix(psiB, [4,3], eigHt.vectors);
qAnneal.thermalization(d, eigHp.values, [4,3])

abs.(eigHt.vectors'*psiB)
abs.(eigHt.vectors'*Ut*psiB)
eigHt.vectors'*Ut*eigHt.vectors

# =======================================================

qAnneal.getConfig("config_diag");
Hpt = Hermitian(BigFloat.(qAnneal.Hamiltonian32(2, 2^2, 0)));
Htt = Hermitian(BigFloat.(qAnneal.Hamiltonian32(4, 2^4, 0)));
Hbt = Hermitian(BigFloat.(qAnneal.Hamiltonian32(2, 2^2, 0,2)));
kron(Hbt,Hpt);
Ub = exp(-im*Hbt);
Up = exp(-im*Hpt);
Ut = exp(-im*Htt);
kron(Ub,Up);
Ft = exp(-Htt);      # T=0.5
Fp = exp(-Hpt);
Fb = exp(-Hbt);
kron(Fb,Fp);
eigHt=eigen(Htt);
eigHp=eigen(Hpt);
eigHb=eigen(Hbt);

Htd = Diagonal(eigHt.values)
Hpd = Diagonal(eigHp.values)
Hbd = Diagonal(eigHb.values)

psi = qAnneal.EqStateBig(4,eigHt.vectors);
T=0.5;
psiB = qAnneal.cannonical_state_fast(Htt,psi,T,Ft);
dt = qAnneal.densityMatrix(psiB, [4,0], eigHt.vectors);
qAnneal.thermalization(d, eigHt.values, [4,0])
dp = qAnneal.densityMatrix(psiB, [2,2], eigHp.vectors);
qAnneal.thermalization(d, eigHp.values, [2,2])

diat = Diagonal(qAnneal.normWaveFun(exp.(eigHt.values)))
diap = Diagonal(qAnneal.normWaveFun(exp.(eigHp.values)))
diab = Diagonal(qAnneal.normWaveFun(exp.(eigHb.values)))



v= eigHt.vectors;
pd1 = qAnneal.EqStateBig(4,I);
p1 = qAnneal.EqStateBig(4,eigHt.vectors);
pd1 == v'*p1

fd1 = Diagonal(exp.(-eigHt.values))
f1 = exp(-Htt);      # T=0.5
f1 = v * fd1 * v'

pBd1 = qAnneal.cannonical_state_fast(Htt,pd1,T,fd1)
pB1 = qAnneal.cannonical_state_fast(Htt,p1,T,f1)
pBd1 == v'*pB1

ddt = qAnneal.densityMatrix(pBd1, [4,0], I)
dt = qAnneal.densityMatrix(pB1, [4,0], eigHt.vectors)

x1 = abs2.(qAnneal.normWaveFun(exp.(-eigHt.values)))
x2 = abs2.(qAnneal.normWaveFun(fd1*pd1))

x3 = qAnneal.normWaveFun(exp.(-eigHt.values))
x3[1]*conj(x3[1])

qAnneal.thermalization(dt, eigHt.values, [4,0])
qAnneal.thermalization(ddt, eigHt.values, [4,0])
qAnneal.thermalization(Diagonal(x1), eigHt.values, [4,0], 1.0e-50)

fdt = Diagonal(exp.(-eigHt.values))
fdb = Diagonal(exp.(-eigHb.values))
fdp = Diagonal(exp.(-eigHp.values))

pd2 = qAnneal.EqStateBig(2,I);
pBdt = qAnneal.cannonical_state_fast(Htt,pd1,T,fdt)
pBdp = qAnneal.cannonical_state_fast(Hpt,pd2,T,fdp)
pBdb = qAnneal.cannonical_state_fast(Hpt,pd2,T,fdb)
kron(pBdb,pBdp)

pt = qAnneal.normWaveFun(exp.(-eigHt.values))
pp = qAnneal.normWaveFun(exp.(-eigHp.values))
pb = qAnneal.normWaveFun(exp.(-eigHb.values))
kron(pb,pp)

qt = qAnneal.normWaveFun(exp.(-im*eigHt.values))
qp = qAnneal.normWaveFun(exp.(-im*eigHp.values))
qb = qAnneal.normWaveFun(exp.(-im*eigHb.values))
kron(qb,qp)

Ub = exp(-im*Hbt);
Up = exp(-im*Hpt);
Ut = exp(-im*Htt);
kron(Ub,Up);

fb = exp(-Hbt);
fp = exp(-Hpt);
ft = exp(-Htt);
kron(fb,fp)-ft

psit = qAnneal.EqStateBig(4,eigHt.vectors);
psip = qAnneal.EqStateBig(2,eigHp.vectors);
psib = qAnneal.EqStateBig(2,eigHb.vectors);

f1t = ft * psit;
f1p = fp * psip;
f1b = fb * psib;
kron(f1b,f1p)-f1t

dtt = qAnneal.densityMatrix(f1t, [4,0], eigHt.vectors);
qAnneal.thermalization(dtt, eigHt.values, [4,0])
dpt = qAnneal.densityMatrix(f1t, [2,2], eigHp.vectors);
qAnneal.thermalization(dpt, eigHp.values, [2,2])

f2t = eigHt.vectors*Diagonal(exp.(-eigHt.values))*qAnneal.EqStateBig(4,I);
f2p = eigHp.vectors*Diagonal(exp.(-eigHp.values))*qAnneal.EqStateBig(2,I);
f2b = eigHb.vectors*Diagonal(exp.(-eigHb.values))*qAnneal.EqStateBig(2,I);
Float64.(kron(f2b,f2p)-f2t)

f3t = exp(-Htt)*eigHt.vectors*qAnneal.EqStateBig(4,I);
f3p = exp(-Hpt)*eigHp.vectors*qAnneal.EqStateBig(2,I);
f3b = exp(-Hbt)*eigHb.vectors*qAnneal.EqStateBig(2,I);
Float64.(kron(f3b,f3p)-f3t)

eigHt.vectors*Diagonal(exp.(-eigHt.values))*eigHt.vectors' - exp(-Htt)
eigHt.vectors*Diagonal(exp.(-im*eigHt.values))*eigHt.vectors' - exp(-im*Htt)

# LinearAlgebra.BLAS.vendor()
BLAS.get_config()

Hpt = (qAnneal.Hamiltonian32(2, 2^2, 0));
Htt = (qAnneal.Hamiltonian32(4, 2^4, 0));
Hbt = (qAnneal.Hamiltonian32(2, 2^2, 0,2));

Hpt = Hermitian(BigFloat.(qAnneal.Hamiltonian32(2, 2^2, 0)));
Htt = Hermitian(BigFloat.(qAnneal.Hamiltonian32(4, 2^4, 0)));
Hbt = Hermitian(BigFloat.(qAnneal.Hamiltonian32(2, 2^2, 0,2)));
eigHt=eigen(Htt,sortby=nothing);
eigHp=eigen(Hpt,sortby=nothing);
eigHb=eigen(Hbt,sortby=nothing);

eigHt=eigen(Htt,sortby=abs2);
eigHp=eigen(Hpt,sortby=abs2);
eigHb=eigen(Hbt,sortby=abs2);

eigHt=eigen(Htt);
eigHp=eigen(Hpt);
eigHb=eigen(Hbt);

Float64.(eigHt.vectors*Diagonal(eigHt.values)*eigHt.vectors' - Htt)
eigHt.vectors
kron(eigHb.vectors,eigHp.vectors)-eigHt.vectors
v = kron(eigHb.vectors,eigHp.vectors);
Float64.(v*Diagonal(eigHt.values)*v' - Htt)
e = kron(eigHb.values,eigHp.values);
Float64.(v*Diagonal(e)*v' - Htt)

qAnneal.partialState(Int(0b000),[3,2],[1 2 3 4])

pt = qAnneal.EqStateBig(4,eigHt.vectors);
pp = qAnneal.EqStateBig(2,eigHp.vectors);
pb = qAnneal.EqStateBig(2,eigHb.vectors);
ComplexF64.(kron(pb,pp)-pt)

using LinearAlgebra: RealHermSymComplexHerm, Algorithm, QRIteration, DivideAndConquer 
ComplexF64.(eigen!(A,QRIteration()))
ComplexF64.(eigen!(A,DivideAndConquer()))