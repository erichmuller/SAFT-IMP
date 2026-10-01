#=
Validation and demonstration script for SAFT-VR IMP.

    julia --project=<env with Clapeyron, ForwardDiff, StaticArrays> test/test_SAFTVRIMP.jl

1. Zero multipoles + order 3            ⇒ identical to Clapeyron's SAFTVRMie (pure and mixture)
2. Dipolar LJ (Keesom) fluid            ⇒ identical to SAFTVRMie with ε_m = εF², σ_m = σF^(-1/6)
3. Every Sutherland slot (λ = 6, 8, 10) ⇒ identical to SAFTVRMie with analytically mapped pair parameters
4. Automatic differentiation through σ_eff(T), ϵ_eff(T), d(T): AD vs finite differences, IFT and envelope theorem
5. n-hexane with the second-order model: p(V,T), gradients, saturation curve, critical point
6. Müller–Gelb IMP fluids: critical points, Clapeyron-equation consistency, 1,2-dichloroethane + cyclohexane
=#

using Clapeyron, ForwardDiff, StaticArrays, Test, Printf, LinearAlgebra

const IMP_ROOT = normpath(joinpath(@__DIR__, ".."))
Base.include(Clapeyron, joinpath(IMP_ROOT, "SAFTVRIMP.jl"))
const SAFTVRIMP = Clapeyron.SAFTVRIMP
const KÅ3 = Clapeyron.IMP_D2_TO_KÅ3

#Richardson-extrapolated central difference, O(h⁴)
function fdiff(f, x; h = 1e-3*abs(x))
    D(hh) = (f(x + hh) - f(x - hh))/(2hh)
    return (4*D(h/2) - D(h))/3
end

#Mie(λr,λa) plus an extra attractive term c·(σ/r)^λa (c ≤ 0, in K) is again Mie(λr,λa) with these parameters
function mie_mapping(σ, ε, λr, λa, c)
    C = Clapeyron.Cλ_mie(λa, λr)
    F = 1 - c/(C*ε)
    return σ*F^(-1/(λr - λa)), ε*F^(λr/(λr - λa))
end

imp_user(; Mw, m, σ, ε, λr, λa, μ, Q, α) = (Mw = Mw, segment = m, sigma = σ, epsilon = ε, lambda_r = λr, lambda_a = λa, dipole = μ, quadrupole = Q, polarizability = α)
mie_user(; Mw, m, σ, ε, λr, λa) = (Mw = Mw, segment = m, sigma = σ, epsilon = ε, lambda_r = λr, lambda_a = λa, epsilon_assoc = nothing, bondvol = nothing)

function test_reduction_to_mie()
    println("\n── 1. Zero multipoles, order = 3  ⇒  SAFT-VR Mie ──")
    states = [(1.3e-4, 300.0), (1.0e-3, 450.0), (2.0e-2, 400.0), (1.6e-4, 1200.0)]   #T = 1200 K uses the Gauss–Legendre branch of d(T)
    pure_imp, pure_mie = SAFTVRIMP(["hexane"]; order = 3), SAFTVRMie(["hexane"])
    mix_imp, mix_mie = SAFTVRIMP(["hexane", "decane"]; order = 3), SAFTVRMie(["hexane", "decane"])
    for (imp, mie, z) in ((pure_imp, pure_mie, SA[1.0]), (mix_imp, mix_mie, [0.3, 0.7]))
        for (V, T) in states
            ai, am = Clapeyron.a_res(imp, V, T, z), Clapeyron.a_res(mie, V, T, z)
            pi_, pm = pressure(imp, V, T, z), pressure(mie, V, T, z)
            μi, μm = Clapeyron.VT_chemical_potential_res(imp, V, T, z), Clapeyron.VT_chemical_potential_res(mie, V, T, z)
            @printf("  %-15s V=%.1e T=%6.1f  |Δa_res/a_res|=%.1e  |Δp/p|=%.1e  max|Δμ_res/μ_res|=%.1e\n",
                length(z) == 1 ? "hexane" : "hexane+decane", V, T, abs(ai/am - 1), abs(pi_/pm - 1), maximum(abs.(μi ./ μm .- 1)))
            @test isapprox(ai, am; rtol = 1e-11)
            @test isapprox(pi_, pm; rtol = 1e-11)
            @test all(isapprox.(μi, μm; rtol = 1e-11))
        end
    end
    Tb_i = saturation_temperature(pure_imp, 101325.0)[1]
    Tb_m = saturation_temperature(pure_mie, 101325.0)[1]
    @printf("  normal boiling point: SAFTVRIMP(order=3) %.8f K, SAFTVRMie %.8f K\n", Tb_i, Tb_m)
    @test isapprox(Tb_i, Tb_m; rtol = 1e-9)
    #a_mono + a_chain must add up to a_res
    V, T, z = 1.3e-4, 300.0, [0.3, 0.7]
    @test isapprox(Clapeyron.a_mono(mix_imp, V, T, z) + Clapeyron.a_chain(mix_imp, V, T, z), Clapeyron.a_res(mix_imp, V, T, z); rtol = 1e-13)
end

function test_keesom_mapping()
    println("\n── 2. Dipolar LJ fluid (μ*² = 2)  ⇒  LJ with ε_m = εF², σ_m = σF^(-1/6), F = 1 + μ*⁴/(12T*) ──")
    ε, σ = 150.0, 3.5
    μD = sqrt(2*ε*σ^3/KÅ3)   #μ*² = μ̂²/(εσ³) = 2
    user = imp_user(Mw = [40.0], m = [1.0], σ = [σ], ε = [ε], λr = [12.0], λa = [6.0], μ = [μD], Q = [0.0], α = [0.0])
    imp = SAFTVRIMP(["dipolar LJ"]; order = 3, userlocations = user)
    z = SA[1.0]
    for T in (120.0, 200.0, 400.0, 3000.0)
        σm, εm = mie_mapping(σ, ε, 12.0, 6.0, -KÅ3^2*μD^4/(3*T*σ^6))
        mie = SAFTVRMie(["LJ"]; userlocations = mie_user(Mw = [40.0], m = [1.0], σ = [σm], ε = [εm], λr = [12.0], λa = [6.0]))
        ep = Clapeyron.imp_effective_parameters(imp, T)
        V = 2.5e-5
        ai, am = Clapeyron.a_res(imp, V, T, z), Clapeyron.a_res(mie, V, T, z)
        @printf("  T=%6.0f K  σ_eff/σ_m−1=%.1e  ϵ_eff/ε_m−1=%.1e  d/d_m−1=%.1e  a_res rel.diff=%.1e\n",
            T, ep.σeff[1]*1e10/σm - 1, ep.ϵeff[1]/εm - 1, ep.d[1]/Clapeyron.d(mie, V, T, z)[1] - 1, abs(ai/am - 1))
        @test isapprox(ep.σeff[1]*1e10, σm; rtol = 1e-13)
        @test isapprox(ep.ϵeff[1], εm; rtol = 1e-13)
        @test isapprox(ai, am; rtol = 1e-12)
        @test isapprox(pressure(imp, V, T, z), pressure(mie, V, T, z); rtol = 1e-12)
    end
    #derivatives of the implicit solves against the analytic mapping (first and second order)
    T0 = 180.0
    εm_T(T) = mie_mapping(σ, ε, 12.0, 6.0, -KÅ3^2*μD^4/(3*T*σ^6))[2]
    σm_T(T) = mie_mapping(σ, ε, 12.0, 6.0, -KÅ3^2*μD^4/(3*T*σ^6))[1]*1e-10
    ϵeff_T(T) = Clapeyron.imp_effective_parameters(imp, T).ϵeff[1]
    σeff_T(T) = Clapeyron.imp_effective_parameters(imp, T).σeff[1]
    d1 = ForwardDiff.derivative(ϵeff_T, T0); d1x = ForwardDiff.derivative(εm_T, T0)
    d2 = ForwardDiff.derivative(t -> ForwardDiff.derivative(ϵeff_T, t), T0); d2x = ForwardDiff.derivative(t -> ForwardDiff.derivative(εm_T, t), T0)
    s1 = ForwardDiff.derivative(σeff_T, T0); s1x = ForwardDiff.derivative(σm_T, T0)
    @printf("  dϵ_eff/dT: AD %.12e, analytic %.12e;  d²ϵ_eff/dT²: AD %.12e, analytic %.12e\n", d1, d1x, d2, d2x)
    @test isapprox(d1, d1x; rtol = 1e-12)
    @test isapprox(d2, d2x; rtol = 1e-10)
    @test isapprox(s1, s1x; rtol = 1e-10)
end

function test_all_slots_mapping()
    println("\n── 3. λ = 6 (Keesom + induction), 8 (μQ cross term), 10 (QQ) slots, chains and mixing  ⇒  mapped SAFT-VR Mie ──")
    m = [1.5, 2.0]; μ = [1.8, 0.0]; Q = [0.0, 3.5]; αp = [6.0, 0.0]
    σ = [3.8 3.95; 3.95 4.1]; ε = [250.0 270.0; 270.0 300.0]
    λr = [16.0 19.0; 19.0 22.0]; λa = [6.0 8.0; 8.0 10.0]
    imp = SAFTVRIMP(["A", "B"]; order = 3, userlocations = imp_user(Mw = [40.0, 60.0], m = m, σ = σ, ε = ε, λr = λr, λa = λa, μ = μ, Q = Q, α = αp))
    μ̂2 = KÅ3 .* μ.^2 ./ m; Q̂2 = KÅ3 .* Q.^2 ./ m; α̂ = αp ./ m
    #every pair has exactly one multipolar slot, with exponent equal to its λa
    cpol(T) = [-(2*μ̂2[1]*α̂[1] + μ̂2[1]^2/(3T))/σ[1,1]^6   -(μ̂2[1]*Q̂2[2] + Q̂2[1]*μ̂2[2])/(2T*σ[1,2]^8);
               -(μ̂2[1]*Q̂2[2] + Q̂2[1]*μ̂2[2])/(2T*σ[1,2]^8)   -7*Q̂2[2]^2/(5T*σ[2,2]^10)]
    for i in 1:2, j in 1:2
        st = Clapeyron.sutherland_terms(imp, 300.0, i, j)
        @test count(==(λa[i,j]), st.λ) == 2   #dispersion + one multipolar term, same exponent
    end
    z = [0.35, 0.65]
    for T in (250.0, 400.0, 1500.0), V in (1.1e-4, 1.0e-3)
        c = cpol(T)
        σm = similar(σ); εm = similar(ε)
        for i in 1:2, j in 1:2
            σm[i,j], εm[i,j] = mie_mapping(σ[i,j], ε[i,j], λr[i,j], λa[i,j], c[i,j])
        end
        mie = SAFTVRMie(["A", "B"]; userlocations = mie_user(Mw = [40.0, 60.0], m = m, σ = σm, ε = εm, λr = λr, λa = λa))
        ai, am = Clapeyron.a_res(imp, V, T, z), Clapeyron.a_res(mie, V, T, z)
        pi_, pm = pressure(imp, V, T, z), pressure(mie, V, T, z)
        μi, μm = Clapeyron.VT_chemical_potential_res(imp, V, T, z), Clapeyron.VT_chemical_potential_res(mie, V, T, z)
        @printf("  T=%6.0f V=%.1e  a_res %.1e   p %.1e   μ_res %.1e   (relative differences)\n", T, V, abs(ai/am - 1), abs(pi_/pm - 1), maximum(abs.(μi ./ μm .- 1)))
        @test isapprox(ai, am; rtol = 1e-11)
        @test isapprox(pi_, pm; rtol = 1e-11)
        @test all(isapprox.(μi, μm; rtol = 1e-11))
    end
end

function test_automatic_differentiation()
    println("\n── 4. AD through the T-dependent potential (order = 2, polar chain + decane) ──")
    #hypothetical polar chain (hexane parameters + μ, Q, α) to exercise all five Sutherland terms at once
    user = imp_user(Mw = [86.177, 142.285], m = [2.1097, 2.9976], σ = [4.423, 4.589], ε = [354.38, 400.79],
                    λr = [17.203, 18.885], λa = [6.0, 6.0], μ = [2.0, 0.0], Q = [4.0, 0.0], α = [11.9, 19.1])
    model = SAFTVRIMP(["polar chain", "decane"]; userlocations = user)
    st = Clapeyron.sutherland_terms(model, 350.0, 1, 1)
    println("  like-pair Sutherland set at 350 K (λₖ, εₖ/K): ", collect(zip(st.λ, round.(st.ε, sigdigits = 6))))
    for (V, T) in ((1.6e-4, 350.0), (1.0e-2, 450.0))
        z = [0.4, 0.6]
        dpdT = ForwardDiff.derivative(t -> pressure(model, V, t, z), T)
        dpdV = ForwardDiff.derivative(v -> pressure(model, v, T, z), V)
        dpdz = ForwardDiff.gradient(x -> pressure(model, V, T, x), z)
        g = ForwardDiff.gradient(x -> pressure(model, x[1], x[2], x[3:4]), [V, T, z...])
        fd_T = fdiff(t -> pressure(model, V, t, z), T)
        fd_V = fdiff(v -> pressure(model, v, T, z), V)
        fd_z1 = fdiff(x -> pressure(model, V, T, [x, z[2]]), z[1])
        #second temperature derivative of A_res (enters Cv, Cp, speed of sound)
        d2A = ForwardDiff.derivative(t -> ForwardDiff.derivative(tt -> Clapeyron.eos_res(model, V, tt, z), t), T)
        fd_d2A = fdiff(t -> ForwardDiff.derivative(tt -> Clapeyron.eos_res(model, V, tt, z), t), T)
        #energy route: Cv = (∂U/∂T)_V must be consistent with U = −T²∂(A/T)/∂T
        Cv = Clapeyron.VT_isochoric_heat_capacity(model, V, T, z)
        fd_Cv = fdiff(t -> Clapeyron.VT_internal_energy(model, V, t, z), T)
        @printf("  V=%.1e T=%.0f  ∂p/∂T AD %.10e FD %.10e | ∂p/∂V rel %.1e | ∂p/∂z₁ rel %.1e | ∂²A_res/∂T² rel %.1e | Cv vs ∂U/∂T rel %.1e\n",
            V, T, dpdT, fd_T, abs(dpdV/fd_V - 1), abs(dpdz[1]/fd_z1 - 1), abs(d2A/fd_d2A - 1), abs(Cv/fd_Cv - 1))
        @test isapprox(dpdT, fd_T; rtol = 1e-7)
        @test isapprox(dpdV, fd_V; rtol = 1e-7)
        @test isapprox(dpdz[1], fd_z1; rtol = 1e-7)
        @test isapprox(d2A, fd_d2A; rtol = 1e-6)
        @test isapprox(Cv, fd_Cv; rtol = 1e-6)
        @test isapprox(g, [dpdV, dpdT, dpdz...]; rtol = 1e-12)
        #p is intensive: V ∂p/∂V + Σ zᵢ ∂p/∂zᵢ = 0
        @test abs(V*dpdV + dot(z, dpdz)) <= 1e-8*abs(V*dpdV)
    end
    #implicit function theorem for σ_eff(T) and envelope theorem for ϵ_eff(T) on the polar like pair
    T0 = 320.0
    λ, c, act = Clapeyron.imp_sutherland(model, T0, 1, 1)
    ∂c∂T = ForwardDiff.derivative(t -> collect(Clapeyron.imp_sutherland(model, t, 1, 1)[2]), T0)
    ep = Clapeyron.imp_effective_parameters(model, T0)
    xe = ep.σeff[1,1]/model.params.sigma.values[1,1]
    ∂u∂T(x) = sum(∂c∂T[k]*x^(-λ[k]) for k in 1:5)
    ∂u∂x(x) = -sum(λ[k]*c[k]*x^(-λ[k] - 1) for k in 1:5)
    dxe_ift = -∂u∂T(xe)/∂u∂x(xe)
    dxe_ad = ForwardDiff.derivative(t -> Clapeyron.imp_effective_parameters(model, t).σeff[1,1], T0)/model.params.sigma.values[1,1]
    xmin = exp(Clapeyron.imp_logroot(λ, (λ[1]*c[1], -λ[2]*c[2], -λ[3]*c[3], -λ[4]*c[4], -λ[5]*c[5]), act, 0.0))
    dϵ_env = -∂u∂T(xmin)
    dϵ_ad = ForwardDiff.derivative(t -> Clapeyron.imp_effective_parameters(model, t).ϵeff[1,1], T0)
    @printf("  dx_eff/dT: AD %.12e IFT %.12e | dϵ_eff/dT: AD %.12e envelope %.12e\n", dxe_ad, dxe_ift, dϵ_ad, dϵ_env)
    @test isapprox(dxe_ad, dxe_ift; rtol = 1e-10)
    @test isapprox(dϵ_ad, dϵ_env; rtol = 1e-10)
end

function demo_hexane()
    println("\n── 5. n-hexane, SAFT-VR IMP with the second-order Barker–Henderson monomer (default) ──")
    hex = SAFTVRIMP(["hexane"])                 #order = 2
    hex3 = SAFTVRIMP(["hexane"]; order = 3)     #≡ SAFT-VR Mie (Lafitte et al. 2013 parameters)
    V, T = 1.4e-4, 350.0
    p = pressure(hex, V, T)
    grad = ForwardDiff.gradient(x -> pressure(hex, x[1], x[2], SA[x[3]]), [V, T, 1.0])
    @printf("  p(V = %.2e m³/mol, T = %.1f K) = %.6e Pa;  ∇p = [∂p/∂V, ∂p/∂T, ∂p/∂n] = [%.6e, %.6e, %.6e]\n", V, T, p, grad...)
    @test abs(V*grad[1] + grad[3]) <= 1e-8*abs(V*grad[1])
    println("   T/K    p_sat/kPa (o2)  p_sat/kPa (o3≡VR-Mie)   ρ_l/(mol/L) o2   o3      ρ_v/(mol/L) o2   o3")
    for T in (300.0, 350.0, 400.0, 450.0, 500.0)
        ps2, vl2, vv2 = saturation_pressure(hex, T)
        ps3, vl3, vv3 = saturation_pressure(hex3, T)
        @printf("  %5.1f   %12.3f   %12.3f          %8.4f  %8.4f       %8.5f  %8.5f\n", T, ps2/1e3, ps3/1e3, 1e-3/vl2, 1e-3/vl3, 1e-3/vv2, 1e-3/vv3)
        @test isfinite(ps2) && vl2 < vv2
    end
    Tb2 = saturation_temperature(hex, 101325.0)[1]; Tb3 = saturation_temperature(hex3, 101325.0)[1]
    Tc2, pc2, _ = crit_pure(hex); Tc3, pc3, _ = crit_pure(hex3)
    @printf("  T_b(1 atm): o2 %.2f K, o3 %.2f K (exp. ≈ 341.9 K)\n", Tb2, Tb3)
    @printf("  T_c, p_c:   o2 %.2f K, %.3f MPa;  o3 %.2f K, %.3f MPa (exp. ≈ 507.8 K, 3.03 MPa)\n", Tc2, pc2/1e6, Tc3, pc3/1e6)
    @test Tc2 > Tc3   #truncating before a₃ raises T_c
end

function demo_muller_gelb()
    println("\n── 6. Müller–Gelb (2003) IMP fluids (their MD parameters, not refitted for this EoS) ──")
    Tc_exp = Dict("benzene_MG2003" => 562.0, "carbon dioxide_MG2003" => 304.1, "1,2-dichloroethane_MG2003" => 561.6,
                  "cyclohexane_MG2003" => 553.6, "n-octane_MG2003" => 568.7)
    println("  fluid                        T_c/K (o2)   T_c/K (o3)   T_c/K (o2, moments off)   exp.")
    for name in ("benzene_MG2003", "carbon dioxide_MG2003", "1,2-dichloroethane_MG2003", "cyclohexane_MG2003", "n-octane_MG2003")
        m2 = SAFTVRIMP([name]); m3 = SAFTVRIMP([name]; order = 3)
        m0 = SAFTVRIMP([name]); m0.params.dipole.values .= 0; m0.params.quadrupole.values .= 0
        Tc2, Tc3, Tc0 = crit_pure(m2)[1], crit_pure(m3)[1], crit_pure(m0)[1]
        @printf("  %-27s  %9.2f    %9.2f    %9.2f                 %6.1f\n", name, Tc2, Tc3, Tc0, Tc_exp[name])
        polar = !iszero(m2.params.dipole.values[1]) || !iszero(m2.params.quadrupole.values[1])
        polar ? (@test Tc2 > Tc0 + 1) : (@test isapprox(Tc2, Tc0; rtol = 1e-8))
    end
    #Clapeyron equation dp_sat/dT = Δh_vap/(TΔv): only holds if the energy contains ∂u(r;T)/∂T
    bz = SAFTVRIMP(["benzene_MG2003"])
    T = 450.0
    psat, vl, vv = saturation_pressure(bz, T)
    Δh = Clapeyron.VT_enthalpy(bz, vv, T) - Clapeyron.VT_enthalpy(bz, vl, T)
    dpdT = fdiff(t -> saturation_pressure(bz, t)[1], T; h = 0.05)
    @printf("  benzene_MG2003 at 450 K: dp_sat/dT (FD) = %.6f Pa/K;  Δh_vap/(TΔv) = %.6f Pa/K\n", dpdT, Δh/(T*(vv - vl)))
    @test isapprox(dpdT, Δh/(T*(vv - vl)); rtol = 1e-6)
    #predicted binary (pure-component parameters only), cf. Müller & Gelb Fig. 8
    mix = SAFTVRIMP(["1,2-dichloroethane_MG2003", "cyclohexane_MG2003"])
    println("  1,2-dichloroethane(1) + cyclohexane(2), p = 1 atm:")
    for x1 in (0.1, 0.3, 0.5, 0.7, 0.9)
        Tb, _, _, y = bubble_temperature(mix, 101325.0, [x1, 1 - x1])
        @printf("     x₁ = %.1f   T_bub = %.2f K   y₁ = %.4f\n", x1, Tb, y[1])
    end
    Taz, _, _, xaz = azeotrope_temperature(mix, 101325.0)
    @printf("  predicted azeotrope: T = %.2f K at x₁ = %.3f (pure boiling points, o2: %.2f K and %.2f K)\n", Taz, xaz[1],
        saturation_temperature(SAFTVRIMP(["1,2-dichloroethane_MG2003"]), 101325.0)[1], saturation_temperature(SAFTVRIMP(["cyclohexane_MG2003"]), 101325.0)[1])
    @test 0 < xaz[1] < 1
    #dipolar LJ fluid of Müller & Gelb Fig. 3 (μ*² = 2)
    ε, σ = 150.0, 3.5
    μD = sqrt(2*ε*σ^3/KÅ3)
    for order in (2, 3)
        lj = SAFTVRIMP(["dipolar LJ"]; order, userlocations = imp_user(Mw = [40.0], m = [1.0], σ = [σ], ε = [ε], λr = [12.0], λa = [6.0], μ = [μD], Q = [0.0], α = [0.0]))
        lj0 = SAFTVRIMP(["LJ"]; order, userlocations = imp_user(Mw = [40.0], m = [1.0], σ = [σ], ε = [ε], λr = [12.0], λa = [6.0], μ = [0.0], Q = [0.0], α = [0.0]))
        Tc, _, Vc = crit_pure(lj); Tc0, _, Vc0 = crit_pure(lj0)
        ρred(v) = Clapeyron.N_A*(σ*1e-10)^3/v
        @printf("  order %d: LJ T_c* = %.4f, ρ_c* = %.4f;  μ*² = 2: T_c* = %.4f, ρ_c* = %.4f\n", order, Tc0/ε, ρred(Vc0), Tc/ε, ρred(Vc))
    end
end

@testset "SAFT-VR IMP" begin
    @testset "reduction to SAFT-VR Mie" test_reduction_to_mie()
    @testset "Keesom mapping" test_keesom_mapping()
    @testset "all Sutherland slots" test_all_slots_mapping()
    @testset "automatic differentiation" test_automatic_differentiation()
    @testset "n-hexane (order 2)" demo_hexane()
    @testset "Müller–Gelb fluids" demo_muller_gelb()
end
