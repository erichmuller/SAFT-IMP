#=
SAFT-VR IMP
===========

SAFT-VR equation of state for chains of tangent segments that interact through the
Isotropic Multipolar Potential (IMP) of Müller & Gelb (2003), written as a finite,
temperature-dependent sum of Sutherland terms and treated with the SAFT-VR Mie /
SAFT-VR Sum perturbation machinery (Lafitte et al. 2013; Jervell et al. 2026).

This file follows the conventions of the Clapeyron source tree and uses Clapeyron
internals unqualified. It can be used in two ways:

1. In-tree: copy to `src/models/SAFT/SAFTVRIMP/SAFTVRIMP.jl`, add
   `include("models/SAFT/SAFTVRIMP/SAFTVRIMP.jl")` to `src/Clapeyron.jl` after the
   SAFT-VR Mie include, and copy `database/SAFTVRIMP_like.csv` to
   `database/SAFT/SAFTVRIMP/`.
2. Out-of-tree (no package modification):

       using Clapeyron
       Base.include(Clapeyron, "path/to/SAFTVRIMP.jl")
       const SAFTVRIMP = Clapeyron.SAFTVRIMP

   The `database` folder next to this file is then used as the default parameter location.

Pair potential between segments of components i and j (all energies in K, i.e. u/k_B):

    u_ij(r;T) = Σₖ cₖ(T) (σ_ij/r)^λₖ ,   k = 1…5

    k   λₖ     cₖ(T)                                        physics
    1   λr     + C_ij ε_ij                                   soft (Mie) repulsion
    2   λa     − C_ij ε_ij                                   dispersion
    3   6      −[μ̂ᵢ²α̂ⱼ + α̂ᵢμ̂ⱼ² + μ̂ᵢ²μ̂ⱼ²/(3T)]/σ_ij⁶        Debye induction + Keesom
    4   8      −(μ̂ᵢ²Q̂ⱼ² + Q̂ᵢ²μ̂ⱼ²)/(2Tσ_ij⁸)                 dipole–quadrupole
    5   10     −7Q̂ᵢ²Q̂ⱼ²/(5Tσ_ij¹⁰)                          quadrupole–quadrupole

with segment-level moments μ̂ᵢ² = μᵢ²/(4πε₀mᵢ), Q̂ᵢ² = Qᵢ²/(4πε₀mᵢ), α̂ᵢ = αᵢ/mᵢ.
In the brief's notation φ(r) = −Σₖ εₖ(σ/r)^λₖ one has εₖ = −cₖ.
=#

struct SAFTVRIMPParam{T} <: ParametricEoSParam{T}
    Mw::SingleParam{T}
    segment::SingleParam{T}
    sigma::PairParam{T}
    lambda_a::PairParam{T}
    lambda_r::PairParam{T}
    epsilon::PairParam{T}
    dipole::SingleParam{T}
    quadrupole::SingleParam{T}
    polarizability::SingleParam{T}
end

function SAFTVRIMPParam(Mw,segment,sigma,lambda_a,lambda_r,epsilon,dipole,quadrupole,polarizability)
    return build_parametric_param(SAFTVRIMPParam,Mw,segment,sigma,lambda_a,lambda_r,epsilon,dipole,quadrupole,polarizability)
end

"""
    SAFTVRIMPOptions(order)

Non-splittable option holder for [`SAFTVRIMP`](@ref). `order = 2` truncates the
Barker–Henderson expansion after the second-order (fluctuation) term; `order = 3`
adds the empirical third-order correlation of Lafitte et al. (2013), evaluated with
the effective well depth and van der Waals energy of the IMP pair potential.
"""
struct SAFTVRIMPOptions
    order::Int
end

is_splittable(::SAFTVRIMPOptions) = false

abstract type SAFTVRIMPModel <: SAFTModel end

struct SAFTVRIMP{I<:IdealModel,T} <: SAFTVRIMPModel
    components::Vector{String}
    params::SAFTVRIMPParam{T}
    idealmodel::I
    imp_options::SAFTVRIMPOptions
    references::Vector{String}
end

"""
    SAFTVRIMPModel <: SAFTModel

    SAFTVRIMP(components;
    idealmodel = BasicIdeal,
    userlocations = String[],
    ideal_userlocations = String[],
    reference_state = nothing,
    order = 2,
    verbose = false)

## Input parameters
- `Mw`: Single Parameter (`Float64`) - Molecular Weight `[g·mol⁻¹]`
- `segment`: Single Parameter (`Float64`) - Number of segments (no units)
- `sigma`: Single Parameter (`Float64`) - Segment size parameter `[Å]`
- `epsilon`: Single Parameter (`Float64`) - Reduced dispersion energy `[K]`
- `lambda_a`: Pair Parameter (`Float64`) - Attractive (dispersion) exponent (no units)
- `lambda_r`: Pair Parameter (`Float64`) - Repulsive exponent (no units)
- `dipole`: Single Parameter (`Float64`) (optional) - Molecular dipole moment `[D]`
- `quadrupole`: Single Parameter (`Float64`) (optional) - Molecular (Buckingham) quadrupole moment `[D·Å]`
- `polarizability`: Single Parameter (`Float64`) (optional) - Molecular volume polarizability `[Å³]`
- `k`: Pair Parameter (`Float64`) (optional) - Binary interaction parameter (no units)

## Model Parameters
- `Mw`: Single Parameter (`Float64`) - Molecular Weight `[g·mol⁻¹]`
- `segment`: Single Parameter (`Float64`) - Number of segments (no units)
- `sigma`: Pair Parameter (`Float64`) - Mixed segment size parameter `[m]`
- `lambda_a`: Pair Parameter (`Float64`) - Attractive exponent (no units)
- `lambda_r`: Pair Parameter (`Float64`) - Repulsive exponent (no units)
- `epsilon`: Pair Parameter (`Float64`) - Mixed reduced dispersion energy `[K]`
- `dipole`, `quadrupole`, `polarizability`: Single Parameters, as above

## Input models
- `idealmodel`: Ideal Model

## Description

SAFT-VR for chains of tangent segments interacting through the isotropic multipolar
potential (IMP) of Müller and Gelb, represented as a temperature-dependent sum of
Sutherland potentials. The monomer term is a Barker–Henderson expansion around a
hard-sphere reference with temperature-dependent diameter, truncated at second order
(`order = 2`, default) or supplemented with the third-order correlation of Lafitte
et al. (`order = 3`). With all multipole moments set to zero and `order = 3` the model
reproduces `SAFTVRMie` exactly. Moments are molecular and are distributed over the
segments as μ̂² = μ²/m, Q̂² = Q²/m, α̂ = α/m. Unit conversions from SI: μ[D] = μ[C·m]/3.33564e-30,
Q[D·Å] = Q[C·m²]/3.33564e-40.

## References
1. Müller, E. A., & Gelb, L. D. (2003). Molecular modeling of fluid-phase equilibria using an isotropic multipolar potential. Industrial & Engineering Chemistry Research, 42(17), 4123–4131. [doi:10.1021/ie030033y](https://doi.org/10.1021/ie030033y)
2. Lafitte, T., Apostolakou, A., Avendaño, C., Galindo, A., Adjiman, C. S., Müller, E. A., & Jackson, G. (2013). Accurate statistical associating fluid theory for chain molecules formed from Mie segments. The Journal of Chemical Physics, 139(15), 154504. [doi:10.1063/1.4819786](https://doi.org/10.1063/1.4819786)
3. Jervell, V. G., Maltby, T. W., Aasen, A., Hammer, M., & Wilhelmsen, Ø. (2026). SAFT-VR Sum: Equation of state and transport properties from ab initio-derived Sutherland sum potentials. The Journal of Chemical Physics, 164, 114101. [doi:10.1063/5.0317322](https://doi.org/10.1063/5.0317322)
"""
SAFTVRIMP

export SAFTVRIMP

const SAFTVRIMP_LOCAL_DATABASE = normpath(joinpath(@__DIR__, "database"))

function default_locations(::Type{SAFTVRIMP})
    if isdir(SAFTVRIMP_LOCAL_DATABASE)
        return [SAFTVRIMP_LOCAL_DATABASE, "properties/molarmass.csv"]
    else
        return ["SAFT/SAFTVRIMP", "properties/molarmass.csv"]
    end
end

default_references(::Type{SAFTVRIMP}) = ["10.1021/ie030033y", "10.1063/1.4819786", "10.1063/5.0317322"]
default_ignore_missing_singleparams(::Type{SAFTVRIMP}) = ["dipole", "quadrupole", "polarizability"]

function SAFTVRIMP(components;
    idealmodel = BasicIdeal,
    userlocations = String[],
    ideal_userlocations = String[],
    reference_state = nothing,
    order::Integer = 2,
    verbose = false)

    order in (2,3) || throw(ArgumentError("SAFTVRIMP: `order` must be 2 (Barker–Henderson, default) or 3 (adds the Lafitte et al. third-order correlation), got $order."))
    userlocations = normalize_userlocations(userlocations)
    ideal_userlocations = normalize_userlocations(ideal_userlocations)
    _components = format_components(components)
    options = default_getparams_arguments(SAFTVRIMP,userlocations,verbose)
    params_in = getparams(_components,default_locations(SAFTVRIMP),options)
    params_out = transform_params(SAFTVRIMP,params_in,_components)
    pkgparam = build_eosparam(SAFTVRIMPParam,params_out)
    init_idealmodel = init_model(idealmodel,_components,ideal_userlocations,verbose)
    model = SAFTVRIMP(_components,pkgparam,init_idealmodel,SAFTVRIMPOptions(Int(order)),default_references(SAFTVRIMP))
    imp_check_params(model)
    set_reference_state!(model,reference_state;verbose)
    return model
end

function transform_params(::Type{SAFTVRIMP},params,components)
    sigma = params["sigma"]
    sigma.values .*= 1E-10
    sigma = sigma_LorentzBerthelot(sigma)
    epsilon = epsilon_HudsenMcCoubreysqrt(params["epsilon"],sigma)
    k = get(params,"k",nothing)
    if k !== nothing
        epsilon .= epsilon .* (1 .- k)
    end
    params["sigma"] = sigma
    params["epsilon"] = epsilon
    params["lambda_a"] = lambda_LorentzBerthelot(params["lambda_a"])
    params["lambda_r"] = lambda_LorentzBerthelot(params["lambda_r"])
    #multipoles need no combining rule: cross terms are built from the pure moments.
    for name in ("dipole","quadrupole","polarizability")
        get!(params,name) do
            SingleParam(name,components)
        end
    end
    return params
end

function recombine_impl!(model::SAFTVRIMPModel)
    sigma = sigma_LorentzBerthelot!(model.params.sigma)
    epsilon_HudsenMcCoubreysqrt!(model.params.epsilon,sigma)
    lambda_LorentzBerthelot!(model.params.lambda_a)
    lambda_LorentzBerthelot!(model.params.lambda_r)
    imp_check_params(model)
    return model
end

function show_info(io,model::SAFTVRIMP)
    println(io)
    if model.imp_options.order == 2
        print(io,"Perturbation order: 2 (Barker–Henderson)")
    else
        print(io,"Perturbation order: 3 (Barker–Henderson + Lafitte a₃)")
    end
end

x0_volume_liquid(model::SAFTVRIMPModel,T,z) = lb_volume(model,T,z)*1.5

#=
IMP pair potential as a Sutherland sum
=#

const IMP_NTERMS = 5
#(1 D)²/(4πε₀k_B) in K·Å³ (identically, (1 D·Å)²/(4πε₀k_B) in K·Å⁵), from D²/(4πε₀) = 1e-49 J·m³
const IMP_D2_TO_KÅ3 = 1e-49/k_B*1e30

"""
    imp_polar_constants(model::SAFTVRIMPModel, i, j)

Temperature-independent constants of the multipolar Sutherland terms of the pair `(i,j)`:
`A₆` [K·Å⁶] (Debye induction), `B₆` [K²·Å⁶] (Keesom), `B₈` [K²·Å⁸] (dipole–quadrupole),
`B₁₀` [K²·Å¹⁰] (quadrupole–quadrupole), such that c₃ = −(A₆ + B₆/T)/σ⁶, c₄ = −B₈/(Tσ⁸), c₅ = −B₁₀/(Tσ¹⁰).
"""
function imp_polar_constants(model::SAFTVRIMPModel,i,j)
    p = model.params
    m = p.segment.values
    μ = p.dipole.values
    Q = p.quadrupole.values
    αp = p.polarizability.values
    μ2i = IMP_D2_TO_KÅ3*μ[i]*μ[i]/m[i]
    μ2j = IMP_D2_TO_KÅ3*μ[j]*μ[j]/m[j]
    Q2i = IMP_D2_TO_KÅ3*Q[i]*Q[i]/m[i]
    Q2j = IMP_D2_TO_KÅ3*Q[j]*Q[j]/m[j]
    αi = αp[i]/m[i]
    αj = αp[j]/m[j]
    A6 = μ2i*αj + αi*μ2j
    B6 = μ2i*μ2j/3
    B8 = (μ2i*Q2j + Q2i*μ2j)/2
    B10 = 7*Q2i*Q2j/5
    return A6,B6,B8,B10
end

"""
    imp_sutherland(model::SAFTVRIMPModel, T, i, j)

Returns `(λ, c, active)` for the pair `(i,j)` at temperature `T`: exponents `λ::NTuple{5}`,
coefficients `c::NTuple{5}` [K] of u_ij(r;T)/k_B = Σₖ cₖ (σ_ij/r)^λₖ, and a static mask of
the terms that are identically non-zero for this parameter set (decided on parameters only,
never on dual values, so it is safe under automatic differentiation).
"""
function imp_sutherland(model::SAFTVRIMPModel,T,i,j)
    p = model.params
    σÅ = p.sigma.values[i,j]*1e10
    ϵ = p.epsilon.values[i,j]
    λr = p.lambda_r.values[i,j]
    λa = p.lambda_a.values[i,j]
    Cϵ = Cλ_mie(λa,λr)*ϵ
    A6,B6,B8,B10 = imp_polar_constants(model,i,j)
    Tinv = 1/T
    σ6 = σÅ^6
    c3 = -(A6 + B6*Tinv)/σ6
    c4 = -B8*Tinv/(σ6*σÅ*σÅ)
    c5 = -B10*Tinv/(σ6*σÅ^4)
    c = promote(Cϵ,-Cϵ,c3,c4,c5)
    λ = promote(λr,λa,6.0,8.0,10.0)
    active = (true,true,!(iszero(A6) && iszero(B6)),!iszero(B8),!iszero(B10))
    return λ,c,active
end

"""
    sutherland_terms(model::SAFTVRIMPModel, T, i = 1, j = i)

Sutherland representation of the IMP pair potential in the sign convention
φ(r) = −Σₖ εₖ (σ_ij/r)^λₖ (εₖ in K; the soft repulsion appears with ε₁ < 0).
Only terms that are non-zero for the current parameters are returned.
"""
function sutherland_terms(model::SAFTVRIMPModel,T,i::Integer = 1,j::Integer = i)
    λ,c,act = imp_sutherland(model,T,i,j)
    idx = [k for k in 1:IMP_NTERMS if act[k]]
    return (λ = [λ[k] for k in idx], ε = [-c[k] for k in idx])
end

function imp_check_params(model::SAFTVRIMPModel)
    n = length(model)
    for i in 1:n, j in 1:i
        λ,c,act = imp_sutherland(model,1.0,i,j)
        λatt = maximum(λ[k] for k in 2:IMP_NTERMS if act[k])
        if !(λ[1] > λatt)
            throw(DomainError(λ[1],"SAFTVRIMP: pair ($(model.components[i]), $(model.components[j])) has λr = $(λ[1]) ≤ $(λatt), the largest active attractive exponent. The IMP would not be repulsive at short range."))
        end
        if !(λ[2] > 4)
            throw(DomainError(λ[2],"SAFTVRIMP: pair ($(model.components[i]), $(model.components[j])) has λa = $(λ[2]); λa > 4 is required by the SAFT-VR B-integrals."))
        end
    end
    return nothing
end

#=
Effective range and well depth.
For u(x) = Σₖ cₖ x^(−λₖ) with c₁ > 0, cₖ ≤ 0 (k ≥ 2) and λ₁ > λₖ, both u(x) = 0 and u′(x) = 0 read

    F(s) = log(w₁) − λ₁ s − log Σ_{k≥2} wₖ e^(−λₖ s) = 0 ,   s = log x,

with wₖ = |cₖ| (zero) or wₖ = λₖ|cₖ| (minimum). F′(s) = ⟨λ⟩_w − λ₁ < 0 and F″(s) = −Var_w(λ) ≤ 0:
F is strictly decreasing and concave, so Newton converges globally. Starting from the Mie
solution (s = 0, resp. s = log(λr/λa)/(λr−λa)) gives F ≤ 0, i.e. monotone convergence.
=#

@inline function imp_logroot_step(λ,w,active,s)
    S = zero(s*w[2])
    ∂S = S
    for k in 2:length(λ)
        active[k] || continue
        tk = w[k]*exp(-λ[k]*s)
        S += tk
        ∂S += λ[k]*tk
    end
    F = log(w[1]) - λ[1]*s - log(S)
    dF = ∂S/S - λ[1]
    return F/dF
end

function imp_logroot(λ,w,active,s0)
    s = s0*one(w[1])
    for _ in 1:100
        Δs = imp_logroot_step(λ,w,active,s)
        s -= Δs
        abs(Δs) < 1e-12 && break
    end
    #Newton steps taken at the converged primal root carry the dual parts to the implicit-function
    #derivative (one step per derivative order), so nested ForwardDiff duals are exact as well.
    for _ in 1:3
        s -= imp_logroot_step(λ,w,active,s)
    end
    return s
end

"""
    imp_effective(λ, c, active)

Returns `(x_eff, ϵ_eff, α)`: x_eff = σ_eff/σ with u(σ_eff) = 0, the effective well depth
ϵ_eff = −min u [K] and the dimensionless van der Waals energy
α = −(ϵ_eff σ_eff³)⁻¹ ∫_σeff^∞ u r² dr = −ϵ_eff⁻¹ Σₖ cₖ x_eff^(−λₖ)/(λₖ − 3).
"""
function imp_effective(λ,c,active)
    w0 = (c[1],-c[2],-c[3],-c[4],-c[5])
    s_eff = imp_logroot(λ,w0,active,0.0)
    wmin = (λ[1]*c[1],-λ[2]*c[2],-λ[3]*c[3],-λ[4]*c[4],-λ[5]*c[5])
    s_min = imp_logroot(λ,wmin,active,log(λ[1]/λ[2])/(λ[1] - λ[2]))
    umin = zero(s_min*c[1])
    vdw = zero(s_eff*c[1])
    for k in 1:IMP_NTERMS
        active[k] || continue
        umin += c[k]*exp(-λ[k]*s_min)
        vdw += c[k]*exp(-λ[k]*s_eff)/(λ[k] - 3)
    end
    ϵeff = -umin
    return exp(s_eff),ϵeff,-vdw/ϵeff
end

#=
Barker–Henderson diameter  d = ∫₀^σeff [1 − exp(−u/T)] dr = σ[x_eff − ∫₀^x_eff exp(−u/T) dx].
Same strategy as `d_vrmie`, generalised to the Sutherland sum: θ = c₁/T > 1 uses 10-point
Gauss–Laguerre after y = x^(−λ₁); otherwise 10-point Gauss–Legendre above the point where
exp(−u/T) < eps (Aasen's cut). For a pure Mie pair both branches reduce to `d_vrmie`.
=#

function d_imp_cut(θ,λ,c,active)
    EPS = eps(typeof(θ))
    K = log(-log(EPS)/θ)
    λ1 = λ[1]
    c1 = c[1]
    j0 = exp(-K/λ1)
    function fdfd2f(r)
        r⁻¹ = 1/r
        lnr = log(r)
        u_r = zero(lnr*c1)
        du_r = u_r
        d2u_r = u_r
        for k in 1:length(λ)
            active[k] || continue
            rλ = (c[k]/c1)*exp(-lnr*λ[k])
            du = rλ*r⁻¹*(-λ[k])
            u_r += rλ
            du_r += du
            d2u_r += du*r⁻¹*(-λ[k] - 1)
        end
        f = exp(-u_r*θ)
        df = -θ*f*du_r
        d2f = df*df - θ*d2u_r*f
        return f,f/df,df/d2f
    end
    j = j0
    for i in 1:5
        fi,f1,f2 = fdfd2f(j)
        dj = f1/(1 - 0.5*f1/f2)
        j = j - dj
        fi < eps(eltype(fi)) && break
    end
    return j
end

function d_imp(T,λ,c,active,σ,xeff)
    θ = c[1]/T
    λ1 = λ[1]
    λ1inv = 1/λ1
    if θ > 1
        yeff = xeff^(-λ1)
        function f_laguerre(y)
            lny = log(y)
            ∑att = zero(lny*c[2])
            for k in 2:length(λ)
                active[k] || continue
                ∑att += c[k]*exp(lny*λ[k]*λ1inv)
            end
            return exp(-λ1inv*lny)*exp(-∑att/T)*λ1inv/y
        end
        ∑fi = Solvers.laguerre10(f_laguerre,θ,yeff)
    else
        j = d_imp_cut(θ,λ,c,active)
        function f_legendre(x)
            lnx = log(x)
            u = zero(lnx*c[1])
            for k in 1:length(λ)
                active[k] || continue
                u += c[k]*exp(-λ[k]*lnx)
            end
            return exp(-u/T)
        end
        ∑fi = Solvers.integral10(f_legendre,j,xeff)
    end
    return σ*(xeff - ∑fi)
end

"""
    imp_pairdata(model::SAFTVRIMPModel, V, T, z)

All temperature-dependent (volume- and composition-independent) pair quantities:
Sutherland exponents/coefficients, σ_eff [m], σ_eff/σ, ϵ_eff [K], α, and the like-segment
Barker–Henderson diameters d [m].
"""
function imp_pairdata(model::SAFTVRIMPModel,V,T,z)
    n = length(model)
    p = model.params
    _σ = p.sigma.values
    _ϵ = p.epsilon.values
    _λr = p.lambda_r.values
    _λa = p.lambda_a.values
    TT = Base.promote_eltype(model,T)
    TL = eltype(model)
    λmat = Matrix{NTuple{IMP_NTERMS,TL}}(undef,n,n)
    cmat = Matrix{NTuple{IMP_NTERMS,TT}}(undef,n,n)
    actmat = Matrix{NTuple{IMP_NTERMS,Bool}}(undef,n,n)
    xeff = Matrix{TT}(undef,n,n)
    σeff = Matrix{TT}(undef,n,n)
    ϵeff = Matrix{TT}(undef,n,n)
    α = Matrix{TT}(undef,n,n)
    _d = Vector{TT}(undef,n)
    for i in 1:n
        for j in 1:i
            λ,c,act = imp_sutherland(model,T,i,j)
            if act[3] || act[4] || act[5]
                xe,ϵe,αe = imp_effective(λ,c,act)
            else #pure Mie pair: closed forms, identical to SAFT-VR Mie
                λr,λa = _λr[i,j],_λa[i,j]
                xe = one(TT)
                ϵe = _ϵ[i,j]*one(TT)
                αe = Cλ_mie(λa,λr)*(1/(λa - 3) - 1/(λr - 3))*one(TT)
            end
            λmat[i,j] = λ; λmat[j,i] = λ
            cmat[i,j] = c; cmat[j,i] = c
            actmat[i,j] = act; actmat[j,i] = act
            xeff[i,j] = xe; xeff[j,i] = xe
            σeff[i,j] = xe*_σ[i,j]; σeff[j,i] = σeff[i,j]
            ϵeff[i,j] = ϵe; ϵeff[j,i] = ϵe
            α[i,j] = αe; α[j,i] = αe
        end
        _d[i] = d_imp(T,λmat[i,i],cmat[i,i],actmat[i,i],_σ[i,i],xeff[i,i])
    end
    return (λ = λmat, c = cmat, active = actmat, xeff = xeff, σeff = σeff, ϵeff = ϵeff, α = α, d = _d)
end

"""
    imp_effective_parameters(model::SAFTVRIMPModel, T)

Returns `(σeff, ϵeff, α, d)` at temperature `T` (matrices for pairs, vector for `d`; lengths in m, energies in K).
"""
function imp_effective_parameters(model::SAFTVRIMPModel,T)
    n = length(model)
    pd = imp_pairdata(model,zero(T),T,ones(n))
    return (σeff = pd.σeff, ϵeff = pd.ϵeff, α = pd.α, d = pd.d)
end

d(model::SAFTVRIMPModel,V,T,z) = imp_pairdata(model,V,T,z).d

function imp_ζ_X_σ3(model::SAFTVRIMPModel,V,T,z,_d,σeff,m̄)
    m = model.params.segment.values
    m̄inv = 1/m̄
    ρS = N_A/V*m̄
    kρS = ρS*π/6/8
    _ζ_X = zero(V+T+first(z)+one(eltype(model)))
    σ3x = _ζ_X
    for i in 1:length(z)
        x_Si = z[i]*m[i]*m̄inv
        σ3x += x_Si*x_Si*(σeff[i,i]^3)
        di = _d[i]
        _ζ_X += kρS*x_Si*x_Si*(2*di)^3
        for j in 1:(i-1)
            x_Sj = z[j]*m[j]*m̄inv
            σ3x += 2*x_Si*x_Sj*(σeff[i,j]^3)
            _ζ_X += 2*kρS*x_Si*x_Sj*(di + _d[j])^3
        end
    end
    return _ζ_X,σ3x
end

function data(model::SAFTVRIMPModel,V,T,z)
    m̄ = dot(z,model.params.segment.values)
    pd = @f(imp_pairdata)
    _d = pd.d
    ζi = @f(ζ0123,_d)
    _ρ_S = N_A/V*m̄
    _ζ_X,σeff3x = @f(imp_ζ_X_σ3,_d,pd.σeff,m̄)
    _ζst = σeff3x*_ρ_S*π/6
    return (_d,_ρ_S,ζi,_ζ_X,_ζst,σeff3x,m̄,pd)
end

function packing_fraction(model::SAFTVRIMPModel,_data::Tuple)
    _,_,ζi = _data
    return ζi[4]
end

#=
SAFT-VR kernels (identical formulas to the SAFT-VR Mie implementation, written as free
functions of λ, x and ζₓ). Each `_fdf` returns (f, ∂(ρₛf)/∂ρₛ) at constant T and composition.
=#

function imp_ζeff_f_ρdf(λ,ζₓ)
    A = SAFTγMieconsts.A
    λ⁻¹ = one(λ)/λ
    Aλ⁻¹ = A * SA[one(λ); λ⁻¹; λ⁻¹*λ⁻¹; λ⁻¹*λ⁻¹*λ⁻¹]
    f = dot(Aλ⁻¹,SA[ζₓ; ζₓ^2; ζₓ^3; ζₓ^4])
    ρdf = dot(Aλ⁻¹,SA[one(ζₓ); 2ζₓ; 3ζₓ^2; 4ζₓ^3])*ζₓ
    return f,ρdf
end

function imp_aS1_fdf(λ,ζₓ)
    ζeff,∂ζeff = imp_ζeff_f_ρdf(λ,ζₓ)
    ζeff3 = (1 - ζeff)^3
    ζeffm1 = (1 - ζeff*0.5)
    ζf = ζeffm1/ζeff3
    λf = -1/(λ - 3)
    f = λf*ζf
    df = λf*(ζf + ∂ζeff*((3*ζeffm1*(1 - ζeff)^2 - 0.5*ζeff3)/ζeff3^2))
    return f,df
end

function imp_B_fdf(λ,x,ζₓ)
    x_3λ = x^(3 - λ)
    I = (1 - x_3λ)/(λ - 3)
    J = (1 - (λ - 3)*x^(4 - λ) + (λ - 4)*x_3λ)/((λ - 3)*(λ - 4))
    ζX2 = (1 - ζₓ)^2
    ζX3 = (1 - ζₓ)^3
    ζX6 = ζX3*ζX3
    f = I*(1 - ζₓ/2)/ζX3 - 9*J*ζₓ*(ζₓ + 1)/(2*ζX3)
    df = (((1 - ζₓ/2)*I/ζX3 - 9*ζₓ*(1 + ζₓ)*J/(2*ζX3))
        + ζₓ*((3*(1 - ζₓ/2)*ζX2 - 0.5*ζX3)*I/ζX6
        - 9*J*((1 + 2*ζₓ)*ζX3 + ζₓ*(1 + ζₓ)*3*ζX2)/(2*ζX6)))
    return f,df
end

function imp_KHS_f_ρdf(ζₓ)
    ζX4 = (1 - ζₓ)^4
    denom1 = evalpoly(ζₓ,(1,4,4,-4,1))
    ∂denom1 = evalpoly(ζₓ,(4,8,-12,4))
    f = ζX4/denom1
    ρdf = -ζₓ*((4*(1 - ζₓ)^3*denom1 + ζX4*∂denom1)/denom1^2)
    return f,ρdf
end

function imp_gHS(x,ζₓ)
    ζX3 = (1 - ζₓ)^3
    k_0 = -log(1 - ζₓ) + evalpoly(ζₓ,(0,42,-39,9,-2))/(6*ζX3)
    k_1 = evalpoly(ζₓ,(0,-12,6,0,1))/(2*ζX3)
    k_2 = -3*ζₓ^2/(8*(1 - ζₓ)^2)
    k_3 = evalpoly(ζₓ,(0,3,3,0,-1))/(6*ζX3)
    return exp(evalpoly(x,(k_0,k_1,k_2,k_3)))
end

function imp_f123456(α)
    ϕ = SAFTVRMieconsts.ϕ
    _0 = zero(α)
    fa = (_0,_0,_0,_0,_0,_0)
    fb = (_0,_0,_0,_0,_0,_0)
    @inbounds for i ∈ 1:4
        ϕi = ϕ[i]::NTuple{6,Float64}
        fa = fa .+ ϕi .* α^(i - 1)
    end
    @inbounds for i ∈ 5:7
        ϕi = ϕ[i]::NTuple{6,Float64}
        fb = fb .+ ϕi .* α^(i - 4)
    end
    return fa ./ (one(_0) .+ fb)
end

#=
Monomer dispersion and chain terms (single fused pass, as `a_dispchain` in SAFT-VR Mie).
With Φ(λ) = aS₁(λ; ζₓ) + B(λ, x_eff; ζₓ):
  a₁,ij  = −2πρₛ d³ Σₖ cₖ x₀^λₖ Φ(λₖ)
  a₂,ij  =  π K_HS (1 + χ) ρₛ d³ Σₖ Σₗ cₖcₗ x₀^(λₖ+λₗ) Φ(λₖ+λₗ)
  a₃,ij  = −ϵ_eff³ f₄(α) ζ̄ exp(f₅ζ̄ + f₆ζ̄²)                               (order = 3 only)
  g₁     = [Σₖ λₖcₖx₀^λₖ Φ(λₖ) − 3 ∂(ρₛS₁)/∂ρₛ]/ϵ_eff
  g₂,MCA = [3·½(ρₛ∂K_HS/∂ρₛ S₂ + K_HS ∂(ρₛS₂)/∂ρₛ) − K_HS Σₖₗ λₖcₖcₗx₀^(λₖ+λₗ)Φ(λₖ+λₗ)]/ϵ_eff²
  ln y(σ_eff) = ln g_HS(x_eff) + [τg₁ + τ²(1 + γc)g₂,MCA]/g_HS ,  τ = ϵ_eff/T
=#

function imp_dispchain(model::SAFTVRIMPModel,V,T,z,_data,disp::Bool,chain::Bool)
    _d,ρS,ζi,ζₓ,ζst,_,m̄,pd = _data
    n = length(z)
    m = model.params.segment.values
    _σ = model.params.sigma.values
    order = model.imp_options.order
    ∑z = sum(z)
    m̄inv = 1/m̄
    _0 = zero(V+T+first(z)+one(eltype(model)))
    a₁,a₂,a₃,achain = _0,_0,_0,_0
    ζst5 = ζst^5
    ζst8 = ζst^8
    KHS,ρS_∂KHS = imp_KHS_f_ρdf(ζₓ)
    for i in 1:n
        x_Si = z[i]*m[i]*m̄inv
        for j in 1:i
            chain_ii = chain && i == j && m[i] != 1
            (disp || chain_ii) || continue
            x_Sj = z[j]*m[j]*m̄inv
            λ = pd.λ[i,j]
            c = pd.c[i,j]
            act = pd.active[i,j]
            ϵeff = pd.ϵeff[i,j]
            α = pd.α[i,j]
            dij = i == j ? _d[i] : 0.5*(_d[i] + _d[j])
            dij3 = dij^3
            x0 = _σ[i,j]/dij
            xeff = pd.σeff[i,j]/dij

            #first-order sums: S₁ = Σ cₖx₀^λₖΦ, ∂S₁ = ∂(ρₛS₁)/∂ρₛ, Λ₁ = Σ λₖcₖx₀^λₖΦ
            S₁,∂S₁,Λ₁ = _0,_0,_0
            for k in 1:IMP_NTERMS
                act[k] || continue
                aS,∂aS = imp_aS1_fdf(λ[k],ζₓ)
                B,∂B = imp_B_fdf(λ[k],xeff,ζₓ)
                t = c[k]*x0^λ[k]
                Φ = aS + B
                S₁ += t*Φ
                ∂S₁ += t*(∂aS + ∂B)
                Λ₁ += λ[k]*t*Φ
            end

            #second-order sums over unordered pairs of Sutherland terms
            S₂,∂S₂,Λ₂ = _0,_0,_0
            for k in 1:IMP_NTERMS
                act[k] || continue
                for l in 1:k
                    act[l] || continue
                    λkl = λ[k] + λ[l]
                    aS,∂aS = imp_aS1_fdf(λkl,ζₓ)
                    B,∂B = imp_B_fdf(λkl,xeff,ζₓ)
                    t = c[k]*c[l]*x0^λkl
                    Φ = aS + B
                    if k == l
                        S₂ += t*Φ
                        ∂S₂ += t*(∂aS + ∂B)
                        Λ₂ += λ[k]*t*Φ
                    else
                        S₂ += 2*t*Φ
                        ∂S₂ += 2*t*(∂aS + ∂B)
                        Λ₂ += λkl*t*Φ
                    end
                end
            end

            if disp
                w = i == j ? x_Si*x_Sj : 2*x_Si*x_Sj
                f1,f2,f3,f4,f5,f6 = imp_f123456(α)
                χ = f1*ζst + f2*ζst5 + f3*ζst8
                a₁ -= w*(2*π*ρS*dij3*S₁)
                a₂ += w*(π*KHS*(1 + χ)*ρS*dij3*S₂)
                if order == 3
                    a₃ -= w*(ϵeff^3*f4*ζst*exp(ζst*(f5 + f6*ζst)))
                end
            end

            if chain_ii
                gHS = imp_gHS(xeff,ζₓ)
                τ = ϵeff/T
                g₁ = (Λ₁ - 3*∂S₁)/ϵeff
                g₂MCA = (1.5*(ρS_∂KHS*S₂ + KHS*∂S₂) - KHS*Λ₂)/(ϵeff*ϵeff)
                θ = expm1(τ)
                γc = 10*(-tanh(10*(0.57 - α)) + 1)*ζst*θ*exp(ζst*(-6.7 - 8*ζst))
                g₂ = (1 + γc)*g₂MCA
                lny = log(gHS) + (τ*g₁ + τ*τ*g₂)/gHS
                achain -= z[i]*(m[i] - 1)*lny
            end
        end
    end
    adisp = (a₁/T + a₂/(T*T) + a₃/(T*T*T))*m̄/∑z
    return adisp,achain/∑z
end

function a_hs(model::SAFTVRIMPModel,V,T,z,_data = @f(data))
    _d,_,ζi,_,_,_,m̄ = _data
    ζ0,ζ1,ζ2,ζ3 = ζi
    if !iszero(ζ3)
        _a_hs = bmcs_hs(ζ0,ζ1,ζ2,ζ3)
    else
        _a_hs = @f(bmcs_hs_zero_v,_d)
    end
    return m̄*_a_hs/sum(z)
end

function a_disp(model::SAFTVRIMPModel,V,T,z,_data = @f(data))
    return first(imp_dispchain(model,V,T,z,_data,true,false))
end

function a_mono(model::SAFTVRIMPModel,V,T,z,_data = @f(data))
    return @f(a_hs,_data) + @f(a_disp,_data)
end

function a_chain(model::SAFTVRIMPModel,V,T,z,_data = @f(data))
    return last(imp_dispchain(model,V,T,z,_data,false,true))
end

function a_res(model::SAFTVRIMPModel,V,T,z,_data = @f(data))
    adisp,achain = imp_dispchain(model,V,T,z,_data,true,true)
    return @f(a_hs,_data) + adisp + achain
end
