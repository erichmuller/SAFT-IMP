# SAFT-VR IMP for Clapeyron.jl

SAFT-VR equation of state for chains of tangent segments interacting through the isotropic multipolar potential of Müller and Gelb (2003), written as a temperature-dependent sum of Sutherland terms and solved with the SAFT-VR Mie / SAFT-VR Sum perturbation theory. Developed and tested against Clapeyron.jl 0.6.28 on Julia 1.11.7.

## Contents

| Path | Purpose |
|---|---|
| `SAFTVRIMP.jl` | Model: parameters, constructor, Sutherland-sum potential, effective parameters, BH diameter, `a_hs`, `a_mono`, `a_disp`, `a_chain`, `a_res` |
| `database/SAFTVRIMP_like.csv` | Parameters: n-hexane and n-decane (SAFT-VR Mie, no moments) and five Müller–Gelb IMP fluids |
| `test/test_SAFTVRIMP.jl` | Validation and demonstration script (97 tests) |
| `test_output.txt` | Output of the script |
| `SAFTVRIMP_derivation.md` | Step-by-step derivation, equation-to-code map, verification and predictions |
| `SAFTVRIMP_intree.patch` | Patch adding the model to the Clapeyron source tree |

## Use without modifying Clapeyron

```julia
using Clapeyron, ForwardDiff
Base.include(Clapeyron, "path/to/SAFTVRIMP/SAFTVRIMP.jl")   # evaluated inside the Clapeyron module
const SAFTVRIMP = Clapeyron.SAFTVRIMP

model = SAFTVRIMP(["hexane"])                 # second-order Barker–Henderson monomer (default)
model3 = SAFTVRIMP(["hexane"]; order = 3)     # adds Lafitte's a₃; identical to SAFTVRMie without moments

p = pressure(model, 1.4e-4, 350.0)
∇p = ForwardDiff.gradient(x -> pressure(model, x[1], x[2], [x[3]]), [1.4e-4, 350.0, 1.0])
psat, vl, vv = saturation_pressure(model, 400.0)
Tc, pc, Vc = crit_pure(model)

Clapeyron.sutherland_terms(SAFTVRIMP(["benzene_MG2003"]), 450.0)     # (λₖ, εₖ) with φ = −Σ εₖ (σ/r)^λₖ
Clapeyron.imp_effective_parameters(SAFTVRIMP(["benzene_MG2003"]), 450.0)   # σ_eff, ϵ_eff, α, d
```

Parameters can also be passed directly, e.g. `SAFTVRIMP(["illustrative polar fluid"]; userlocations = (Mw = [58.08], segment = [1.5], sigma = [3.6], epsilon = [300.0], lambda_r = [15.0], lambda_a = [6.0], dipole = [2.9], quadrupole = [0.0], polarizability = [6.4]))`. (The numbers are placeholders, not fitted parameters.) Units: σ in Å, ε in K, μ in D, Q in D·Å (Buckingham), α in Å³ (all molecular; they are spread over segments as μ²/m, Q²/m, α/m).

## In-tree integration

Apply `SAFTVRIMP_intree.patch` from the parent directory of the Clapeyron checkout (`patch -p1 -d Clapeyron.jl < SAFTVRIMP_intree.patch`), or do it by hand: copy `SAFTVRIMP.jl` to `src/models/SAFT/SAFTVRIMP/`, the CSV to `database/SAFT/SAFTVRIMP/`, and add `include("models/SAFT/SAFTVRIMP/SAFTVRIMP.jl")` to `src/Clapeyron.jl` after the SAFT-γ Mie includes.

## Run the tests

```
julia --project=<env with Clapeyron, ForwardDiff, StaticArrays> test/test_SAFTVRIMP.jl
```

About 3.5 minutes on one CPU, most of it compilation.

## Caveats

The Müller–Gelb parameters were fitted to molecular dynamics with a potential cut and shifted at 5σ; they are included to exercise the multipolar terms and are not accurate EoS parameters (for example, they give boiling points of 1,2-dichloroethane and cyclohexane that are 25–45 K too low). Refit ε, σ, λr (and m) for the chosen perturbation order with the moments fixed. There is no association term.
