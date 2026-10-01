# SAFT-VR IMP: derivation

SAFT-VR for chains of tangent segments interacting through the isotropic multipolar potential (IMP) of Müller and Gelb (2003), written as a temperature-dependent Sutherland sum and treated with the SAFT-VR Mie (Lafitte et al., 2013) and SAFT-VR Sum (Jervell et al., 2026) perturbation machinery. Every equation below maps one-to-one onto a function in `SAFTVRIMP.jl`; the correspondence is listed in Section 8.

Conventions. Energies are written in kelvin ($u \equiv u/k_\mathrm{B}$), so $\beta u = u/T$. Components are indexed $i,j$; since the model is homonuclear, segment types coincide with components and pair indices $k,l$ run over components as well. For amounts $z_i$, $x_i = z_i/\sum_j z_j$, the mean chain length is $\bar m = \sum_i x_i m_i$, the segment number density is $\rho_s = N_\mathrm{A}\sum_i z_i m_i/V$ (the $\rho_m$ of the brief), and the segment fractions are $x_{s,i} = z_i m_i/\sum_j z_j m_j$.

---

## 1. The IMP as a Sutherland sum

### 1.1 Orientational averaging

For an axially symmetric multipolar pair with separation $r$ and relative orientation $\omega$, the pair Boltzmann factor integrated over orientations defines an isotropic potential of mean force $\Gamma(r;T)$,

$$
e^{-\beta\Gamma(r;T)} = \left\langle e^{-\beta u(r,\omega)} \right\rangle_\omega .
$$

The cumulant expansion gives $\beta\Gamma = \beta\langle u\rangle_\omega - \tfrac{\beta^2}{2}\left(\langle u^2\rangle_\omega - \langle u\rangle_\omega^2\right) + O(\beta^3)$. For permanent multipole–multipole interactions the unweighted angular average vanishes, $\langle u\rangle_\omega = 0$, hence

$$
\Gamma(r;T) \simeq -\frac{\beta}{2}\langle u^2\rangle_\omega .
$$

With the standard angular averages (Gaussian units; in SI replace $\mu^2 \to \mu^2/4\pi\varepsilon_0$ and likewise for $Q^2$) this yields eqs. (11)–(13) of Müller and Gelb,

$$
\Gamma^{\mu\mu} = -\frac{\beta\,\mu_k^2\mu_l^2}{3r^6},\qquad
\Gamma^{\mu Q} = -\frac{\beta\left(\mu_k^2Q_l^2 + Q_k^2\mu_l^2\right)}{2r^8},\qquad
\Gamma^{QQ} = -\frac{7\beta\,Q_k^2Q_l^2}{5r^{10}},
$$

and the orientationally averaged dipole–induced-dipole (Debye) term, which carries no $\beta$, is $\Gamma^{\mu\alpha} = -(\mu_k^2\alpha_l + \alpha_k\mu_l^2)/r^6$ (eq. 14).

Because $\Gamma$ is defined through the configurational integral, it reproduces the pair configurational integral (hence the second virial coefficient) of the anisotropic fluid to this order; for the dense fluid, replacing the angle-dependent potential by $\Gamma$ neglects orientational correlations among three or more molecules, which is the approximation that defines the IMP. Within it, the Helmholtz energy $A(V,T,N)$ built with the temperature-dependent pair potential is a free energy, not an energy, so every temperature derivative must include $\partial\Gamma/\partial T$. A direct check of this statement: for $\Gamma \propto \beta$ one has $\partial(\beta\Gamma)/\partial\beta = 2\Gamma$, so the energy obtained from $A$ is twice the free energy of the pair, which is the classical Keesom energy $-2\mu^4/(3k_\mathrm{B}Tr^6)$; this is the factor of two noted by Müller and Gelb. In the implementation this happens automatically because $T$ is propagated through every coefficient by ForwardDiff.jl (Section 7).

The truncation is meaningful only when $|\beta\Gamma(\sigma)|$ is small. In reduced units $\mu^{*2} = \hat\mu^2/(\varepsilon\sigma^3)$, $Q^{*2} = \hat Q^2/(\varepsilon\sigma^5)$, $T^* = T/\varepsilon$, the contact values are $|\beta\Gamma^{\mu\mu}(\sigma)| = \mu^{*4}/(3T^{*2})$ and $|\beta\Gamma^{QQ}(\sigma)| = 7Q^{*4}/(5T^{*2})$.

### 1.2 Moments on chains

The monomer term of SAFT-VR is a sum over segment pairs. Distributing a molecular moment uniformly over the $m_i$ identical segments such that the far field of the $m_k m_l$ segment pairs reproduces the molecular interaction requires $m_k m_l\,\hat\mu_k^2\hat\mu_l^2 = \mu_k^2\mu_l^2$, i.e.

$$
\hat\mu_i^2 = \frac{\mu_i^2}{4\pi\varepsilon_0\, m_i},\qquad \hat Q_i^2 = \frac{Q_i^2}{4\pi\varepsilon_0\, m_i},\qquad \hat\alpha_i = \frac{\alpha_i}{m_i},
$$

which is consistent for all cross products ($\mu Q$, $\mu\alpha$) and is the same scaling as the reduced moments $\mu^{*2} = \mu^2/(m\varepsilon\sigma^3)$ of the Gross–Vrabec polar PC-SAFT terms. For $m_i = 1$ the Müller–Gelb potential is recovered exactly. Numerically, $(1\,\mathrm{D})^2/(4\pi\varepsilon_0k_\mathrm{B}) = 7242.97\ \mathrm{K\,Å^3}$, and the same constant converts $(\mathrm{D\,Å})^2$ to $\mathrm{K\,Å^5}$.

### 1.3 The Sutherland representation

Adding a Mie core for repulsion and dispersion (Müller and Gelb used the Lennard-Jones case $\lambda_r = 12$, $\lambda_a = 6$), the pair potential between segments of $k$ and $l$ is a finite Sutherland sum

$$
u_{kl}(r;T) = \sum_{n=1}^{5} c_{kl,n}(T)\left(\frac{\sigma_{kl}}{r}\right)^{\lambda_n},
$$

| $n$ | $\lambda_n$ | $c_{kl,n}(T)$ [K] | physics |
|---|---|---|---|
| 1 | $\lambda_{r,kl}$ | $+C_{kl}\,\varepsilon_{kl}$ | soft repulsion |
| 2 | $\lambda_{a,kl}$ | $-C_{kl}\,\varepsilon_{kl}$ | dispersion |
| 3 | 6 | $-\left[\hat\mu_k^2\hat\alpha_l + \hat\alpha_k\hat\mu_l^2 + \hat\mu_k^2\hat\mu_l^2/(3T)\right]/\sigma_{kl}^6$ | induction + Keesom |
| 4 | 8 | $-\left(\hat\mu_k^2\hat Q_l^2 + \hat Q_k^2\hat\mu_l^2\right)/(2T\sigma_{kl}^8)$ | dipole–quadrupole |
| 5 | 10 | $-7\hat Q_k^2\hat Q_l^2/(5T\sigma_{kl}^{10})$ | quadrupole–quadrupole |

with $C_{kl} = \frac{\lambda_r}{\lambda_r-\lambda_a}\left(\frac{\lambda_r}{\lambda_a}\right)^{\lambda_a/(\lambda_r-\lambda_a)}$. In the notation of the brief, $\phi(r) = -\sum_n \varepsilon_n(\sigma/r)^{\lambda_n}$ with $\varepsilon_n = -c_n$ (`sutherland_terms` returns exactly these pairs).

The multipolar coefficients need no combining rule: they are products of pure-component moments, and a combining rule applied to $\varepsilon_n$ could not reproduce, for example, $\hat\mu_k^2\hat Q_l^2 + \hat Q_k^2\hat\mu_l^2$ for a dipolar–quadrupolar pair whose like terms at $\lambda = 8$ both vanish. This is why the parameter struct stores the moments from which the $(\lambda_n, \varepsilon_n)$ set is generated rather than the $\varepsilon_n$ themselves. The Mie part uses the SAFT-VR Mie rules $\sigma_{kl} = (\sigma_k+\sigma_l)/2$, $\varepsilon_{kl} = \sqrt{\sigma_k^3\sigma_l^3}\,\sigma_{kl}^{-3}\sqrt{\varepsilon_k\varepsilon_l}\,(1-k_{kl})$ and $\lambda_{kl}-3 = \sqrt{(\lambda_k-3)(\lambda_l-3)}$.

Well-posedness requires the repulsion to dominate at short range, $\lambda_r > \max\{\lambda_n : c_n\not\equiv 0,\ n\ge 2\}$ (so $\lambda_r > 10$ whenever a quadrupole is present), and $\lambda_a > 4$ for the integrals $J_\lambda$ below. Both are checked when the model is built.

### 1.4 Relation to the hard-core form of the brief

The piecewise potential $\phi = \infty$ for $r<\sigma$, $-\sum_n\varepsilon_n(\sigma/r)^{\lambda_n}$ for $r\ge\sigma$, is the $\lambda_r\to\infty$ limit of the table above ($C\to 1$). In that limit $d = \sigma$ is temperature independent, so a temperature-dependent diameter $d_i(T)$ exists only for the soft core, which is also the potential actually used by Müller and Gelb. The monomer terms derived below reduce continuously to hard-core SAFT-VR Sutherland in this limit ($\sigma_\mathrm{eff}\to\sigma$, $d\to\sigma$, and the $B$ integrals vanish because $I_\lambda(1) = J_\lambda(1) = 0$). The chain term does not commute with the limit: with a hard core the potential jumps at contact and the cavity function must be taken as $y(\sigma) = e^{-\beta\sum_n\varepsilon_n}g(\sigma^+)$ (Gil-Villegas et al., 1997), a factor that is non-perturbative in $\beta$. With the soft core the contact is taken where $u=0$ (Section 5), where $y = g$ exactly.

---

## 2. Effective range, well depth and van der Waals energy

Following SAFT-VRQ Mie and SAFT-VR Sum, define $\sigma_\mathrm{eff}$ by $u(\sigma_\mathrm{eff}) = 0$, $r_\mathrm{min}$ by $u'(r_\mathrm{min}) = 0$, $\varepsilon_\mathrm{eff} = -u(r_\mathrm{min})$, and the dimensionless van der Waals energy

$$
\alpha = -\frac{1}{\varepsilon_\mathrm{eff}\sigma_\mathrm{eff}^3}\int_{\sigma_\mathrm{eff}}^\infty u(r)\,r^2\,dr = -\frac{1}{\varepsilon_\mathrm{eff}}\sum_n \frac{c_n}{\lambda_n-3}\left(\frac{\sigma}{\sigma_\mathrm{eff}}\right)^{\lambda_n}.
$$

With $x = r/\sigma$ and $s = \ln x$, both conditions take the form (weights $w_1 = c_1$, $w_n = |c_n|$ for the zero; $w_1 = \lambda_1 c_1$, $w_n = \lambda_n|c_n|$ for the minimum)

$$
F(s) = \ln w_1 - \lambda_1 s - \ln\sum_{n\ge2} w_n e^{-\lambda_n s} = 0 .
$$

Writing $\langle\lambda\rangle_w$ and $\mathrm{Var}_w(\lambda)$ for the mean and variance of the attractive exponents under the normalized weights $w_ne^{-\lambda_ns}$, one finds $F'(s) = \langle\lambda\rangle_w - \lambda_1 < 0$ and $F''(s) = -\mathrm{Var}_w(\lambda) \le 0$. $F$ is strictly decreasing and concave, so the root is unique and Newton's method converges globally. Starting from the Mie solutions ($s = 0$ for the zero, $s = \ln(\lambda_r/\lambda_a)/(\lambda_r-\lambda_a)$ for the minimum) always gives $F \le 0$, because the extra multipolar weights only increase the logarithm, and from that side the Newton iterates decrease monotonically onto the root. For pairs without active multipolar terms the closed forms $\sigma_\mathrm{eff} = \sigma$, $\varepsilon_\mathrm{eff} = \varepsilon$, $\alpha = C\left[(\lambda_a-3)^{-1}-(\lambda_r-3)^{-1}\right]$ are used, which makes the model identical to SAFT-VR Mie in that limit.

Their temperature derivatives follow from the implicit function theorem and the envelope theorem,

$$
\frac{dx_\mathrm{eff}}{dT} = -\left.\frac{\partial u/\partial T}{\partial u/\partial x}\right|_{x_\mathrm{eff}},\qquad
\frac{d\varepsilon_\mathrm{eff}}{dT} = -\left.\frac{\partial u}{\partial T}\right|_{x_\mathrm{min}},\qquad
\frac{\partial u}{\partial T} = \sum_n \frac{dc_n}{dT}\,x^{-\lambda_n}.
$$

A closed form exists when a single multipolar term shares the dispersion exponent, $u = C\varepsilon x^{-\lambda_r} - (C\varepsilon + |c|)x^{-\lambda_a}$: the potential is again Mie$(\lambda_r,\lambda_a)$ with

$$
F = 1 + \frac{|c|}{C\varepsilon},\qquad \sigma_m = \sigma F^{-1/(\lambda_r-\lambda_a)},\qquad \varepsilon_m = \varepsilon F^{\lambda_r/(\lambda_r-\lambda_a)},
$$

which for the Lennard-Jones Keesom fluid is $\varepsilon_m = \varepsilon F^2$, $\sigma_m = \sigma F^{-1/6}$, $F = 1+\mu^{*4}/(12T^*)$ (eqs. 16–18 of Müller and Gelb). This mapping is used as an exact numerical check in Section 9.

---

## 3. Reference fluid

The Barker–Henderson split is made at $\sigma_\mathrm{eff}$: $u_0 = u$ for $r\le\sigma_\mathrm{eff}$ and $u_1 = u$ beyond. The reference is replaced by hard spheres of diameter

$$
d_k(T) = \int_0^{\sigma_{\mathrm{eff},kk}}\left[1 - e^{-\beta u_{kk}(r;T)}\right]dr .
$$

Because $u(\sigma_\mathrm{eff}) = 0$, the integrand vanishes at the moving upper limit, so

$$
\frac{dd_k}{dT} = \int_0^{\sigma_\mathrm{eff}} e^{-\beta u}\left[-\frac{u}{T^2} + \frac{1}{T}\frac{\partial u}{\partial T}\right]dr ,
$$

where the second bracket term is the new contribution of the temperature-dependent potential. Numerically the integral is evaluated exactly as `d_vrmie` does for Mie fluids, generalized to the sum: for $\theta = c_1/T > 1$ the substitution $y = x^{-\lambda_1}$ turns it into a 10-point Gauss–Laguerre integral on $(x_\mathrm{eff}^{-\lambda_1},\infty)$ with weight $e^{-\theta y}$; otherwise a 10-point Gauss–Legendre rule is applied above the point where $e^{-\beta u}$ drops below machine precision.

Mixtures use an additive hard-sphere reference, $d_{kl} = (d_k+d_l)/2$, as in SAFT-VR Mie; this keeps the BMCSL reference and the contact values of the chain term mutually consistent (SAFT-VRQ Mie, which has no chain term, uses a non-additive reference instead). With $\zeta_\ell = \frac{\pi}{6}\rho_s\sum_k x_{s,k}d_k^\ell$ the Boublík–Mansoori–Carnahan–Starling–Leland result per segment is

$$
\tilde a^\mathrm{HS} = \frac{6}{\pi\rho_s}\left[\left(\frac{\zeta_2^3}{\zeta_3^2}-\zeta_0\right)\ln(1-\zeta_3) + \frac{3\zeta_1\zeta_2}{1-\zeta_3} + \frac{\zeta_2^3}{\zeta_3(1-\zeta_3)^2}\right],\qquad \frac{A^\mathrm{HS}}{Nk_\mathrm{B}T} = \bar m\,\tilde a^\mathrm{HS}.
$$

Two one-fluid packing fractions enter the perturbation terms:

$$
\zeta_x = \frac{\pi}{6}\rho_s\sum_{k,l}x_{s,k}x_{s,l}\,d_{kl}^3,\qquad \bar\zeta_x = \frac{\pi}{6}\rho_s\sum_{k,l}x_{s,k}x_{s,l}\,\sigma_{\mathrm{eff},kl}^3 .
$$

As in SAFT-VR Mie, the pair contact values $g^\mathrm{HS}_{kl}(d_{kl}^+)$ inside the perturbation integrals are evaluated with the one-fluid $\zeta_x$ (van der Waals one-fluid theory for the segment mixture); $\bar\zeta_x$ enters only the empirical corrections $\chi$, $a_3$ and $\gamma_c$.

---

## 4. Monomer perturbation terms

$$
\frac{A^\mathrm{mono}}{Nk_\mathrm{B}T} = \bar m\left[\tilde a^\mathrm{HS} + \beta a_1 + \beta^2 a_2\ \left(+\,\beta^3 a_3\right)\right],\qquad a_p = \sum_{k,l}x_{s,k}x_{s,l}\,a_{p,kl}.
$$

The default model stops at $\beta^2$ as specified; `order = 3` adds the Lafitte correlation for $a_3$. The expansion is in $\beta$ at fixed pair potential: the $1/T$ inside the multipolar $c_n(T)$ is not reordered into higher powers of $\beta$, which is the same effective-potential convention as SAFT-VRQ Mie (Feynman–Hibbs) and SAFT-VR Sum.

### 4.1 First order

$$
a_{1,kl} = 2\pi\rho_s\int_{\sigma_\mathrm{eff}}^\infty g^\mathrm{HS}(r)\,u_{kl}(r)\,r^2\,dr = 2\pi\rho_s\sum_n c_n\,\mathcal I(\lambda_n),\qquad \mathcal I(\lambda) = \int_{\sigma_\mathrm{eff}}^\infty g^\mathrm{HS}(r)\left(\frac{\sigma}{r}\right)^\lambda r^2\,dr .
$$

Split $\int_{\sigma_\mathrm{eff}}^\infty = \int_d^\infty - \int_d^{\sigma_\mathrm{eff}}$ and set $x_0 = \sigma/d$, $x_\mathrm{eff} = \sigma_\mathrm{eff}/d$.

The first piece is a Sutherland integral with hard core $d$; the mean-value theorem with the effective packing fraction of Lafitte et al. gives

$$
\int_d^\infty g^\mathrm{HS}(r)\left(\frac{\sigma}{r}\right)^\lambda r^2\,dr \simeq \frac{x_0^\lambda d^3}{\lambda-3}\,\frac{1-\zeta_\mathrm{eff}/2}{(1-\zeta_\mathrm{eff})^3},\qquad \zeta_\mathrm{eff}(\zeta_x,\lambda) = \sum_{j=1}^4 \gamma_j(\lambda)\,\zeta_x^j,
$$

with $\gamma_j(\lambda)$ cubic polynomials in $1/\lambda$ (matrix $A$ of Lafitte et al.).

The second piece is short, so $g^\mathrm{HS}$ is linearized about contact, $g(r)\simeq g(d) + g'(d)(r-d)$, with the Carnahan–Starling contact value and the Percus–Yevick slope, $g(d) = (1-\zeta_x/2)/(1-\zeta_x)^3$ and $d\,g'(d) = -9\zeta_x(1+\zeta_x)/[2(1-\zeta_x)^3]$:

$$
\int_d^{\sigma_\mathrm{eff}} g\left(\frac{\sigma}{r}\right)^\lambda r^2\,dr \simeq x_0^\lambda d^3\left[g(d)\,I_\lambda(x_\mathrm{eff}) + d\,g'(d)\,J_\lambda(x_\mathrm{eff})\right],
$$

$$
I_\lambda(x) = \int_1^x t^{2-\lambda}dt = \frac{1-x^{3-\lambda}}{\lambda-3},\qquad
J_\lambda(x) = \int_1^x (t-1)\,t^{2-\lambda}dt = \frac{1-(\lambda-3)x^{4-\lambda}+(\lambda-4)x^{3-\lambda}}{(\lambda-3)(\lambda-4)} .
$$

Defining the dimensionless kernels

$$
a^S_1(\lambda) = -\frac{1}{\lambda-3}\frac{1-\zeta_\mathrm{eff}/2}{(1-\zeta_\mathrm{eff})^3},\qquad
B(\lambda,x) = I_\lambda(x)\frac{1-\zeta_x/2}{(1-\zeta_x)^3} - J_\lambda(x)\frac{9\zeta_x(1+\zeta_x)}{2(1-\zeta_x)^3},\qquad
\Phi(\lambda) = a^S_1(\lambda) + B(\lambda,x_\mathrm{eff}),
$$

the integral is $\mathcal I(\lambda) = -d^3x_0^\lambda\,\Phi(\lambda)$ and

$$
\boxed{\,a_{1,kl} = -2\pi\rho_s\,d_{kl}^3\sum_n c_{kl,n}\,x_0^{\lambda_n}\,\Phi(\lambda_n)\,}
$$

Note that $x_0 = \sigma/d$ multiplies the kernel because the Sutherland terms are written in units of $\sigma$, whereas the upper limit of the $B$ integral is $x_\mathrm{eff} = \sigma_\mathrm{eff}/d$. For a Mie pair ($c_1 = -c_2 = C\varepsilon$, $x_\mathrm{eff} = x_0$) this is the SAFT-VR Mie expression $a_1 = 2\pi\rho_s\varepsilon d^3 C[x_0^{\lambda_a}\Phi(\lambda_a) - x_0^{\lambda_r}\Phi(\lambda_r)]$.

### 4.2 Second order

In the macroscopic compressibility approximation the energy fluctuations of a shell of fluid are those of a hard-sphere fluid of isothermal compressibility $K^\mathrm{HS} = k_\mathrm{B}T(\partial\rho/\partial p)^\mathrm{HS}$; with the correction factor $\chi$,

$$
a_{2,kl} = -\pi\rho_s K^\mathrm{HS}(1+\chi_{kl})\int_{\sigma_\mathrm{eff}}^\infty g^\mathrm{HS}\,u_{kl}^2\,r^2\,dr .
$$

Since $u^2 = \sum_{n,m}c_nc_m(\sigma/r)^{\lambda_n+\lambda_m}$ is again a Sutherland sum, the same kernels apply:

$$
\boxed{\,a_{2,kl} = \pi\rho_s K^\mathrm{HS}(1+\chi_{kl})\,d_{kl}^3\sum_{n,m}c_{kl,n}c_{kl,m}\,x_0^{\lambda_n+\lambda_m}\,\Phi(\lambda_n+\lambda_m)\,}
$$

$$
K^\mathrm{HS} = \frac{(1-\zeta_x)^4}{1+4\zeta_x+4\zeta_x^2-4\zeta_x^3+\zeta_x^4},\qquad
\chi_{kl} = f_1(\alpha_{kl})\bar\zeta_x + f_2(\alpha_{kl})\bar\zeta_x^5 + f_3(\alpha_{kl})\bar\zeta_x^8,\qquad
f_i(\alpha) = \frac{\sum_{j=0}^3\phi_{i,j}\alpha^j}{1+\sum_{j=4}^6\phi_{i,j}\alpha^{j-3}} .
$$

The double sum runs over the 15 unordered pairs of the five terms (terms that vanish identically for a given parameter set are skipped). For the Mie pair it reproduces $C^2\varepsilon^2[x_0^{2\lambda_a}\Phi(2\lambda_a) - 2x_0^{\lambda_a+\lambda_r}\Phi(\lambda_a+\lambda_r) + x_0^{2\lambda_r}\Phi(2\lambda_r)]$.

### 4.3 Optional third order

$$
a_{3,kl} = -\varepsilon_{\mathrm{eff},kl}^3\,f_4(\alpha_{kl})\,\bar\zeta_x\exp\!\left[f_5(\alpha_{kl})\bar\zeta_x + f_6(\alpha_{kl})\bar\zeta_x^2\right].
$$

The coefficients $\phi_{i,j}$ of $\chi$ and $a_3$ were regressed for Mie fluids and are used here through the $\alpha$-hypothesis, evaluated with the effective $\varepsilon_\mathrm{eff}$, $\sigma_\mathrm{eff}$ and $\alpha$ of the IMP. Jervell et al. showed this hypothesis can fail for potentials whose force profile cannot be mimicked by a Mie potential; the IMP keeps a single minimum and a monotone force, but the difference between `order = 2` and `order = 3` is a useful indicator of this model uncertainty.

---

## 5. Chain term

### 5.1 TPT1 with contact at the effective diameter

$$
\frac{A^\mathrm{chain}}{Nk_\mathrm{B}T} = -\sum_i x_i(m_i-1)\ln y_{ii}(\sigma_{\mathrm{eff},ii}),\qquad y(r) = g(r)\,e^{\beta u(r)} .
$$

The bond length is taken as $\sigma_\mathrm{eff}$ for two reasons. First, $u(\sigma_\mathrm{eff}) = 0$ implies $y = g$ there, so the cavity function of the brief is available from the radial distribution function expansion. Second, $\lim_{\rho\to0}g(r_c) = 1$ requires $u(r_c) = 0$ (Jervell et al.). With zero moments $\sigma_\mathrm{eff} = \sigma$ and SAFT-VR Mie is recovered; with multipoles the bond length shrinks slightly and becomes temperature dependent, consistently with the effective-potential picture.

The contact value is expanded as $g = g^\mathrm{HS} + \tau g_1 + \tau^2 g_2$ with $\tau = \beta\varepsilon_\mathrm{eff}$ and then resummed exponentially, which keeps $g$ positive,

$$
\ln y_{ii}(\sigma_\mathrm{eff}) = \ln g^\mathrm{HS}_d(\sigma_\mathrm{eff}) + \frac{\tau g_1 + \tau^2 g_2}{g^\mathrm{HS}_d(\sigma_\mathrm{eff})},
$$

where the hard-sphere value away from contact is Boublík's expansion in $x_\mathrm{eff}$,

$$
g^\mathrm{HS}_d(x_\mathrm{eff}\,d) = \exp\!\left(k_0 + k_1x_\mathrm{eff} + k_2x_\mathrm{eff}^2 + k_3x_\mathrm{eff}^3\right),
$$

$$
k_0 = -\ln(1-\zeta_x) + \frac{42\zeta_x-39\zeta_x^2+9\zeta_x^3-2\zeta_x^4}{6(1-\zeta_x)^3},\quad
k_1 = \frac{\zeta_x^4+6\zeta_x^2-12\zeta_x}{2(1-\zeta_x)^3},\quad
k_2 = -\frac{3\zeta_x^2}{8(1-\zeta_x)^2},\quad
k_3 = \frac{-\zeta_x^4+3\zeta_x^2+3\zeta_x}{6(1-\zeta_x)^3} .
$$

### 5.2 First-order contact value

Write the compressibility factor of a one-component fluid of spheres in two ways. The thermodynamic route with the expansion of Section 4 gives

$$
Z = 1 + \rho\frac{\partial\tilde a^\mathrm{HS}}{\partial\rho} + \beta\rho\frac{\partial a_1}{\partial\rho} + \beta^2\rho\frac{\partial a_2}{\partial\rho} + \dots
$$

The virial route, $Z = 1 - \frac{2\pi\beta\rho}{3}\int_0^\infty u'(r)g(r)r^3dr$, split at $\sigma_\mathrm{eff}$ with $g = y\,e^{-\beta u}$ and the Barker–Henderson step of $e^{-\beta u}$ at $d$ in the repulsive region, gives

$$
Z = 1 + 4\eta\,g(d) - \frac{2\pi\beta\rho}{3}\int_{\sigma_\mathrm{eff}}^\infty u'(r)g(r)r^3dr,\qquad \eta = \frac{\pi\rho d^3}{6}.
$$

For the Sutherland sum $u'(r) = -r^{-1}\sum_n\lambda_nc_n(\sigma/r)^{\lambda_n}$, so the tail term is $+\frac{2\pi\beta\rho}{3}\sum_n\lambda_nc_n\int_{\sigma_\mathrm{eff}}^\infty(\sigma/r)^{\lambda_n}g\,r^2dr$. Equating the $O(\beta)$ terms, with $g\simeq g^\mathrm{HS}$ inside the tail integral and $g(d)\to g^\mathrm{HS} + \beta\varepsilon_\mathrm{eff}g_1$ in the contact term,

$$
4\eta\,\varepsilon_\mathrm{eff}\,g_1 = \rho\frac{\partial a_1}{\partial\rho} - \frac{2\pi\rho}{3}\sum_n\lambda_nc_n\,\mathcal I(\lambda_n)
\quad\Longrightarrow\quad
g_1 = \frac{3}{2\pi\varepsilon_\mathrm{eff}d^3}\frac{\partial a_1}{\partial\rho} - \frac{1}{\varepsilon_\mathrm{eff}d^3}\sum_n\lambda_nc_n\,\mathcal I(\lambda_n).
$$

Inserting $\mathcal I = -d^3x_0^\lambda\Phi$ and $a_1 = -2\pi\rho_sd^3S_1$ with $S_1 = \sum_nc_nx_0^{\lambda_n}\Phi(\lambda_n)$,

$$
\boxed{\,g_1 = \frac{1}{\varepsilon_\mathrm{eff}}\left[\sum_n\lambda_nc_n\,x_0^{\lambda_n}\Phi(\lambda_n) - 3\frac{\partial(\rho_sS_1)}{\partial\rho_s}\right]\,}
$$

Two remarks on the SAFT-VR Sum paper. The tail term enters $g_1$ with a minus sign, as in their eq. (29) and in Lafitte's $g_1$ for Mie fluids; the plus sign printed in their eq. (32) is inconsistent with eq. (29), and only the minus sign reduces to Clapeyron's SAFT-VR Mie chain term, which the implementation reproduces to machine precision. Also, when $u$ is written as $\varepsilon\sum_kC_k(\sigma/r)^{\lambda_k}$ with dimensionless $C_k$ while $g_n$ is scaled by $\varepsilon_\mathrm{eff}^n$, the tail term carries a factor $\varepsilon/\varepsilon_\mathrm{eff}$; writing $c_n = \varepsilon C_n$ in kelvin, as done here, makes it explicit.

In the mixture the like-pair relation is applied at the mixture density, i.e. $a_{1,ii}$, $S_{1,ii}$ and $\Phi$ are evaluated with $\zeta_x$ of the mixture, as in SAFT-VR Mie.

### 5.3 Second-order contact value

At $O(\beta^2)$ the tail integral contains the first-order correction of $g(r)$, which the macroscopic compressibility approximation writes as $\beta\varepsilon_\mathrm{eff}g_1(r) \simeq -\beta K^\mathrm{HS}g^\mathrm{HS}(r)u(r)$. The tail term becomes $-\frac{2\pi\beta^2\rho K^\mathrm{HS}}{3}\sum_{n,m}\lambda_nc_nc_m\,\mathcal I(\lambda_n+\lambda_m)$, and matching with $\beta^2\rho\,\partial a_2^\mathrm{MCA}/\partial\rho$ gives

$$
g_2^\mathrm{MCA} = \frac{3}{2\pi\varepsilon_\mathrm{eff}^2d^3}\frac{\partial a_2^\mathrm{MCA}}{\partial\rho} + \frac{K^\mathrm{HS}}{\varepsilon_\mathrm{eff}^2d^3}\sum_{n,m}\lambda_nc_nc_m\,\mathcal I(\lambda_n+\lambda_m).
$$

With $a_2^\mathrm{MCA} = a_2/(1+\chi) = \pi\rho_sK^\mathrm{HS}d^3S_2$ and $S_2 = \sum_{n,m}c_nc_mx_0^{\lambda_n+\lambda_m}\Phi(\lambda_n+\lambda_m)$,

$$
\boxed{\,g_2^\mathrm{MCA} = \frac{1}{\varepsilon_\mathrm{eff}^2}\left\{\frac{3}{2}\left[\rho_s\frac{\partial K^\mathrm{HS}}{\partial\rho_s}S_2 + K^\mathrm{HS}\frac{\partial(\rho_sS_2)}{\partial\rho_s}\right] - K^\mathrm{HS}\sum_{n,m}\lambda_nc_nc_m\,x_0^{\lambda_n+\lambda_m}\Phi(\lambda_n+\lambda_m)\right\}\,}
$$

(for $n\ne m$ the ordered pairs combine into $(\lambda_n+\lambda_m)c_nc_m$). The empirical correction of Lafitte et al. is applied with the effective quantities,

$$
g_2 = (1+\gamma_c)\,g_2^\mathrm{MCA},\qquad
\gamma_c = \phi_{7,0}\left[1-\tanh\!\big(\phi_{7,1}(\phi_{7,2}-\alpha)\big)\right]\bar\zeta_x\,\big(e^{\beta\varepsilon_\mathrm{eff}}-1\big)\,e^{\phi_{7,3}\bar\zeta_x+\phi_{7,4}\bar\zeta_x^2},
$$

with $\phi_7 = (10, 10, 0.57, -6.7, -8)$. The explicit minus sign inside $1-\tanh$ corresponds to Lafitte's $-\tanh(\cdot)+1$.

### 5.4 Density derivatives

At constant $T$ and composition every density dependence enters through explicit factors of $\rho_s$ and through $\zeta_x = \rho_s\frac{\pi}{6}\sum x_sx_sd^3$, so $\partial(\rho_sf)/\partial\rho_s = f + \zeta_x f'(\zeta_x)$. With $g_c(\zeta) = (1-\zeta/2)/(1-\zeta)^3$, $g_c'(\zeta) = (5/2-\zeta)/(1-\zeta)^4$, $h(\zeta) = 9\zeta(1+\zeta)/[2(1-\zeta)^3]$ and $h'(\zeta) = \frac92(1+4\zeta+\zeta^2)/(1-\zeta)^4$:

$$
\frac{\partial(\rho_sa^S_1)}{\partial\rho_s} = -\frac{g_c(\zeta_\mathrm{eff}) + \zeta_x\,\zeta_\mathrm{eff}'(\zeta_x)\,g_c'(\zeta_\mathrm{eff})}{\lambda-3},\qquad
\frac{\partial(\rho_sB)}{\partial\rho_s} = B + \zeta_x\left[I_\lambda g_c'(\zeta_x) - J_\lambda h'(\zeta_x)\right],\qquad
\rho_s\frac{\partial K^\mathrm{HS}}{\partial\rho_s} = \zeta_x\frac{dK^\mathrm{HS}}{d\zeta_x}.
$$

These are the `_fdf` kernels of the code; AD verifies them against finite differences through the pressure and chemical potentials.

---

## 6. Residual Helmholtz energy

$$
\boxed{\,\frac{A^\mathrm{res}}{Nk_\mathrm{B}T} = \bar m\,\tilde a^\mathrm{HS} + \frac{\bar m}{T}\sum_{k,l}x_{s,k}x_{s,l}a_{1,kl} + \frac{\bar m}{T^2}\sum_{k,l}x_{s,k}x_{s,l}a_{2,kl}\left[+\frac{\bar m}{T^3}\sum_{k,l}x_{s,k}x_{s,l}a_{3,kl}\right] - \sum_ix_i(m_i-1)\ln y_{ii}(\sigma_{\mathrm{eff},ii})\,}
$$

with mixing entering in four places: the segment fractions $x_{s,i}$ and density $\rho_s$; the BMCSL moments $\zeta_\ell$; the one-fluid $\zeta_x$ (over $d_{kl}$) and $\bar\zeta_x$ (over $\sigma_{\mathrm{eff},kl}$); and the pair potentials $u_{kl}$ (Mie combining rules plus parameter-free multipolar cross coefficients), which determine $\sigma_{\mathrm{eff},kl}$, $\varepsilon_{\mathrm{eff},kl}$ and $\alpha_{kl}$ pair by pair.

---

## 7. Temperature dependence and automatic differentiation

The temperature enters along the chain

$$
T \to c_n(T) \to \left\{\sigma_\mathrm{eff},\,r_\mathrm{min}\right\}\ \text{(implicit)} \to \left\{\varepsilon_\mathrm{eff},\,\alpha,\,x_\mathrm{eff}\right\} \to d(T)\ \text{(quadrature with $T$ in integrand and limits)} \to \left\{\zeta_\ell,\,\zeta_x,\,\bar\zeta_x,\,K^\mathrm{HS}\right\} \to A^\mathrm{res},
$$

plus the explicit factors $\beta$, $\tau$ and $e^{\beta\varepsilon_\mathrm{eff}}-1$. The implementation keeps every link differentiable with ForwardDiff.jl in four ways. Arrays are typed with `Base.promote_eltype(model,T)`, so no dual is ever truncated to `Float64`. The mask of active Sutherland terms is decided on the (Float64) parameters and never by comparing dual values. The implicit solves iterate Newton on the primal value and then take three additional Newton steps at the converged root; each step propagates one further order of dual parts to the implicit-function-theorem derivative, so nested duals (heat capacities, speed of sound, Joule–Thomson coefficients) are exact. Finally the quadrature for $d(T)$ uses fixed rules whose nodes depend smoothly on $T$ through $\theta$ and the integration limits, with the same switch between the Laguerre and Legendre branches at $\theta = 1$ as `d_vrmie`.

A thermodynamic consequence worth checking independently of the code is the Clapeyron equation, $dp_\mathrm{sat}/dT = \Delta h_\mathrm{vap}/(T\Delta v)$. It holds only if the enthalpy contains the $\partial u/\partial T$ contributions of Section 1.1, and it is satisfied to $4\times10^{-11}$ for quadrupolar benzene (Section 9).

---

## 8. Equations and code

| Quantity | Equation(s) | Function in `SAFTVRIMP.jl` |
|---|---|---|
| $c_n(T)$, $\lambda_n$, term mask | §1.3 | `imp_polar_constants`, `imp_sutherland`, `sutherland_terms` |
| $\sigma_\mathrm{eff}$, $\varepsilon_\mathrm{eff}$, $\alpha$ | §2 | `imp_logroot`, `imp_effective` |
| $d(T)$ | §3 | `d_imp`, `d_imp_cut` |
| pair data, $\zeta_\ell$, $\zeta_x$, $\bar\zeta_x$, $\rho_s$ | §3 | `imp_pairdata`, `imp_ζ_X_σ3`, `data` |
| $\tilde a^\mathrm{HS}$ | §3 | `a_hs` |
| $a^S_1$, $B$, $K^\mathrm{HS}$, $g^\mathrm{HS}_d$, $f_i$ and density derivatives | §4, §5.4 | `imp_aS1_fdf`, `imp_B_fdf`, `imp_KHS_f_ρdf`, `imp_gHS`, `imp_f123456` |
| $a_1$, $a_2$, $a_3$, $g_1$, $g_2$, $\ln y$ | §4, §5 | `imp_dispchain` (single fused pass) |
| $A^\mathrm{mono}$, $A^\mathrm{chain}$, $A^\mathrm{res}$ | §4, §5, §6 | `a_mono`, `a_disp`, `a_chain`, `a_res` |

---

## 9. Numerical verification

All checks below are in `test/test_SAFTVRIMP.jl` (97 passing tests).

With all moments zero and `order = 3`, the model reproduces Clapeyron's `SAFTVRMie` for n-hexane and n-hexane + n-decane at liquid, vapour and supercritical states, including the Gauss–Legendre branch of $d(T)$ at 1200 K: residual Helmholtz energies, pressures and residual chemical potentials agree to better than $10^{-13}$ (relative), and the normal boiling point agrees to all printed digits (341.86441622 K). This checks the reference, both perturbation terms, the chain term (and the sign of $g_1$) and the mixing rules.

For the dipolar Lennard-Jones fluid with $\mu^{*2} = 2$, $\sigma_\mathrm{eff}$, $\varepsilon_\mathrm{eff}$, $d$ and $A^\mathrm{res}$ equal the analytic Keesom mapping of Section 2 to $\le 5\times10^{-15}$ between 120 and 3000 K, and the first and second temperature derivatives of $\varepsilon_\mathrm{eff}$ from nested ForwardDiff agree with the analytic ones to 13 digits.

To exercise every Sutherland slot, a binary of chains ($m = 1.5$ and $2$) was constructed in which each pair has exactly one multipolar term sharing its dispersion exponent: Keesom plus induction at $\lambda_a = 6$ for A–A, the dipole–quadrupole cross term at $\lambda_a = 8$ for A–B (explicit unlike parameters) and quadrupole–quadrupole at $\lambda_a = 10$ for B–B. Each pair then maps analytically onto a Mie pair, and the model agrees with `SAFTVRMie` built from the mapped parameters to $\le 6.5\times10^{-14}$ in $A^\mathrm{res}$, $p$ and $\mu^\mathrm{res}$ at 250–1500 K.

For a polar chain with all five terms active (μ = 2 D, Q = 4 D·Å, α = 11.9 Å³, mixed with n-decane, `order = 2`), ForwardDiff derivatives $\partial p/\partial T$, $\partial p/\partial V$, $\partial p/\partial z_1$ and $\partial^2A^\mathrm{res}/\partial T^2$ agree with Richardson-extrapolated finite differences to better than $4\times10^{-11}$ (the finite-difference resolution), $C_v$ from AD agrees with $\partial U/\partial T$ to $6\times10^{-12}$, the gradient of $p(V,T,\mathbf z)$ satisfies the Euler relation of an intensive property, and $dx_\mathrm{eff}/dT$ and $d\varepsilon_\mathrm{eff}/dT$ match the implicit-function and envelope theorems to 13 digits.

---

## 10. Predictions with the Müller–Gelb parameters

The parameters of Müller and Gelb were fitted to molecular dynamics of a potential cut and shifted at $5\sigma$ (liquid densities for the polar fluids, the Lennard-Jones corresponding-states critical point for the non-polar ones), not to this equation of state, so the numbers below test plausibility rather than accuracy.

| Fluid | $T_c$ (order 2) | $T_c$ (order 3) | $T_c$ (order 2, moments off) | exp. |
|---|---|---|---|---|
| benzene (Q) | 571.5 K | 553.5 K | 488.2 K | 562.0 K |
| CO₂ (Q) | 341.7 K | 331.0 K | 292.6 K | 304.1 K |
| 1,2-dichloroethane (μ) | 599.2 K | 578.9 K | 568.6 K | 561.6 K |
| cyclohexane | 560.6 K | 540.6 K | 560.6 K | 553.6 K |
| n-octane | 588.2 K | 567.2 K | 588.2 K | 568.7 K |

The third-order model reproduces the Lennard-Jones critical temperature ($T_c^* = 1.3122$; the non-polar fits used $T_c^* = 1.31$), and the second-order truncation raises it by about 3.7 %. For the dipolar Lennard-Jones fluid of their Fig. 3 ($\mu^{*2} = 2$) the equation of state gives $T_c^* = 1.833$ (order 3), above the highest two-phase simulation state shown there ($T^* = 1.7$). With pure-component parameters only, 1,2-dichloroethane + cyclohexane is predicted azeotropic at 1 atm, as observed, but the pure boiling points are far too low (329 K and 310 K), which is the poor vapour-pressure behaviour that Müller and Gelb reported for these parameters. For n-hexane with SAFT-VR Mie parameters the second-order model gives $T_b = 341.88$ K (exp. ≈ 341.9 K) and $T_c = 530.4$ K (order 3: 515.1 K; exp. ≈ 507.8 K).

For quantitative work, $\varepsilon$, $\sigma$ and $\lambda_r$ (and $m$ for chains) should be refitted for the chosen perturbation order with the moments fixed at experimental values, as is standard for polar SAFT variants.

---

## 11. Scope and possible extensions

The model has no association term. Adding `a_assoc` is mechanical in Clapeyron, but requires choosing how the SAFT-VR Mie association kernel, correlated in $T/\varepsilon$ and $\rho_s\sigma^3$, should be scaled for the IMP ($\varepsilon_\mathrm{eff}$ and $\sigma_\mathrm{eff}$ are the natural candidates). The empirical corrections $\chi$, $a_3$ and $\gamma_c$ inherit the Mie-based $\alpha$-hypothesis. The uniform distribution of moments over segments is the simplest choice for homonuclear chains; a heteronuclear, group-contribution version (SAFT-γ IMP) would place the moments on specific segments. Higher-order IMP cumulants ($\propto\beta^2$) and polarizability beyond the Debye term could be added as further Sutherland terms with $c_n \propto T^{-2}$ without changing the perturbation machinery.

## References

1. E. A. Müller, L. D. Gelb, *Ind. Eng. Chem. Res.* **42**, 4123 (2003). doi:10.1021/ie030033y
2. T. Lafitte et al., *J. Chem. Phys.* **139**, 154504 (2013). doi:10.1063/1.4819786
3. V. G. Jervell, T. W. Maltby, A. Aasen, M. Hammer, Ø. Wilhelmsen, *J. Chem. Phys.* **164**, 114101 (2026). doi:10.1063/5.0317322
4. A. Gil-Villegas et al., *J. Chem. Phys.* **106**, 4168 (1997).
5. A. Aasen et al., *J. Chem. Phys.* **151**, 064508 (2019).
6. J. A. Barker, D. Henderson, *Rev. Mod. Phys.* **48**, 587 (1976).
7. P. J. Walker, H.-W. Yew, A. Riedemann, *Ind. Eng. Chem. Res.* **61**, 7130 (2022).
