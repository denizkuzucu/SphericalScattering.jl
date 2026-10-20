
# =====================================================================================
#  Fields of a multilayer dielectric sphere under a plane wave, in rescaled quantities.
#
#  Same series and same boundary conditions as in `scatteredLayeredInterior.jl`, but the
#  Riccati-Bessel functions are never evaluated on their own: Ĵₙ underflows and Ĥₙ⁽²⁾
#  overflows once the order n exceeds the argument by far, which is what happens at the
#  inner interfaces of an electrically large sphere. Following
#
#      L. Guo and J. Shen, "Internal and external-fields of a multilayered sphere
#      illuminated by the shaped beam: Rescaled quantities for numerical calculation",
#      J. Quant. Spectrosc. Radiat. Transf. 250, 107004 (2020),
#
#  only their logarithmic derivatives and ratios are used (adapted to the exp(+jωt)
#  convention and the Ĥₙ⁽²⁾ of this package):
#
#      D⁽¹⁾ₙ(z) = Ĵₙ′(z)/Ĵₙ(z)            D⁽³⁾ₙ(z) = Ĥₙ⁽²⁾′(z)/Ĥₙ⁽²⁾(z)
#      Ĵₙ(z₁)/Ĵₙ(z₂)                     Ĥₙ⁽²⁾(z₁)/Ĥₙ⁽²⁾(z₂)
#
#  The radial function of layer i (i = 1 … N), between the radii aᵢ₋₁ and aᵢ, is written as
#
#      U⁽ⁱ⁾(kᵢr) = cᵢ Ĵₙ(kᵢr)/Ĵₙ(kᵢaᵢ) + aᵢ Ĥₙ⁽²⁾(kᵢr)/Ĥₙ⁽²⁾(kᵢaᵢ₋₁)
#
#  so that each part is referred to the interface where it is largest and both ratios stay
#  bounded inside the layer. The core holds the first part alone. In the background the
#  incident plane wave is added in closed form and only the scattered part is expanded,
#
#      U⁽ᴺ⁺¹⁾(kr) = s Ĥₙ⁽²⁾(kr)/Ĥₙ⁽²⁾(kaₙ).
#
#  The outward pass propagates U′/U through the interfaces, the inward pass then fixes the
#  amplitudes cᵢ and aᵢ. Quantities that underflow only ever multiply, such that orders which
#  do not reach an inner layer drop out by themselves.
# =====================================================================================


"""
    riccatiLogDerivatives(z, nmax::Int)

Logarithmic derivatives ``D⁽¹⁾ₙ(z) = Ĵₙ′(z)/Ĵₙ(z)`` and ``D⁽³⁾ₙ(z) = Ĥₙ⁽²⁾′(z)/Ĥₙ⁽²⁾(z)`` of the Riccati-Bessel
functions and the product ``Ĵₙ(z) Ĥₙ⁽²⁾(z)`` for the orders `n = 0 … nmax` (stored at the index `n + 1`).

``D⁽¹⁾ₙ`` is obtained by the downward recurrence, ``D⁽³⁾ₙ`` and the product by upward ones.
"""
function riccatiLogDerivatives(z::Complex{T}, nmax::Int) where {T}

    # --- downward recurrence for D⁽¹⁾, started well above the orders needed and above |z|
    nstart = max(nmax, ceil(Int, abs(z)))
    nstart += ceil(Int, 8 * cbrt(nstart)) + 15

    D1 = Vector{Complex{T}}(undef, nmax + 1)
    D = zero(z)

    for n in nstart:-1:1
        D = n / z - 1 / (D + n / z)      # D⁽¹⁾ₙ₋₁ from D⁽¹⁾ₙ
        n <= nmax + 1 && (D1[n] = D)
    end

    # --- upward recurrences for ĴₙĤₙ⁽²⁾ and D⁽³⁾, with Ĵ₀Ĥ₀⁽²⁾ = (1 - exp(-2jz))/2 and D⁽³⁾₀ = -j
    D3 = Vector{Complex{T}}(undef, nmax + 1)
    ĴĤ = Vector{Complex{T}}(undef, nmax + 1)

    D3[1] = -im
    ĴĤ[1] = -expm1(-2im * z) / 2

    for n in 1:nmax
        ĴĤ[n + 1] = ĴĤ[n] * (n / z - D3[n]) / (D1[n + 1] + n / z)
        D3[n + 1] = D1[n + 1] - im / ĴĤ[n + 1]   # from the Wronskian ĴₙĤₙ⁽²⁾′ - Ĵₙ′Ĥₙ⁽²⁾ = -j
    end

    return D1, D3, ĴĤ
end


"""
    sineRatio(z₁, z₂)

``sin(z₁)/sin(z₂) = Ĵ₀(z₁)/Ĵ₀(z₂)``, evaluated without overflow for arguments with a large imaginary part.
"""
function sineRatio(z₁, z₂)
    if imag(z₂) <= 0
        return exp(im * (z₁ - z₂)) * expm1(-2im * z₁) / expm1(-2im * z₂)
    else
        return exp(-im * (z₁ - z₂)) * expm1(2im * z₁) / expm1(2im * z₂)
    end
end


"""
    riccatiBesselRatios(z₁, D1z₁, z₂, D1z₂, nmax::Int)

Ratios ``Ĵₙ(z₁)/Ĵₙ(z₂)`` for `n = 0 … nmax` (index `n + 1`), from the logarithmic derivatives ``D⁽¹⁾`` at
both arguments via ``Ĵₙ(z) = Ĵₙ₋₁(z) / (D⁽¹⁾ₙ(z) + n/z)``.
"""
function riccatiBesselRatios(z₁, D1z₁, z₂, D1z₂, nmax::Int)

    Q = Vector{eltype(D1z₁)}(undef, nmax + 1)
    Q[1] = sineRatio(z₁, z₂)

    for n in 1:nmax
        Q[n + 1] = Q[n] * (D1z₂[n + 1] + n / z₂) / (D1z₁[n + 1] + n / z₁)
    end

    return Q
end


"""
    riccatiHankelRatios(z₁, D3z₁, z₂, D3z₂, nmax::Int)

Ratios ``Ĥₙ⁽²⁾(z₁)/Ĥₙ⁽²⁾(z₂)`` for `n = 0 … nmax` (index `n + 1`), from the logarithmic derivatives ``D⁽³⁾``
at both arguments via ``Ĥₙ⁽²⁾(z) = Ĥₙ₋₁⁽²⁾(z) (n/z - D⁽³⁾ₙ₋₁(z))``.
"""
function riccatiHankelRatios(z₁, D3z₁, z₂, D3z₂, nmax::Int)

    Q = Vector{eltype(D3z₁)}(undef, nmax + 1)
    Q[1] = exp(-im * (z₁ - z₂))

    for n in 1:nmax
        Q[n + 1] = Q[n] * (n / z₁ - D3z₁[n]) / (n / z₂ - D3z₂[n])
    end

    return Q
end


"""
    RescaledLayerCoefficients

Rescaled expansion coefficients of a [`LayeredSphere`](@ref) under a plane wave for the orders `1 … nmax`,
see [`rescaledLayerCoefficients`](@ref).

`cA[i, n]`, `aA[i, n]` weight ``Ĵₙ(kᵢr)/Ĵₙ(kᵢaᵢ)`` and ``Ĥₙ⁽²⁾(kᵢr)/Ĥₙ⁽²⁾(kᵢaᵢ₋₁)`` in the TM radial function of
layer `i`, `cF`, `aF` those of the TE one. `sA[n]`, `sF[n]` weight ``Ĥₙ⁽²⁾(kr)/Ĥₙ⁽²⁾(ka_N)`` in the scattered
field of the background. `aₙ`, `bₙ` are the conventional scattering coefficients, as returned by `scatterCoeff`.
"""
struct RescaledLayerCoefficients{C,R}
    nmax::Int
    cA::Matrix{C}
    aA::Matrix{C}
    cF::Matrix{C}
    aF::Matrix{C}
    sA::Vector{C}
    sF::Vector{C}
    aₙ::Vector{C}
    bₙ::Vector{C}
    radii::Vector{R}
    k::Vector{C}
    ε::Vector{C}
    μ::Vector{C}
    D1out::Vector{Vector{C}}   # D⁽¹⁾ at kᵢaᵢ,   i = 1 … N
    D3in::Vector{Vector{C}}    # D⁽³⁾ at kᵢaᵢ₋₁, i = 2 … N+1 (the entry i = 1 is empty)
end


"""
    seriesTruncation(sphere::LayeredSphere, excitation::PlaneWave)

Number of orders of the series: the largest size parameter ``|kᵢ a|`` found on either side of any interface,
enlarged by the transition region of the Bessel functions.
"""
function seriesTruncation(sphere::LayeredSphere, excitation::PlaneWave)

    ω = 2π * excitation.frequency

    radii = layerRadii(sphere)
    fillings = layerFillings(sphere)

    N = length(radii)

    x = abs(ω * sqrt(excitation.embedding.ε * excitation.embedding.μ) * radii[N])

    for i in 1:N
        kᵢ = ω * sqrt(fillings[i].ε * fillings[i].μ)

        x = max(x, abs(kᵢ * radii[i]))
        i > 1 && (x = max(x, abs(kᵢ * radii[i - 1])))
    end

    return ceil(Int, x + 11 * cbrt(x) + 1) + 15
end


"""
    rescaledLayerCoefficients(sphere::LayeredSphere, excitation::PlaneWave; parameter::Parameter=Parameter())

Rescaled coefficients of every layer for a plane wave travelling in +z-direction with polarization in
x-direction, see [`RescaledLayerCoefficients`](@ref).

The number of orders is determined from the size parameters of the layers, unless `parameter.nmax` is positive.
The coefficients do not depend on the observation point; they can be passed as `coefficients` to
[`totalfieldRescaled`](@ref).
"""
function rescaledLayerCoefficients(sphere::LayeredSphere, excitation::PlaneWave; parameter::Parameter=Parameter())

    T = typeof(excitation.frequency)
    ω = 2π * excitation.frequency

    radii = layerRadii(sphere)
    fillings = layerFillings(sphere)

    N = length(radii) # number of dielectric layers: the core and the shells

    CT = promote_type(Complex{T}, typeof(excitation.embedding.ε), typeof(first(fillings).ε), typeof(last(fillings).ε))

    nmax = parameter.nmax > 0 ? parameter.nmax : seriesTruncation(sphere, excitation)

    # media of the N layers (inner → outer), the background is appended as layer N+1
    ε = CT[i <= N ? fillings[i].ε : excitation.embedding.ε for i in 1:(N + 1)]
    μ = CT[i <= N ? fillings[i].μ : excitation.embedding.μ for i in 1:(N + 1)]
    k = CT[ω * sqrt(ε[i] * μ[i]) for i in 1:(N + 1)]

    # --- logarithmic derivatives and ratios at the interfaces
    D1out = Vector{Vector{CT}}(undef, N)       # at kᵢaᵢ
    D3out = Vector{Vector{CT}}(undef, N)
    D1in = Vector{Vector{CT}}(undef, N + 1)    # at kᵢaᵢ₋₁
    D3in = Vector{Vector{CT}}(undef, N + 1)
    QJ = Vector{Vector{CT}}(undef, N)          # Ĵₙ(kᵢaᵢ₋₁)/Ĵₙ(kᵢaᵢ)
    QH = Vector{Vector{CT}}(undef, N)          # Ĥₙ(kᵢaᵢ)/Ĥₙ(kᵢaᵢ₋₁)

    D1in[1] = CT[]
    D3in[1] = CT[]

    for i in 1:N
        D1out[i], D3out[i], _ = riccatiLogDerivatives(k[i] * radii[i], nmax)

        if i > 1
            D1in[i], D3in[i], _ = riccatiLogDerivatives(k[i] * radii[i - 1], nmax)

            QJ[i] = riccatiBesselRatios(k[i] * radii[i - 1], D1in[i], k[i] * radii[i], D1out[i], nmax)
            QH[i] = riccatiHankelRatios(k[i] * radii[i], D3out[i], k[i] * radii[i - 1], D3in[i], nmax)
        end
    end

    # background at the outermost interface, where Ĵₙ itself is needed for the incident wave
    xₑ = k[N + 1] * radii[N]
    D1in[N + 1], D3in[N + 1], ĴĤ = riccatiLogDerivatives(xₑ, nmax)

    Ĵ = Vector{CT}(undef, nmax + 1)
    Ĵ[1] = sin(xₑ)
    for n in 1:nmax
        Ĵ[n + 1] = Ĵ[n] / (D1in[N + 1][n + 1] + n / xₑ)
    end

    η = sqrt(μ[N + 1] / ε[N + 1])

    cA = zeros(CT, N, nmax)
    aA = zeros(CT, N, nmax)
    cF = zeros(CT, N, nmax)
    aF = zeros(CT, N, nmax)
    sA = zeros(CT, nmax)
    sF = zeros(CT, nmax)
    aₙ = zeros(CT, nmax)
    bₙ = zeros(CT, nmax)

    t = zeros(CT, N)   # aᵢ/cᵢ
    u = zeros(CT, N)   # U⁽ⁱ⁾(kᵢaᵢ₋₁)/cᵢ
    w = zeros(CT, N)   # U⁽ⁱ⁾(kᵢaᵢ)/cᵢ

    for n in 1:nmax

        pF = im^(-T(n)) * (2 * n + 1) / (n * (n + 1))

        # TM (from Aᵣ): U/μ and U′/k are continuous;  TE (from Fᵣ): U/ε and U′/k are continuous
        for (m, c, a, s, A₀, TM) in ((μ, cA, aA, sA, pF, true), (ε, cF, aF, sF, pF / η, false))

            # --- outward pass: L = U′/U at the outer interface of each layer
            t[1] = 0
            w[1] = 1
            L = D1out[1][n + 1]

            for i in 2:N
                Lin = k[i] * m[i - 1] / (k[i - 1] * m[i]) * L

                t[i] = QJ[i][n + 1] * (D1in[i][n + 1] - Lin) / (Lin - D3in[i][n + 1])
                u[i] = QJ[i][n + 1] * (D1in[i][n + 1] - D3in[i][n + 1]) / (Lin - D3in[i][n + 1])
                w[i] = 1 + t[i] * QH[i][n + 1]

                L = (D1out[i][n + 1] + t[i] * QH[i][n + 1] * D3out[i][n + 1]) / w[i]
            end

            # --- background: incident Ĵₙ plus scattered Ĥₙ⁽²⁾ match the sphere at the outermost interface
            Lin = k[N + 1] * m[N] / (k[N] * m[N + 1]) * L

            D1ₑ = D1in[N + 1][n + 1]
            D3ₑ = D3in[N + 1][n + 1]

            ratio = (D1ₑ - Lin) / (Lin - D3ₑ)

            s[n] = A₀ * Ĵ[n + 1] * ratio
            Uₑ = A₀ * Ĵ[n + 1] * (D1ₑ - D3ₑ) / (Lin - D3ₑ)   # total radial function at the interface

            if TM
                aₙ[n] = pF * Ĵ[n + 1]^2 / ĴĤ[n + 1] * ratio
            else
                bₙ[n] = pF * Ĵ[n + 1]^2 / ĴĤ[n + 1] * ratio
            end

            # --- inward pass: amplitudes from the continuity of U/μ (TM) or U/ε (TE)
            c[N, n] = m[N] / m[N + 1] * Uₑ / w[N]

            for i in (N - 1):-1:1
                c[i, n] = m[i] / m[i + 1] * c[i + 1, n] * u[i + 1] / w[i]
            end

            for i in 2:N
                a[i, n] = t[i] * c[i, n]
            end
        end
    end

    return RescaledLayerCoefficients(nmax, cA, aA, cF, aF, sA, sF, aₙ, bₙ, Vector(radii), k, ε, μ, D1out, D3in)
end


"""
    totalfieldRescaled(sphere::LayeredSphere, excitation::PlaneWave, quantity::Field; parameter::Parameter=Parameter(), coefficients=nothing)

Total (incident plus scattered) electric or magnetic field of a layered dielectric sphere at the locations of
`quantity`, valid at any radius. It is computed in rescaled quantities, which stay within the floating-point
range for electrically large spheres with many layers, see [`rescaledLayerCoefficients`](@ref).

The points and the returned fields are in Cartesian coordinates. `coefficients` optionally takes the result
of [`rescaledLayerCoefficients`](@ref).
"""
function totalfieldRescaled(
    sphere::LayeredSphere,
    excitation::PlaneWave,
    quantity::Union{ElectricField,MagneticField};
    parameter::Parameter=Parameter(),
    coefficients=nothing,
)

    T = typeof(excitation.frequency)
    F = zeros(SVector{3,Complex{T}}, size(quantity.locations))

    coeffs = isnothing(coefficients) ? rescaledLayerCoefficients(sphere, excitation; parameter=parameter) : coefficients

    # --- rotate coordinates
    points = rotate(excitation, quantity.locations; inverse=true)

    p = progress(length(points))

    # --- compute field in Cartesian representation
    @tasks for ind in eachindex(points)
        F[ind] = totalfieldRescaled(sphere, excitation, points[ind], quantity, coeffs)
        next!(p)
    end
    finish!(p)

    # --- rotate resulting field
    rotate!(excitation, F; inverse=false)

    return F
end


"""
    totalfieldRescaled(sphere::LayeredSphere, excitation::PlaneWave, point, quantity::Field, coeffs::RescaledLayerCoefficients)

Total field at `point` for an incident plane wave travelling in +z-direction with E-field polarization in
x-direction.

The point and the returned field are in Cartesian coordinates.
"""
function totalfieldRescaled(
    sphere::LayeredSphere,
    excitation::PlaneWave,
    point,
    quantity::Union{ElectricField,MagneticField},
    coeffs::RescaledLayerCoefficients{CT},
) where {CT}

    T = typeof(excitation.frequency)
    ω = 2π * excitation.frequency
    E₀ = excitation.amplitude

    point_sph = cart2sph(point) # [r ϑ φ]
    r = point_sph[1]

    N = length(coeffs.radii)
    nmax = coeffs.nmax

    i = layer(sphere, r)

    k = coeffs.k[i]
    x = k * r

    # --- radial functions U_A, U_A′, U_F, U_F′ of all orders (derivatives with respect to kᵢr)
    D1x, D3x, _ = riccatiLogDerivatives(x, nmax)

    U_A = zeros(CT, nmax)
    dU_A = zeros(CT, nmax)
    U_F = zeros(CT, nmax)
    dU_F = zeros(CT, nmax)

    if i <= N
        qJ = riccatiBesselRatios(x, D1x, k * coeffs.radii[i], coeffs.D1out[i], nmax)

        for n in 1:nmax
            U_A[n] = coeffs.cA[i, n] * qJ[n + 1]
            U_F[n] = coeffs.cF[i, n] * qJ[n + 1]

            dU_A[n] = U_A[n] * D1x[n + 1]
            dU_F[n] = U_F[n] * D1x[n + 1]
        end
    end

    if i > 1
        qH = riccatiHankelRatios(x, D3x, k * coeffs.radii[i - 1], coeffs.D3in[i], nmax)

        for n in 1:nmax
            hA = (i <= N ? coeffs.aA[i, n] : coeffs.sA[n]) * qH[n + 1]
            hF = (i <= N ? coeffs.aF[i, n] : coeffs.sF[n]) * qH[n + 1]

            U_A[n] += hA
            U_F[n] += hF

            dU_A[n] += hA * D3x[n + 1]
            dU_F[n] += hF * D3x[n + 1]
        end
    end

    # --- sum the series
    sinϑ = abs(sin(point_sph[2]))  # note: theta only defined from from 0 to pi
    cosϑ = cos(point_sph[2])
    sinϕ = sin(point_sph[3])
    cosϕ = cos(point_sph[3])

    ωε = ω * coeffs.ε[i]
    ωμ = ω * coeffs.μ[i]

    Fr = zero(CT)
    Fϑ = zero(CT)
    Fϕ = zero(CT)

    # associated Legendre polynomials Pₙ¹(cosϑ) of the orders n - 1 and n
    Pₙ₋₁ = zero(T)
    Pₙ = -sinϑ

    for n in 1:nmax

        # πₙ = Pₙ¹/sinϑ and τₙ = dPₙ¹/dϑ, with their limits on the z-axis
        if abs(cosϑ) < 0.999999
            πₙ = Pₙ / sinϑ
            τₙ = (n * cosϑ * Pₙ - (n + 1) * Pₙ₋₁) / sinϑ
        elseif cosϑ > 0
            πₙ = -(n + T(1.0)) * n / T(2.0)
            τₙ = πₙ
        else
            πₙ = (n + T(1.0)) * n / T(2.0) * T(-1.0)^n
            τₙ = -πₙ
        end

        if quantity isa ElectricField
            Fr += cosϕ / (im * x^2) * n * (n + 1) * U_A[n] * Pₙ
            Fϑ -= cosϕ / r * (im / k * dU_A[n] * τₙ + U_F[n] / ωε * πₙ)
            Fϕ += sinϕ / r * (im / k * dU_A[n] * πₙ + U_F[n] / ωε * τₙ)
        else
            Fr += sinϕ / (im * x^2) * n * (n + 1) * U_F[n] * Pₙ
            Fϑ -= sinϕ / r * (U_A[n] / ωμ * πₙ + im / k * dU_F[n] * τₙ)
            Fϕ -= cosϕ / r * (U_A[n] / ωμ * τₙ + im / k * dU_F[n] * πₙ)
        end

        Pₙ₋₁, Pₙ = Pₙ, ((2 * n + 1) * cosϑ * Pₙ - (n + 1) * Pₙ₋₁) / n
    end

    F = convertSpherical2Cartesian(E₀ .* SVector{3,CT}(Fr, Fϑ, Fϕ), point_sph)

    # --- in the background only the scattered field has been expanded: add the incident plane wave
    if i == N + 1
        F += field(planeWave(; embedding=excitation.embedding, frequency=excitation.frequency, amplitude=E₀), point, quantity)
    end

    return F
end
