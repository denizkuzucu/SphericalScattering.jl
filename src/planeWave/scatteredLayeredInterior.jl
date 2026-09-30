
# =====================================================================================
#  Fields inside a multilayer dielectric sphere under a plane wave.
#  This file is calculates inner fields once aₙ and bₙ are known,and 
#  recover the expansion coefficients of every shell.
#
#  Representation used in layer `i` (i = 1 … N shells, i = N+1 the background), with
#  x = kᵢr and U(x) = Ĵₙ(x) + σ Ĥₙ⁽²⁾(x):
#
#      Aᵣ⁽ⁱ⁾ = (E₀cosφ/ω)   Σₙ αᵢ U_A(kᵢr) Pₙ¹(cosϑ)        TM
#      Fᵣ⁽ⁱ⁾ = (E₀sinφ/ω)   Σₙ βᵢ U_F(kᵢr) Pₙ¹(cosϑ)        TE
#
#  The core holds Ĵₙ alone (σ⁽¹⁾ = 0, regular at the origin); the background is incident
#  plus scattered, so α⁽ᴺ⁺¹⁾ = pₙ, σH⁽ᴺ⁺¹⁾ = aₙ/pₙ and β⁽ᴺ⁺¹⁾ = pₙ/η.
#
#  Continuity of the tangential fields at r = aᵢ, Jin (7.4.79)-(7.4.82), reads
#
#      U_A⁽ⁱ⁾/μᵢ = U_A⁽ⁱ⁺¹⁾/μᵢ₊₁      U_A⁽ⁱ⁾′/kᵢ = U_A⁽ⁱ⁺¹⁾′/kᵢ₊₁      (TM)
#      U_F⁽ⁱ⁾/εᵢ = U_F⁽ⁱ⁺¹⁾/εᵢ₊₁      U_F⁽ⁱ⁾′/kᵢ = U_F⁽ⁱ⁺¹⁾′/kᵢ₊₁      (TE)
#
#  The outward pass fixes every σ through the derivative conditions; the value conditions
#  then give the amplitudes directly, which is the downward pass implemented here.
#
#  The field evaluation cannot reuse `expansion`: that routine selects EITHER Ĵₙ (inside)
#  OR Ĥₙ⁽²⁾ (outside) by radius, whereas a shell needs both at once.
# =====================================================================================


"""
    LayerCoefficients

Expansion coefficients of one order `n` in every layer of a [`LayeredSphere`](@ref).
Index `i` runs `1 … N+1`, where `N+1` denotes the background.

`σH`, `σE` are the ratios of the ``Ĥₙ⁽²⁾`` to the ``Ĵₙ`` coefficient (TM and TE); `α`, `β`
are the absolute amplitudes of the TM and TE radial functions.
"""
struct LayerCoefficients{C}
    α::Vector{C}
    σH::Vector{C}
    β::Vector{C}
    σE::Vector{C}
    k::Vector{C}
    ε::Vector{C}
    μ::Vector{C}
end


"""
    layerCoefficients(sphere::LayeredSphere, excitation::PlaneWave, n::Int)

Coefficients of order `n` in every layer: the outward pass of Jin (7.4.83)-(7.4.86)
keeping each ratio, followed by the downward pass that turns those ratios into amplitudes.

Returns a [`LayerCoefficients`](@ref).
"""
function layerCoefficients(sphere::LayeredSphere{N,R,C}, excitation::PlaneWave, n::Int) where {N,R,C}

    T = typeof(excitation.frequency)
    ω = 2π * excitation.frequency

    CT = promote_type(C, typeof(excitation.embedding.ε), typeof(excitation.embedding.μ), Complex{T})

    radii = sphere.radii

    ε = CT[i <= N ? sphere.filling[i].ε : excitation.embedding.ε for i in 1:(N + 1)]
    μ = CT[i <= N ? sphere.filling[i].μ : excitation.embedding.μ for i in 1:(N + 1)]
    k = CT[ω * sqrt(ε[i] * μ[i]) for i in 1:(N + 1)]

    # --- outward pass: the ratios, as in scatterCoeff(::LayeredSphere, ...) but all kept
    σH = zeros(CT, N + 1)
    σE = zeros(CT, N + 1)          # σ⁽¹⁾ = 0: the core holds Ĵₙ alone

    for i in 1:N
        a = radii[i]

        Ĵ, dĴ = riccatiBessel(k[i] * a, n, T)
        Ĥ, dĤ = riccatiHankel2(k[i] * a, n, T)

        S = sqrt(μ[i + 1] * ε[i] / (ε[i + 1] * μ[i]))

        RH = S * (Ĵ + σH[i] * Ĥ) / (dĴ + σH[i] * dĤ)      # Jin (7.4.83)
        RE = (Ĵ + σE[i] * Ĥ) / (dĴ + σE[i] * dĤ) / S      # Jin (7.4.84)

        Ĵ, dĴ = riccatiBessel(k[i + 1] * a, n, T)
        Ĥ, dĤ = riccatiHankel2(k[i + 1] * a, n, T)

        σH[i + 1] = (RH * dĴ - Ĵ) / (Ĥ - RH * dĤ)         # Jin (7.4.85)
        σE[i + 1] = (RE * dĴ - Ĵ) / (Ĥ - RE * dĤ)         # Jin (7.4.86)
    end

    # --- downward pass: the amplitudes, from the value conditions
    pF = im^(-T(n)) * (2 * n + 1) / (n * (n + 1))
    η = sqrt(μ[N + 1] / ε[N + 1])

    α = zeros(CT, N + 1)
    β = zeros(CT, N + 1)

    α[N + 1] = pF                  # the incident Ĵₙ carries pₙ, the scattered Ĥₙ⁽²⁾ carries aₙ
    β[N + 1] = pF / η              # Fᵣ is written with H₀ = E₀/η

    for i in N:-1:1
        a = radii[i]

        Ĵo, _ = riccatiBessel(k[i + 1] * a, n, T)
        Ĥo, _ = riccatiHankel2(k[i + 1] * a, n, T)
        Ĵi, _ = riccatiBessel(k[i] * a, n, T)
        Ĥi, _ = riccatiHankel2(k[i] * a, n, T)

        α[i] = (μ[i] / μ[i + 1]) * α[i + 1] * (Ĵo + σH[i + 1] * Ĥo) / (Ĵi + σH[i] * Ĥi)
        β[i] = (ε[i] / ε[i + 1]) * β[i + 1] * (Ĵo + σE[i + 1] * Ĥo) / (Ĵi + σE[i] * Ĥi)
    end

    return LayerCoefficients{CT}(α, σH, β, σE, k, ε, μ)
end


"""
    layerCoefficientCache(sphere::LayeredSphere, excitation::PlaneWave, nmax::Int)

[`layerCoefficients`](@ref) for the orders `1:nmax`. The coefficients do not depend on the
observation point, so passing this as `coefficients` to [`totalfield`](@ref) avoids
repeating the recursion at every point.
"""
function layerCoefficientCache(sphere::LayeredSphere, excitation::PlaneWave, nmax::Int)
    return [layerCoefficients(sphere, excitation, n) for n in 1:nmax]
end


"""
    layerRadialFunctions(c::LayerCoefficients, i, r, n, ::Type{T})

``U_A``, ``U_A′``, ``U_F``, ``U_F′`` of layer `i` at radius `r`, derivatives taken with
respect to ``kᵢr``.
"""
function layerRadialFunctions(c::LayerCoefficients, i::Int, r, n::Int, ::Type{T}) where {T}

    x = c.k[i] * r

    Ĵ, dĴ = riccatiBessel(x, n, T)
    Ĥ, dĤ = riccatiHankel2(x, n, T)

    U_A = c.α[i] * (Ĵ + c.σH[i] * Ĥ)
    dU_A = c.α[i] * (dĴ + c.σH[i] * dĤ)

    U_F = c.β[i] * (Ĵ + c.σE[i] * Ĥ)
    dU_F = c.β[i] * (dĴ + c.σE[i] * dĤ)

    return U_A, dU_A, U_F, dU_F
end


"""
    layerAngularFunctions(n, cosϑ, sinϑ, plm)

``Pₙ¹(cosϑ)``, ``πₙ = Pₙ¹/sinϑ`` and ``τₙ = dPₙ¹/dϑ``, extending the cache `plm` in place
by the same recurrence used elsewhere in the package.
"""
function layerAngularFunctions(n::Int, cosϑ, sinϑ, plm::Vector{T}) where {T}

    while length(plm) < n
        m = length(plm)
        push!(plm, ((2m + 1) * cosϑ * plm[m] - (m + 1) * plm[m - 1]) / m)
    end

    P = plm[n]
    Pprev = n > 1 ? plm[n - 1] : zero(T)

    πn = P / sinϑ
    τn = (n * cosϑ * P - (n + 1) * Pprev) / sinϑ

    return P, πn, τn
end


"""
    totalfield(sphere::LayeredSphere, excitation::PlaneWave, point, quantity::Field; nmax, rtol, coefficients)

Total (incident plus scattered) electric or magnetic field at `point`, valid at **any**
radius, including inside the shells where `scatteredfield` is not available.

The point and the returned field are in Cartesian coordinates.

`nmax` caps the number of orders, `rtol` stops the series once the relative change falls
below it, and `coefficients` optionally takes the result of [`layerCoefficientCache`](@ref)
to avoid recomputing the recursion at every point.
"""
function totalfield(
    sphere::LayeredSphere{N,R,C},
    excitation::PlaneWave,
    point::StaticVector,
    quantity::Field;
    nmax=60,
    rtol=1e-12,
    coefficients=nothing,
) where {N,R,C}

    T = typeof(excitation.frequency)
    ω = 2π * excitation.frequency
    E₀ = excitation.amplitude

    sph = cart2sph(point)
    r, ϑ, φ = sph[1], sph[2], sph[3]

    i = layer(sphere, r)

    sinϑ, cosϑ = abs(sin(ϑ)), cos(ϑ)
    sinφ, cosφ = sin(φ), cos(φ)

    CT = Complex{T}
    Fr = zero(CT)
    Fϑ = zero(CT)
    Fφ = zero(CT)

    plm = T[-sinϑ, -3 * sinϑ * cosϑ]

    δ = T(Inf)
    n = 0

    while n < nmax && (n < 10 || δ > rtol)
        n += 1

        c = coefficients === nothing ? layerCoefficients(sphere, excitation, n) : coefficients[n]

        U_A, dU_A, U_F, dU_F = layerRadialFunctions(c, i, r, n, T)
        P, πn, τn = layerAngularFunctions(n, cosϑ, sinϑ, plm)

        kᵢr = c.k[i] * r
        ωε = ω * c.ε[i]
        ωμ = ω * c.μ[i]

        if quantity isa ElectricField
            ΔFr = E₀ * cosφ / (im * kᵢr^2) * n * (n + 1) * U_A * P
            ΔFϑ = -E₀ * cosφ / r * (im / c.k[i] * dU_A * τn + U_F / ωε * πn)
            ΔFφ = +E₀ * sinφ / r * (im / c.k[i] * dU_A * πn + U_F / ωε * τn)
        else
            ΔFr = E₀ * sinφ / (im * kᵢr^2) * n * (n + 1) * U_F * P
            ΔFϑ = -E₀ * sinφ / r * (U_A / ωμ * πn + im / c.k[i] * dU_F * τn)
            ΔFφ = -E₀ * cosφ / r * (U_A / ωμ * τn + im / c.k[i] * dU_F * πn)
        end

        Fr += ΔFr
        Fϑ += ΔFϑ
        Fφ += ΔFφ

        δ = norm(SVector(ΔFr, ΔFϑ, ΔFφ)) / norm(SVector(Fr, Fϑ, Fφ))
    end

    return convertSpherical2Cartesian(SVector{3,CT}(Fr, Fϑ, Fφ), sph)
end


"""
    totalfield(sphere::LayeredSphere, excitation::PlaneWave, points, quantity::Field; kwargs...)

Total field at several points.
"""
function totalfield(sphere::LayeredSphere, excitation::PlaneWave, points::AbstractVecOrMat{<:StaticVector}, quantity::Field; kwargs...)
    return [totalfield(sphere, excitation, p, quantity; kwargs...) for p in points]
end
