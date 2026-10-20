
"""
    scatteredfield(sphere::Scatterer{<:ElectromagneticBoundary}, excitation::PlaneWave, quantity::Field; parameter::Parameter=Parameter())
    
Compute the electric field scattered by a PEC sphere, for an incident plane wave.
"""
function scatteredfield(
    sphere::Scatterer{<:ElectromagneticBoundary}, excitation::PlaneWave, quantity::Field; parameter::Parameter=Parameter()
)

    T = typeof(excitation.frequency)
    F = zeros(SVector{3,Complex{T}}, size(quantity.locations))

    # --- rotate coordinates
    points = rotate(excitation, quantity.locations; inverse=true)

    p = progress(length(points))

    # --- compute field in Cartesian representation
    @tasks for ind in eachindex(points)
        F[ind] = scatteredfield(sphere, excitation, points[ind], quantity; parameter=parameter)
        next!(p)
    end
    finish!(p)

    # --- rotate resulting field
    rotate!(excitation, F; inverse=false)

    return F
end



"""
    scatteredfield(sphere::Scatterer{<:ElectromagneticBoundary}, excitation::PlaneWave, point, quantity::Field; parameter::Parameter=Parameter())

Compute the electric field scattered by a PEC or dielectric sphere, for an incident plane wave
travelling in +z-direction with E-field polarization in x-direction.

The point and the returned field are in Cartesian coordinates.
"""
function scatteredfield(
    sphere::Scatterer{<:ElectromagneticBoundary}, excitation::PlaneWave, point, quantity::Field; parameter::Parameter=Parameter()
)

    point_sph = cart2sph(point) # [r ϑ φ]
    r = point_sph[1]

    T = typeof(excitation.frequency)

    eps = parameter.relativeAccuracy

    F = SVector{3,Complex{T}}(0.0, 0.0, 0.0)

    A₀ = amplitude(sphere, excitation, quantity, r)

    A₀ == 0.0 && return F # Inside of PEC return zero field

    δF = T(Inf)
    n = 0

    sinϑ = abs(sin(point_sph[2]))  # note: theta only defined from from 0 to pi
    cosϑ = cos(point_sph[2])       # ok for theta > pi
    sinϕ = sin(point_sph[3])
    cosϕ = cos(point_sph[3])

    # first two values of the Associated Legendre Polynomial
    plm = Vector{T}()
    push!(plm, -sinϑ)
    push!(plm, -T(3.0) * sinϑ * cosϑ)

    try
        while δF > eps || n < 10
            n += 1

            coeffs = scatterCoeff(sphere, excitation, n)
            expansions = expansion(sphere, excitation, quantity, r, plm, cosϑ, sinϑ, n)
            ΔF = Δfieldₙ(sphere, excitation, quantity, r, coeffs, expansions, sinϕ, cosϕ, n)

            F += ΔF

            δF = norm(ΔF) / norm(F) # relative change

            n > 1 && push!(plm, (T(2.0) * n + 1) * cosϑ * plm[n] / n - (n + 1) * plm[n - 1] / n) # recurrence relationship for next associated Legendre polynomials
        end
    catch

    end

    Fin = inside(sphere, excitation, point, quantity; parameter=parameter)

    return convertSpherical2Cartesian(A₀ .* F, point_sph) + Fin
end


function inside(
    sphere::Scatterer{<:ElectromagneticBoundary}, excitation::PlaneWave{T,R,C}, point, quantity::Field; parameter
) where {T,R,C}

    return SVector{3,Complex{R}}(0.0, 0.0, 0.0) # no correction needed
end

function inside(sphere::Sphere{<:Dielectric}, excitation::PlaneWave{T,R,C}, point, quantity::FarField; parameter) where {T,R,C}

    return SVector{3,Complex{R}}(0.0, 0.0, 0.0) # no correction needed
end

function inside(sphere::Sphere{<:Dielectric}, excitation::PlaneWave{T,R,C}, point, quantity::Field; parameter) where {T,R,C}

    # inside the sphere the incident field has to be substracted to get only the scattered part
    if norm(point) < sphere.radius
        return -field(excitation, point, quantity; parameter=parameter)
    end

    return SVector{3,Complex{R}}(0.0, 0.0, 0.0)
end



function inside(sphere::LayeredSphere, excitation::PlaneWave{T,R,C}, point, quantity::FarField; parameter) where {T,R,C}

    return SVector{3,Complex{R}}(0.0, 0.0, 0.0) # no correction needed
end

function inside(sphere::LayeredSphere, excitation::PlaneWave{T,R,C}, point, quantity::Field; parameter) where {T,R,C}

    # `scatterCoeff(::LayeredSphere, ...)` only provides the exterior coefficients, so the
    # Mie loop silently evaluates to zero inside.
    norm(point) < sphere.radius && error(
        "The scattered field inside a `LayeredSphere` is not implemented for a plane-wave excitation; " *
        "only points with r >= $(sphere.radius) are supported. Use `totalfield` for the field inside.",
    )

    return SVector{3,Complex{R}}(0.0, 0.0, 0.0)
end



function Δfieldₙ(sphere, excitation::PlaneWave, quantity::ElectricField, r, coeffs, expansion, sinϕ, cosϕ, n)

    (Nn_r, Nn_ϑ, Nn_ϕ, Mn_ϑ, Mn_ϕ) = expansion

    k = wavenumber(sphere, excitation, r)
    kr = k * r

    aₙ, bₙ = scatterCoeff_of_layer(sphere, r, coeffs)

    ΔEr = +(cosϕ / (im * kr^2)) * aₙ * Nn_r
    ΔEϑ = -(cosϕ / kr) * (aₙ * Nn_ϑ + bₙ * Mn_ϑ)
    ΔEϕ = +(sinϕ / kr) * (aₙ * Nn_ϕ + bₙ * Mn_ϕ)

    return SVector(ΔEr, ΔEϑ, ΔEϕ)
end



function Δfieldₙ(sphere, excitation::PlaneWave, quantity::MagneticField, r, coeffs, expansion, sinϕ, cosϕ, n)

    (Nn_r, Nn_ϑ, Nn_ϕ, Mn_ϑ, Mn_ϕ) = expansion

    k = wavenumber(sphere, excitation, r)
    kr = k * r

    aₙ, bₙ = scatterCoeff_of_layer(sphere, r, coeffs)

    ΔHr = +(sinϕ / (im * kr^2)) * bₙ * Nn_r
    ΔHϑ = -(sinϕ / kr) * (aₙ * Mn_ϑ + bₙ * Nn_ϑ)
    ΔHϕ = -(cosϕ / kr) * (aₙ * Mn_ϕ + bₙ * Nn_ϕ)

    return SVector(ΔHr, ΔHϑ, ΔHϕ)
end



function Δfieldₙ(sphere, excitation::PlaneWave, quantity::FarField, r, coeffs, expansion, sinϕ, cosϕ, n)

    (~, Nn_ϑ, Nn_ϕ, Mn_ϑ, Mn_ϕ) = expansion

    k = wavenumber(excitation)
    aₙ = coeffs[1]
    bₙ = coeffs[2]

    # See Jin, (7.4.44)-(7.4.45)
    # We deviate from Jin by replacing exp(-im*kr) / kr with 1/k
    ΔEϑ = -im * cosϕ * 1 / k * im^n * (aₙ * Nn_ϑ + bₙ * Mn_ϑ)
    ΔEϕ = +im * sinϕ * 1 / k * im^n * (aₙ * Nn_ϕ + bₙ * Mn_ϕ)

    return SVector(eltype(ΔEϑ)(0.0), ΔEϑ, ΔEϕ)
end



"""
    amplitude(sphere, excitation::PlaneWave, quantity::MagneticField, r)

Returns ``H₀/ηᵢ``, where ``H₀`` is the magnetic field of the incident plane wave.
For PEC layers, it returns ``0``.
"""
function amplitude(sphere, excitation::PlaneWave, quantity::MagneticField, r)
    η = impedance(sphere, excitation, r)

    if η == 0.0 # PEC case
        return η * excitation.amplitude # return zero of correct type
    else
        return 1 / η * excitation.amplitude
    end
end



"""
    amplitude(sphere, excitation::PlaneWave, quantity::MagneticField, r)

Returns ``E₀``, where ``E₀`` is the electric field of the incident plane wave.
For PEC layers, it returns ``0``.
"""
function amplitude(sphere, excitation::PlaneWave, quantity::ElectricField, r)
    η = impedance(sphere, excitation, r)

    if η == 0.0 # PEC case
        return η * excitation.amplitude # return zero of correct type
    else
        return excitation.amplitude
    end
end


"""
    amplitude(sphere, excitation::PlaneWave, quantity::MagneticField, r)

Returns ``E₀``, where ``E₀`` is the electric field of the incident plane wave.
"""
function amplitude(sphere, excitation::PlaneWave, quantity::FarField, r)
    return excitation.amplitude
end



"""
    scatterCoeff_of_layer(sphere, r, coeffs)

Given a tuple of scattering coefficients `coeffs`, it returns the two scattering
coefficients of the layer to which `r` belongs.
"""
function scatterCoeff_of_layer(sphere, r, coeffs)
    cmp_index = layer(sphere, r)
    aₙ = coeffs[end - 2 * (cmp_index - 1) - 1]
    bₙ = coeffs[end - 2 * (cmp_index - 1)]
    return aₙ, bₙ
end



"""
    scatterCoeff(sphere::Sphere{PEC}, excitation::PlaneWave, n::Int)

Compute scattering coefficients for a plane wave travelling in +z-direction 
with polarization in x-direction.
"""
function scatterCoeff(sphere::Sphere{PEC}, excitation::PlaneWave, n::Int)

    T = typeof(excitation.frequency)

    k = wavenumber(excitation)

    ka = k * sphere.radius

    s = sqrt(π / 2 / ka)

    Ĵ  = ka * s * besselj(n + T(0.5), ka)   # Riccati-Bessel function
    Ĥ  = ka * s * hankelh2(n + T(0.5), ka)  # Riccati-Hankel function
    Ĵ2 = ka * s * besselj(n - T(0.5), ka)   # for derivate needed
    Ĥ2 = ka * s * hankelh2(n - T(0.5), ka)  # for derivate needed

    # Use recurrence relationship
    dĴ = (Ĵ2 - n / ka * Ĵ)    # derivative Riccati-Bessel function
    dĤ = (Ĥ2 - n / ka * Ĥ)    # derivative Riccati-Hankel function

    aₙ = -im^(-T(n)) * (dĴ / dĤ) * (2 * n + 1) / (n * (n + 1))  # Jin (7.4.41)
    bₙ = -im^(-T(n)) * (Ĵ / Ĥ) * (2 * n + 1) / (n * (n + 1))    # Jin (7.4.42)

    # PEC: ensure field is zero inside
    # We could enforce this somewhat differently
    # But by ensuring that the number of coefficients = 2 * number of layers
    # the code is more elegant

    cₙ = T(0) * aₙ
    dₙ = T(0) * aₙ

    return aₙ, bₙ, cₙ, dₙ
end



function scatterCoeff(sphere::Sphere{<:Dielectric}, excitation::PlaneWave, n::Int)

    f = excitation.frequency
    T = typeof(f)

    ε2 = excitation.embedding.ε
    μ2 = excitation.embedding.μ

    ε1 = sphere.boundary.filling.ε
    μ1 = sphere.boundary.filling.μ

    c2 = 1 / sqrt(ε2 * μ2)
    c1 = 1 / sqrt(ε1 * μ1)

    k2 = 2π * f / c2
    k1 = 2π * f / c1

    εᵣ = ε1 / ε2
    μᵣ = μ1 / μ2

    k₂a = k2 * sphere.radius
    s₂ = sqrt(π / 2 / k₂a)

    Ĵ₂  = k₂a * s₂ * besselj(n + T(0.5), k₂a)   # Riccati-Bessel function
    Ĥ₂  = k₂a * s₂ * hankelh2(n + T(0.5), k₂a)  # Riccati-Hankel function
    Ĵ₂2 = k₂a * s₂ * besselj(n - T(0.5), k₂a)   # for derivate needed
    Ĥ₂2 = k₂a * s₂ * hankelh2(n - T(0.5), k₂a)  # for derivate needed

    # Use recurrence relationship
    dĴ₂ = (Ĵ₂2 - n / k₂a * Ĵ₂)    # derivative Riccati-Bessel function
    dĤ₂ = (Ĥ₂2 - n / k₂a * Ĥ₂)    # derivative Riccati-Hankel function

    k₁a = k1 * sphere.radius
    s₁ = sqrt(π / 2 / k₁a)

    Ĵ₁ = k₁a * s₁ * besselj(n + T(0.5), k₁a)   # Riccati-Bessel function
    #Ĥ₁  = k₁a * s₁ * hankelh2(n + T(0.5), k₁a)  # Riccati-Hankel function
    Ĵ₁2 = k₁a * s₁ * besselj(n - T(0.5), k₁a)   # for derivate needed
    #Ĥ₁2 = k₁a * s₁ * hankelh2(n - T(0.5), k₁a)  # for derivate needed

    # Use recurrence relationship
    dĴ₁ = (Ĵ₁2 - n / k₁a * Ĵ₁)    # derivative Riccati-Bessel function
    #dĤ₁ = (Ĥ₁2 - n / k₁a * Ĥ₁)    # derivative Riccati-Hankel function

    pF = im^(-T(n)) * (2 * n + 1) / (n * (n + 1))

    aₙ = pF * (√εᵣ * dĴ₂ * Ĵ₁ - √μᵣ * Ĵ₂ * dĴ₁) / (√μᵣ * Ĥ₂ * dĴ₁ - √εᵣ * dĤ₂ * Ĵ₁)     # Jin (7.4.65)
    bₙ = pF * (√μᵣ * dĴ₂ * Ĵ₁ - √εᵣ * Ĵ₂ * dĴ₁) / (√εᵣ * Ĥ₂ * dĴ₁ - √μᵣ * dĤ₂ * Ĵ₁)     # Jin (7.4.66)
    cₙ = pF * (im * √εᵣ * μᵣ) / (√μᵣ * Ĥ₂ * dĴ₁ - √εᵣ * dĤ₂ * Ĵ₁)      # Jin (7.4.66)
    dₙ = pF * (im * √εᵣ * μᵣ) / (√εᵣ * Ĥ₂ * dĴ₁ - √μᵣ * dĤ₂ * Ĵ₁)      # Jin (7.4.66)

    return aₙ, bₙ, cₙ, dₙ
end



"""
    riccatiBessel(z, n::Int, ::Type{T})

Riccati-Bessel function ``Ĵₙ(z) = z jₙ(z)`` and its derivative with respect to `z`,
using the recurrence ``Ĵₙ'(z) = Ĵₙ₋₁(z) - (n/z) Ĵₙ(z)``.
"""
function riccatiBessel(z, n::Int, ::Type{T}) where {T}
    s = sqrt(π / 2 / z)

    f  = z * s * besselj(n + T(0.5), z)
    f₋ = z * s * besselj(n - T(0.5), z)

    return f, f₋ - n / z * f
end

"""
    riccatiHankel2(z, n::Int, ::Type{T})

Riccati-Hankel function of the second kind ``Ĥₙ⁽²⁾(z) = z hₙ⁽²⁾(z)`` and its derivative.
"""
function riccatiHankel2(z, n::Int, ::Type{T}) where {T}
    s = sqrt(π / 2 / z)

    f  = z * s * hankelh2(n + T(0.5), z)
    f₋ = z * s * hankelh2(n - T(0.5), z)

    return f, f₋ - n / z * f
end

"""
    rescale(p)

Rescale a `(numerator, denominator)` pair so that the larger component has unit magnitude.
The represented ratio is unchanged; this keeps both components in a sane numerical range
while they are propagated outward through many layers.
"""
function rescale(p)
    s = max(abs(p[1]), abs(p[2]))

    iszero(s) && return p

    if !isfinite(s)
        isfinite(p[1]) && return (zero(p[1]), one(p[2]))  # ratio → 0
        isfinite(p[2]) && return (one(p[1]), zero(p[2]))  # ratio → ∞
        return p
    end

    return (p[1] / s, p[2] / s)
end



"""
    scatterCoeff(sphere::LayeredSphere, excitation::PlaneWave, n::Int)

Compute scattering coefficients of a layered dielectric sphere for a plane wave travelling in +z-direction
with polarization in x-direction. Only the coefficients of the exterior are evaluated; for the fields inside
see [`totalfield`](@ref).
"""
function scatterCoeff(sphere::Sphere{<:Layered{<:Dielectric,M}}, excitation::PlaneWave, n::Int) where {M}

    T = typeof(excitation.frequency)

    ω = 2π * excitation.frequency

    N = M + 1 # number of dielectric layers: the core and the M shells

    radii = layerRadii(sphere)
    fillings = layerFillings(sphere)

    CT = promote_type(Complex{T}, typeof(excitation.embedding.ε), typeof(first(fillings).ε), typeof(last(fillings).ε))

    # media of the N shells (inner → outer), the background is appended as layer N+1
    ε = SVector{N + 1,CT}(ntuple(i -> i <= N ? fillings[i].ε : excitation.embedding.ε, N + 1))
    μ = SVector{N + 1,CT}(ntuple(i -> i <= N ? fillings[i].μ : excitation.embedding.μ, N + 1))

    k = ω .* sqrt.(ε .* μ)

    # starting values, the innermost layer is uniform, so it holds Ĵₙ alone and is
    # regular at the origin. Jin (7.4.87) in the Ĵₙ, Ĥₙ⁽²⁾ basis.
    σH = (zero(CT), one(CT))  # TM, from Aᵣ
    σE = (zero(CT), one(CT))  # TE, from Fᵣ

    for i in 1:N
        a = radii[i]

        # ratio inside the interface, being referred to the medium outside
        Ĵ, dĴ = riccatiBessel(k[i] * a, n, T)
        Ĥ, dĤ = riccatiHankel2(k[i] * a, n, T)

        S = sqrt(μ[i + 1] * ε[i] / (ε[i + 1] * μ[i]))

        RH = rescale((S * (σH[2] * Ĵ + σH[1] * Ĥ), σH[2] * dĴ + σH[1] * dĤ))  # Jin (7.4.83)
        RE = rescale(((σE[2] * Ĵ + σE[1] * Ĥ) / S, σE[2] * dĴ + σE[1] * dĤ))  # Jin (7.4.84)

        # same ratio but outside, fixes the amplitudes of layer i+1
        Ĵ, dĴ = riccatiBessel(k[i + 1] * a, n, T)
        Ĥ, dĤ = riccatiHankel2(k[i + 1] * a, n, T)

        σH = rescale((RH[1] * dĴ - RH[2] * Ĵ, RH[2] * Ĥ - RH[1] * dĤ))  # Jin (7.4.85)
        σE = rescale((RE[1] * dĴ - RE[2] * Ĵ, RE[2] * Ĥ - RE[1] * dĤ))  # Jin (7.4.86)
    end

    pF = im^(-T(n)) * (2 * n + 1) / (n * (n + 1))

    aₙ = pF * σH[1] / σH[2]  # Jin (7.4.92)
    bₙ = pF * σE[1] / σE[2]  # Jin (7.4.93)

    # Keep the tuple length at 2 * numlayers so that `scatterCoeff_of_layer` keeps working.
    # The interior coefficients are not evaluated -- see the note in the docstring.
    z = zero(aₙ)

    return (aₙ, bₙ, ntuple(i -> z, Val(2 * N))...)
end



# =====================================================================================
#  Fields inside a multilayer dielectric sphere under a plane wave: the expansion
#  coefficients of every layer and the total field at any radius.
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
function layerCoefficients(sphere::LayeredSphere, excitation::PlaneWave, n::Int)

    T = typeof(excitation.frequency)
    ω = 2π * excitation.frequency

    radii = layerRadii(sphere)
    fillings = layerFillings(sphere)

    N = length(radii) # number of dielectric layers: the core and the shells

    CT = promote_type(Complex{T}, typeof(excitation.embedding.ε), typeof(first(fillings).ε), typeof(last(fillings).ε))

    ε = CT[i <= N ? fillings[i].ε : excitation.embedding.ε for i in 1:(N + 1)]
    μ = CT[i <= N ? fillings[i].μ : excitation.embedding.μ for i in 1:(N + 1)]
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
    sphere::LayeredSphere, excitation::PlaneWave, point::StaticVector, quantity::Field; nmax=60, rtol=1e-12, coefficients=nothing
)

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



"""
    expansion(sphere::Scatterer{<:ElectromagneticBoundary}, excitation::PlaneWave, quantity::Field, r, plm, cosϑ, sinϑ, n::Int) 

Compute functional dependencies of the Mie series for a plane wave
travelling in +z-direction with polarization in x-direction.
"""
function expansion(sphere::Scatterer{<:ElectromagneticBoundary}, excitation::PlaneWave, quantity::Field, r, plm, cosϑ, sinϑ, n::Int)

    T = typeof(excitation.frequency)

    if quantity isa FarField
        B = T(1.0)
        dB = T(1.0)
    else
        k = wavenumber(sphere, excitation, r)
        kr = k * r
        s = sqrt(π / 2 / kr)

        if r >= sphere.radius
            B  = kr * s * hankelh2(n + T(0.5), kr)     # Riccati-Hankel function
            B2 = kr * s * hankelh2(n - T(0.5), kr)
        else
            B  = kr * s * besselj(n + T(0.5), kr)     # Riccati-Hankel function
            B2 = kr * s * besselj(n - T(0.5), kr)
        end

        dB = B2 - n / kr * B          # derivative of Riccati-Hankel function
    end

    p = plm[n]

    if abs(cosϑ) < 0.999999
        if n == 1 # derivative of associated Legendre Polynomial
            dp = cosϑ * plm[1] / sqrt(T(1.0) - cosϑ * cosϑ)
        else
            dp = (n * cosϑ * plm[n] - (n + 1) * plm[n - 1]) / sqrt(T(1.0) - cosϑ * cosϑ)
        end

        Mn_ϑ = B * p / sinϑ
        Mn_ϕ = B * dp

        Nn_ϑ = dB * dp
        Nn_ϕ = dB * p / sinϑ

    elseif cosϑ > 0.999999
        aux = (n + T(1.0)) * n / T(2.0)

        Mn_ϑ = -B * aux
        Mn_ϕ = Mn_ϑ

        Nn_ϑ = -dB * aux
        Nn_ϕ = Nn_ϑ

    elseif cosϑ < -0.999999
        aux = (n + T(1.0)) * n / T(2.0) * T(-1.0)^n

        Mn_ϑ = B * aux
        Mn_ϕ = -Mn_ϑ

        Nn_ϑ = -dB * aux
        Nn_ϕ = -Nn_ϑ
    end

    if !(quantity isa FarField)
        Nn_r = n * (n + 1) * B * p
        Nn_ϑ = im * Nn_ϑ
        Nn_ϕ = im * Nn_ϕ
    else
        Nn_r = T(0)
    end

    return Nn_r, Nn_ϑ, Nn_ϕ, Mn_ϑ, Mn_ϕ
end
