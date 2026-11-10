
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



#  layered dielectric sphere: series in rescaled quantities


"""
    riccatiLogDerivatives(z, nmax::Int)

Logarithmic derivatives ``D⁽¹⁾ₙ(z) = Ĵₙ′(z)/Ĵₙ(z)`` and ``D⁽³⁾ₙ(z) = Ĥₙ⁽²⁾′(z)/Ĥₙ⁽²⁾(z)`` of the Riccati-Bessel
functions and the product ``Ĵₙ(z) Ĥₙ⁽²⁾(z)`` for the orders `n = 0 … nmax` (stored at the index `n + 1`).

``D⁽¹⁾ₙ`` is obtained by the downward recurrence, ``D⁽³⁾ₙ`` and the product by upward ones.
"""
function riccatiLogDerivatives(z::Complex{T}, nmax::Int) where {T}

    # downward recurrence for D⁽¹⁾
    nstart = max(nmax, ceil(Int, abs(z)))
    nstart += ceil(Int, 8 * cbrt(nstart)) + 15

    D1 = Vector{Complex{T}}(undef, nmax + 1)
    D = zero(z)

    for n in nstart:-1:1
        D = n / z - 1 / (D + n / z)      # D⁽¹⁾ₙ₋₁ from D⁽¹⁾ₙ
        n <= nmax + 1 && (D1[n] = D)
    end

    # upward recurrences for ĴₙĤₙ⁽²⁾ and D⁽³⁾, with Ĵ₀Ĥ₀⁽²⁾ = (1 - exp(-2jz))/2 and D⁽³⁾₀ = -j
    D3 = Vector{Complex{T}}(undef, nmax + 1)
    ĴĤ = Vector{Complex{T}}(undef, nmax + 1)

    D3[1] = -im
    ĴĤ[1] = -expm1(-2im * z) / 2
    σ, Ĵ₁ˢ = riccatiBessel1Scaled(z, D1[2])
    ĴĤ[2] = exp(im * (σ - 1) * z) * Ĵ₁ˢ * (im / z - 1)   # Ĵ₁Ĥ₁, Ĥ₁ = e^{-jz}(j/z − 1)

    for n in 1:nmax
        n >= 2 &&(ĴĤ[n + 1] = ĴĤ[n] * (n / z - D3[n]) / (D1[n + 1] + n / z))
        #if ka has a positive imaginary part, the Wronskian is used otherwise direct recurrece is used  
        D3[n + 1] = imag(z) > 0 ? D1[n + 1] - im / ĴĤ[n + 1] : -n / z + 1 / (n / z - D3[n]) 
        #D3[n + 1] = D1[n + 1] - im / ĴĤ[n + 1]   # from the Wronskian ĴₙĤₙ⁽²⁾′ - Ĵₙ′Ĥₙ⁽²⁾ = -j
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

function riccatiBessel1Scaled(z, D1₁)
    σ = imag(z) <= 0 ? 1 : -1
    e = expm1(-2im * σ * z) # exp(-2jσz) - 1
    s = σ * im * e / 2 # exp(-jσz) sin z
    d = D1₁ + 1 / z # Ĵ₀/Ĵ₁
    abs(d) >= 1 && return σ, s / d # Ĵ₁ = sin z / d
    return σ, s / z - (1 + e / 2)   # Ĵ₁ = sin z/z − cos z
end
"""
    riccatiBesselRatios(z₁, D1z₁, z₂, D1z₂, nmax::Int)

Ratios ``Ĵₙ(z₁)/Ĵₙ(z₂)`` for `n = 0 … nmax` (index `n + 1`), from the logarithmic derivatives ``D⁽¹⁾`` at
both arguments via ``Ĵₙ(z) = Ĵₙ₋₁(z) / (D⁽¹⁾ₙ(z) + n/z)``.
"""
function riccatiBesselRatios(z₁, D1z₁, z₂, D1z₂, nmax::Int)

    Q = Vector{eltype(D1z₁)}(undef, nmax + 1)
    Q[1] = sineRatio(z₁, z₂)
    σ₁, Ĵ₁ˢ = riccatiBessel1Scaled(z₁, D1z₁[2])
    σ₂, Ĵ₂ˢ = riccatiBessel1Scaled(z₂, D1z₂[2])
    Q[2] = exp(im * (σ₁ * z₁ - σ₂ * z₂)) * Ĵ₁ˢ / Ĵ₂ˢ  # Ĵ₁(z₁)/Ĵ₁(z₂)
    for n in 2:nmax
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

Number of orders of the series.
"""
function seriesTruncation(sphere::LayeredSphere, excitation::PlaneWave)

    ω = 2π * excitation.frequency

    radii = layerRadii(sphere)
    fillings = layerFillings(sphere)

    N = length(radii)

    x = abs(ω * sqrt(complex(excitation.embedding.ε * excitation.embedding.μ)) * radii[N])

    for i in 1:N
        kᵢ = ω * sqrt(complex(fillings[i].ε * fillings[i].μ))

        x = max(x, abs(kᵢ * radii[i]))
        i > 1 && (x = max(x, abs(kᵢ * radii[i - 1])))
    end

    return ceil(Int, x + 11 * cbrt(x) + 1) + 15
end


"""
    rescaledLayerCoefficients(sphere::LayeredSphere, excitation::PlaneWave; parameter::Parameter=Parameter())

The number of orders is determined by [`seriesTruncation`](@ref), unless `parameter.nmax` is non-negative. A given
`nmax` that is at least as large is used as it is. A smaller one is verified, as for the
acoustic spheroid: while the omitted orders are not negligible, measured by [`truncationTail`](@ref) against
`parameter.relativeAccuracy`, the number of orders is raised by half and the coefficients are recomputed.

The coefficients do not depend on the observation point; [`scatteredfield`](@ref) computes them once for all
points.
"""
function rescaledLayerCoefficients(sphere::LayeredSphere, excitation::PlaneWave; parameter::Parameter=Parameter())

    nmax = seriesTruncation(sphere, excitation)

    parameter.nmax < 0 && return rescaledLayerCoefficients(sphere, excitation, nmax)
    parameter.nmax >= nmax && return rescaledLayerCoefficients(sphere, excitation, parameter.nmax)

    # a smaller truncation is raised until the omitted orders are negligible. There is no upper limit.
    N = max(parameter.nmax, 2)
    coeffs = rescaledLayerCoefficients(sphere, excitation, N)

    while truncationTail(coeffs) > parameter.relativeAccuracy
        N = max(ceil(Int, 3 * N / 2), N + 1)
        coeffs = rescaledLayerCoefficients(sphere, excitation, N)
    end

    return coeffs
end


"""
    truncationTail(coeffs::RescaledLayerCoefficients)

Size of the last two orders of the coefficients relative to the largest order, as a measure of the omitted orders.

The size of order n is ``n(n+1)`` times its largest amplitude in any layer and in the background, TM and TE: the
amplitudes multiply radial functions referred to the interface where they are largest, and angular functions growing
as ``n(n+1)/2``. Two orders are taken, as one may vanish by itself.
"""
function truncationTail(coeffs::RescaledLayerCoefficients)

    nmax = coeffs.nmax

    orderSize(n) =
        n *
        (n + 1) *
        max(
            maximum(abs, view(coeffs.cA, :, n)),
            maximum(abs, view(coeffs.aA, :, n)),
            maximum(abs, view(coeffs.cF, :, n)),
            maximum(abs, view(coeffs.aF, :, n)),
            abs(coeffs.sA[n]),
            abs(coeffs.sF[n]),
        )

    sizes = [orderSize(n) for n in 1:nmax]

    return maximum(sizes[max(nmax - 1, 1):nmax]) / maximum(sizes)
end


"""
    rescaledLayerCoefficients(sphere::LayeredSphere, excitation::PlaneWave, nmax::Int)

Rescaled coefficients of every layer for the orders `1 … nmax`, see [`RescaledLayerCoefficients`](@ref).
"""
function rescaledLayerCoefficients(sphere::LayeredSphere, excitation::PlaneWave, nmax::Int)

    T = typeof(excitation.frequency)
    ω = 2π * excitation.frequency

    radii = layerRadii(sphere)
    fillings = layerFillings(sphere)

    N = length(radii) # number of dielectric layers: the core and the shells

    CT = promote_type(Complex{T}, typeof(excitation.embedding.ε), typeof(first(fillings).ε), typeof(last(fillings).ε))

    # media of the N layers (inner → outer), the background is appended as layer N+1
    ε = CT[i <= N ? fillings[i].ε : excitation.embedding.ε for i in 1:(N + 1)]
    μ = CT[i <= N ? fillings[i].μ : excitation.embedding.μ for i in 1:(N + 1)]
    k = CT[ω * sqrt(ε[i] * μ[i]) for i in 1:(N + 1)]

    #the statement below is needed for the consistency of the ratios.
    for i in 1:N
        imag(k[i]) > 0 && (k[i] = -k[i])
    end

    # logarithmic derivatives and ratios at the interfaces
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
    σₑ, Ĵ₁ˢ = riccatiBessel1Scaled(xₑ, D1in[N + 1][2])
    Ĵ[2] = exp(im * σₑ * xₑ) * Ĵ₁ˢ
    for n in 2:nmax
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

            # background: incident Ĵₙ plus scattered Ĥₙ⁽²⁾ match the sphere at the outermost interface
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

            # inward pass: amplitudes from the continuity of U/μ (TM) or U/ε (TE)
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
    scatteredfield(sphere::LayeredSphere, excitation::PlaneWave, quantity::Field; parameter::Parameter=Parameter())

Compute the electric or magnetic field, or the far field, scattered by a layered dielectric sphere, for an incident
plane wave. Inside the sphere, the scattered field is the total field minus the incident one, as for the dielectric
sphere.

The fields are computed in rescaled quantities, which stay within the range of the floating-point numbers for
electrically large spheres with many layers, see [`rescaledLayerCoefficients`](@ref). The coefficients are computed
once for all points.
"""
function scatteredfield(
    sphere::LayeredSphere,
    excitation::PlaneWave,
    quantity::Union{ElectricField,MagneticField,FarField};
    parameter::Parameter=Parameter(),
)

    T = typeof(excitation.frequency)
    F = zeros(SVector{3,Complex{T}}, size(quantity.locations))

    coeffs = rescaledLayerCoefficients(sphere, excitation; parameter=parameter)

    # --- rotate coordinates
    points = rotate(excitation, quantity.locations; inverse=true)

    p = progress(length(points))

    # --- compute field in Cartesian representation
    @tasks for ind in eachindex(points)
        F[ind] = scatteredfield(sphere, excitation, points[ind], quantity, coeffs)
        next!(p)
    end
    finish!(p)

    # --- rotate resulting field
    rotate!(excitation, F; inverse=false)

    return F
end


"""
    scatteredfield(sphere::LayeredSphere, excitation::PlaneWave, point, quantity::Field; parameter::Parameter=Parameter())

Compute the field scattered by a layered dielectric sphere at `point`, for an incident plane wave travelling in
+z-direction with E-field polarization in x-direction.

The point and the returned field are in Cartesian coordinates.
"""
function scatteredfield(
    sphere::LayeredSphere,
    excitation::PlaneWave,
    point,
    quantity::Union{ElectricField,MagneticField,FarField};
    parameter::Parameter=Parameter(),
)
    return scatteredfield(sphere, excitation, point, quantity, rescaledLayerCoefficients(sphere, excitation; parameter=parameter))
end


"""
    scatteredfield(sphere::LayeredSphere, excitation::PlaneWave, point, quantity::Field, coeffs::RescaledLayerCoefficients)

Scattered electric or magnetic field at `point` for the coefficients `coeffs`, for an incident plane wave travelling
in +z-direction with E-field polarization in x-direction.

The point and the returned field are in Cartesian coordinates.
"""
function scatteredfield(
    sphere::LayeredSphere,
    excitation::PlaneWave,
    point,
    quantity::Union{ElectricField,MagneticField},
    coeffs::RescaledLayerCoefficients{CT},
) where {CT}

    ω = 2π * excitation.frequency
    E₀ = excitation.amplitude

    k₁ = coeffs.k[1]
    if abs(k₁ * norm(point)) <= eps(real(CT))
        z₁ = k₁ * coeffs.radii[1]
        σ, Ĵ₁ˢ = riccatiBessel1Scaled(z₁, coeffs.D1out[1][2])
        Ĵ₁ = exp(im * σ * z₁) * Ĵ₁ˢ   # Ĵ₁(k₁a₁)

        if quantity isa ElectricField
            F = E₀ * 2im / 3 * coeffs.cA[1, 1] / Ĵ₁ * SVector{3,CT}(1, 0, 0)
        else
            F = E₀ * 2im / 3 * coeffs.cF[1, 1] / Ĵ₁ * SVector{3,CT}(0, 1, 0)
        end

        return F - field(planeWave(; embedding=excitation.embedding, frequency=excitation.frequency, amplitude=E₀), point, quantity)
    end

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

    πs, τs = angularFunctions(cosϑ, nmax)

    for n in 1:nmax

        πₙ = πs[n]
        τₙ = τs[n]
        Pₙ = sinϑ * πₙ   # associated Legendre polynomial Pₙ¹(cosϑ)

        if quantity isa ElectricField
            Fr += cosϕ / (im * x^2) * n * (n + 1) * U_A[n] * Pₙ
            Fϑ -= cosϕ / r * (im / k * dU_A[n] * τₙ + U_F[n] / ωε * πₙ)
            Fϕ += sinϕ / r * (im / k * dU_A[n] * πₙ + U_F[n] / ωε * τₙ)
        else
            Fr += sinϕ / (im * x^2) * n * (n + 1) * U_F[n] * Pₙ
            Fϑ -= sinϕ / r * (U_A[n] / ωμ * πₙ + im / k * dU_F[n] * τₙ)
            Fϕ -= cosϕ / r * (U_A[n] / ωμ * τₙ + im / k * dU_F[n] * πₙ)
        end
    end

    F = convertSpherical2Cartesian(E₀ .* SVector{3,CT}(Fr, Fϑ, Fϕ), point_sph)

    # --- in the background only the scattered field is expanded; inside, the series is the total field, from which
    #     the incident one is subtracted, as for the dielectric sphere
    i == N + 1 && return F

    return F - field(planeWave(; embedding=excitation.embedding, frequency=excitation.frequency, amplitude=E₀), point, quantity)
end


"""
    angularFunctions(cosϑ, nmax::Int)

``πₙ = ((2n - 1) cosϑ πₙ₋₁ - n πₙ₋₂) / (n - 1)``,   ``τₙ = n cosϑ πₙ - (n + 1) πₙ₋₁``,   ``π₀ = 0``, ``π₁ = -1``.
"""
function angularFunctions(cosϑ, nmax::Int)

    πs = zeros(typeof(cosϑ), nmax)
    τs = zeros(typeof(cosϑ), nmax)

    πₙ₋₁ = zero(cosϑ)
    πₙ = -one(cosϑ)

    for n in 1:nmax
        πs[n] = πₙ
        τs[n] = n * cosϑ * πₙ - (n + 1) * πₙ₋₁

        πₙ₋₁, πₙ = πₙ, ((2 * n + 1) * cosϑ * πₙ - (n + 1) * πₙ₋₁) / n
    end

    return πs, τs
end


"""
    scatteredfield(sphere::LayeredSphere, excitation::PlaneWave, point, quantity::FarField, coeffs::RescaledLayerCoefficients)

Scattered far field in the direction of `point` for the coefficients `coeffs`, for an incident plane wave travelling
in +z-direction with E-field polarization in x-direction: the series of the other spheres, with the scattering
coefficients ``aₙ``, ``bₙ`` of the layered sphere.

The point and the returned field are in Cartesian coordinates.
"""
function scatteredfield(
    sphere::LayeredSphere, excitation::PlaneWave, point, quantity::FarField, coeffs::RescaledLayerCoefficients{CT}
) where {CT}

    point_sph = cart2sph(point) # [r ϑ φ]

    cosϑ = cos(point_sph[2])
    sinϕ = sin(point_sph[3])
    cosϕ = cos(point_sph[3])

    k = wavenumber(excitation)

    πs, τs = angularFunctions(cosϑ, coeffs.nmax)

    Fϑ = zero(CT)
    Fϕ = zero(CT)

    # See Jin, (7.4.44)-(7.4.45), with exp(-jkr)/kr replaced by 1/k, as in Δfieldₙ for the other spheres
    for n in 1:(coeffs.nmax)
        Fϑ += -im * cosϕ / k * im^n * (coeffs.aₙ[n] * τs[n] + coeffs.bₙ[n] * πs[n])
        Fϕ += +im * sinϕ / k * im^n * (coeffs.aₙ[n] * πs[n] + coeffs.bₙ[n] * τs[n])
    end

    return convertSpherical2Cartesian(excitation.amplitude .* SVector{3,CT}(0, Fϑ, Fϕ), point_sph)
end
