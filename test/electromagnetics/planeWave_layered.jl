
@testitem "Layered dielectric sphere" setup = [Setup] begin

    SpecialFunctions = SphericalScattering.SpecialFunctions

    medium(εr, μr=1.0) = Medium(𝜀 * ComplexF64(εr), 𝜇 * ComplexF64(μr))
    layered(radii, media) = LayeredSphere(; radii=SVector(radii...), filling=SVector(media...))

    # frequency for which the embedding wavenumber times the radius a equals ka
    frequencyOf(ka; a=spRadius, embedding=Medium(𝜀, 𝜇)) = ka / (2π * a * sqrt(embedding.ε * embedding.μ))

    relativeDifference(F, Fref) = maximum(norm.(F - Fref)) / maximum(norm.(Fref))

    points_out = vec(points_cartNF)
    points_in = vec(points_cartNF_inside)
    directions = vec(points_cartFF)

    # largest difference of the near fields (E, H, inside and outside), the far field and the monostatic RCS
    function largestDifference(sp, ref, ex)
        d = 0.0
        for Q in (ElectricField, MagneticField), points in (points_out, points_in)
            d = max(d, relativeDifference(field(sp, ex, Q(points)), field(ref, ex, Q(points))))
        end
        d = max(d, relativeDifference(scatteredfield(sp, ex, FarField(directions)), scatteredfield(ref, ex, FarField(directions))))
        σ = rcs(sp, ex)
        σref = rcs(ref, ex)
        return max(d, abs(σ - σref) / σref)
    end

    @testset "One layer against DielectricSphere" begin

        fillings = (
            "lossless" => medium(4.0),
            "lossy" => medium(2.0 + 3.0im),
            "magnetic, lossy" => medium(3.0 - 0.5im, 2.0 - 0.3im),
            "high contrast" => medium(16.0),
            "negative permittivity" => medium(-2.0),
        )

        @testset "$name, ka = $ka" for (name, filling) in fillings, ka in (0.1, 2.0, 20.0)

            sp = layered((spRadius,), (filling,))
            ref = DielectricSphere(; radius=spRadius, filling=filling)
            ex = planeWave(; frequency=frequencyOf(ka))

            @test largestDifference(sp, ref, ex) < 1e-8
        end

        embedding = Medium(𝜀 * 3.0, 𝜇 * 2.0)
        filling = medium(2.0 - 0.4im, 1.5)

        sp = layered((spRadius,), (filling,))
        ref = DielectricSphere(; radius=spRadius, filling=filling)

        @testset "Embedding and amplitude" begin
            ex = planeWave(; frequency=frequencyOf(5.0; embedding=embedding), embedding=embedding, amplitude=2.5)

            @test largestDifference(sp, ref, ex) < 1e-8
        end

        # outside only: for a rotated plane wave, DielectricSphere subtracts the incident field inside the sphere at the
        # point already rotated to the standard frame, so its inside field is not a reference here (the continuity of
        # the layered fields for oblique incidence is tested below)
        @testset "Direction of incidence" begin
            ex = planeWave(;
                frequency=frequencyOf(5.0; embedding=embedding),
                embedding=embedding,
                amplitude=2.5,
                direction=normalize(SVector(1.0, 1.0, 1.0)),
                polarization=normalize(SVector(1.0, -1.0, 0.0)),
            )

            for Q in (ElectricField, MagneticField)
                @test relativeDifference(field(sp, ex, Q(points_out)), field(ref, ex, Q(points_out))) < 1e-8
            end
            @test relativeDifference(scatteredfield(sp, ex, FarField(directions)), scatteredfield(ref, ex, FarField(directions))) <
                1e-8
            @test abs(rcs(sp, ex) - rcs(ref, ex)) < 1e-8 * rcs(ref, ex)
        end
    end

    @testset "Radial functions near ka = mπ" begin ##burda z1 nedendi

        Ĵ(n, z) = z * SpecialFunctions.sphericalbesselj(n, z)

        @testset "Ratios at z₂ = $m π + $δ" for m in 1:3, δ in (0.0, 1e-9, -1e-6, 1e-3)

            nmax = 20
            z₂ = complex(m * π + δ)
            z₁ = 0.7 * z₂

            D1z₁, _, _ = SphericalScattering.riccatiLogDerivatives(z₁, nmax)
            D1z₂, _, _ = SphericalScattering.riccatiLogDerivatives(z₂, nmax)
            Q = SphericalScattering.riccatiBesselRatios(z₁, D1z₁, z₂, D1z₂, nmax)

            Qref = [Ĵ(n, real(z₁)) / Ĵ(n, real(z₂)) for n in 1:nmax]

            @test all(isfinite, Q[2:end])
            @test maximum(abs.(Q[2:end] - Qref) ./ abs.(Qref)) < 1e-12
        end

        # the outer radius, an interface and a field point at k a = mπ, for equal layers against the homogeneous sphere
        filling = medium(4.0)   # k₁ = 2 k₀
        ref = DielectricSphere(; radius=spRadius, filling=filling)

        @testset "Background k₀a = $m π" for m in 1:3, δ in (0.0, 1e-6)

            ex = planeWave(; frequency=frequencyOf(m * π + δ))

            @test largestDifference(layered((spRadius,), (filling,)), ref, ex) < 1e-8
            @test largestDifference(layered((0.5spRadius, spRadius), (filling, filling)), ref, ex) < 1e-8
        end

        @testset "Interface k₁a₁ = $m π" for m in 1:3

            ex = planeWave(; frequency=frequencyOf(6.0))
            a₁ = m * π / 12.0   # k₁ a₁ = 2 * 6.0 * a₁

            @test largestDifference(layered((a₁, spRadius), (filling, filling)), ref, ex) < 1e-8
        end

        @testset "Field point k₁r = $m π" for m in 1:3

            ex = planeWave(; frequency=frequencyOf(5.0))
            points, _ = getDefaultPoints(m * π / 10.0)   # k₁ r = 10 r

            sp = layered((0.5spRadius, spRadius), (filling, filling))
            for Q in (ElectricField, MagneticField)
                @test relativeDifference(field(sp, ex, Q(vec(points))), field(ref, ex, Q(vec(points)))) < 1e-8
            end
        end
    end

    @testset "Continuity at the interfaces" begin

        normals = vec(points_cartFF)

        tangentialJump(Fo, Fi) = maximum(norm.(cross.(normals, Fo - Fi))) / maximum(norm.(cross.(normals, Fo)))

        function normalJump(Fo, Fi, mo, mi)
            jump = maximum(abs.(dot.(normals, mo .* Fo - mi .* Fi)))
            return jump / max(abs(mo) * maximum(norm.(Fo)), abs(mi) * maximum(norm.(Fi)))
        end

        fillingSets = (
            "dielectric" => (medium(2.0), medium(6.0), medium(1.5)),
            "lossy, magnetic" => (medium(2.0 - 0.5im, 1.1 - 0.2im), medium(4.0 - 1.0im, 0.9), medium(3.0 - 0.1im, 2.0 - 0.5im)),
            "high contrast" => (medium(16.0), medium(1.0), medium(80.0 - 5.0im)),
            "negative shell" => (medium(4.0), medium(-2.0 - 0.1im), medium(2.0)),
        )
        incidences = (
            (SVector(0.0, 0.0, 1.0), SVector(1.0, 0.0, 0.0)), (normalize(SVector(1.0, 1.0, 1.0)), normalize(SVector(1.0, -1.0, 0.0)))
        )

        @testset "$name, ka = $ka" for (name, fillings) in fillingSets, ka in (1.0, 10.0), (dir, pol) in incidences

            radii = (0.3spRadius, 0.6spRadius, spRadius)
            sp = layered(radii, fillings)
            ex = planeWave(; frequency=frequencyOf(ka), direction=dir, polarization=pol)

            for (i, a) in enumerate(radii)
                inner = fillings[i]
                outer = i < length(radii) ? fillings[i + 1] : ex.embedding

                kmax = 2π * ex.frequency * max(abs(sqrt(inner.ε * inner.μ)), abs(sqrt(outer.ε * outer.μ)))
                δ = 1e-10 / max(1, kmax * a)

                Eo = field(sp, ex, ElectricField(a * (1 + δ) .* normals))
                Ei = field(sp, ex, ElectricField(a * (1 - δ) .* normals))
                Ho = field(sp, ex, MagneticField(a * (1 + δ) .* normals))
                Hi = field(sp, ex, MagneticField(a * (1 - δ) .* normals))

                @test tangentialJump(Eo, Ei) < 1e-8
                @test tangentialJump(Ho, Hi) < 1e-8
                @test normalJump(Eo, Ei, outer.ε, inner.ε) < 1e-8
                @test normalJump(Ho, Hi, outer.μ, inner.μ) < 1e-8
            end
        end
    end

    @testset "Identical-material interfaces" begin

        A = medium(4.0 - 1.0im)
        B = medium(2.0, 1.5)
        ex = planeWave(; frequency=frequencyOf(4.0))

        sp = layered((0.4spRadius, spRadius), (A, B))
        coeffs = SphericalScattering.rescaledLayerCoefficients(sp, ex)

        # the same sphere with an additional interface inside the shell and inside the core
        @testset "$name" for (name, split) in (
            "shell split" => layered((0.4spRadius, 0.7spRadius, spRadius), (A, B, B)),
            "core split" => layered((0.2spRadius, 0.4spRadius, spRadius), (A, A, B)),
        )
            @test largestDifference(split, sp, ex) < 1e-10

            splitCoeffs = SphericalScattering.rescaledLayerCoefficients(split, ex)
            @test maximum(abs.(splitCoeffs.aₙ - coeffs.aₙ)) < 1e-12 * maximum(abs.(coeffs.aₙ))
            @test maximum(abs.(splitCoeffs.bₙ - coeffs.bₙ)) < 1e-12 * maximum(abs.(coeffs.bₙ))
        end
    end

    @testset "Zero material contrast" begin

        @testset "$name" for (name, embedding) in (("vacuum", Medium(𝜀, 𝜇)), ("embedding", Medium(𝜀 * 3.0, 𝜇 * 2.0)))

            ex = planeWave(; frequency=frequencyOf(5.0; embedding=embedding), embedding=embedding)

            for sp in
                (layered((spRadius,), (embedding,)), layered((0.3spRadius, 0.6spRadius, spRadius), (embedding, embedding, embedding)))
                for Q in (ElectricField, MagneticField), points in (points_out, points_in)
                    Finc = field(ex, Q(points))
                    @test maximum(norm.(scatteredfield(sp, ex, Q(points)))) < 1e-12 * maximum(norm.(Finc))
                end

                @test maximum(norm.(scatteredfield(sp, ex, FarField(directions)))) <
                    1e-12 * ex.amplitude / SphericalScattering.wavenumber(ex)
                @test rcs(sp, ex) < 1e-20 * π * spRadius^2
            end
        end
    end

    @testset "Field at the origin" begin

        ex = planeWave(; frequency=frequencyOf(5.0))
        origin = [SVector(0.0, 0.0, 0.0)]

        dirs = normalize.([
            SVector(1.0, 0.0, 0.0),
            SVector(-1.0, 0.0, 0.0),
            SVector(0.0, 1.0, 0.0),
            SVector(0.0, 0.0, 1.0),
            SVector(0.0, 0.0, -1.0),
            SVector(1.0, -2.0, 0.5),
        ])

        for sp in (
            layered((spRadius,), (medium(4.0),)),
            layered((0.3spRadius, 0.6spRadius, spRadius), (medium(4.0), medium(2.0 - 0.5im), medium(3.0))),
        )

            E₀ = field(sp, ex, ElectricField(origin))[1]
            H₀ = field(sp, ex, MagneticField(origin))[1]

            # finite, and polarized as the incident wave
            @test all(isfinite, E₀)
            @test all(isfinite, H₀)
            @test norm(E₀[2:3]) ≤ 1e-14 * norm(E₀)
            @test abs(H₀[1]) + abs(H₀[3]) ≤ 1e-14 * norm(H₀)

            # the limit from all directions: the field differs from the value at the origin in proportion to r
            for r in (1e-5, 1e-8, 1e-12)
                E = field(sp, ex, ElectricField([r * d for d in dirs]))
                H = field(sp, ex, MagneticField([r * d for d in dirs]))

                @test maximum(norm.(E .- Ref(E₀))) < 1e3 * r * norm(E₀)
                @test maximum(norm.(H .- Ref(H₀))) < 1e3 * r * norm(H₀)
            end
        end

        # against the homogeneous sphere, which is itself not defined at r = 0
        ref = DielectricSphere(; radius=spRadius, filling=medium(4.0))
        sp = layered((0.5spRadius, spRadius), (medium(4.0), medium(4.0)))
        E₀ = field(sp, ex, ElectricField(origin))[1]

        @test norm(E₀ - field(ref, ex, ElectricField([SVector(1e-9, 0.0, 0.0)]))[1]) < 1e-7 * norm(E₀)
    end

    @testset "Angular functions near the poles" begin

        nmax = 1000

        # on the axis, in the sign convention P₁¹ = -sinϑ: πₙ(1) = τₙ(1) = -n(n+1)/2, πₙ(-1) = -τₙ(-1) = (-1)ⁿ n(n+1)/2
        π₊, τ₊ = SphericalScattering.angularFunctions(1.0, nmax)
        π₋, τ₋ = SphericalScattering.angularFunctions(-1.0, nmax)
        limit = [n * (n + 1) / 2 for n in 1:nmax]

        @test π₊ ≈ -limit rtol = 1e-12
        @test τ₊ ≈ -limit rtol = 1e-12
        @test π₋ ≈ [(-1)^n * limit[n] for n in 1:nmax] rtol = 1e-12
        @test τ₋ ≈ -π₋ rtol = 1e-12

        # lowest orders in closed form: π₁ = -1, τ₁ = -cosϑ, π₂ = -3cosϑ, τ₂ = -3cos2ϑ
        ϑ = 0.7
        πs, τs = SphericalScattering.angularFunctions(cos(ϑ), 2)
        @test πs ≈ [-1.0, -3cos(ϑ)]
        @test τs ≈ [-cos(ϑ), -3cos(2ϑ)]

        # close to both poles, against the same recurrence in 256-bit arithmetic, relative to the size n(n+1)/2
        @testset "ϑ = $ϑ" for ϑ in (1e-8, 1e-5, 1.4e-3, π - 1.4e-3, π - 1e-5, π - 1e-8)
            x = cos(ϑ)
            πs, τs = SphericalScattering.angularFunctions(x, nmax)
            πref, τref = setprecision(BigFloat, 256) do
                SphericalScattering.angularFunctions(BigFloat(x), nmax)
            end

            @test maximum(abs.(πs - πref) ./ limit) < 1e-10
            @test maximum(abs.(τs - τref) ./ limit) < 1e-10
        end
    end

    @testset "Fields near the poles" begin

        # an electrically large sphere, so that the series runs to high orders
        filling = medium(2.25)
        sp = layered((0.6spRadius, spRadius), (filling, filling))
        ref = DielectricSphere(; radius=spRadius, filling=filling)
        ex = planeWave(; frequency=frequencyOf(40.0))

        ϑc = acos(0.999999)   # the former cutoff
        point(r, ϑ) = SVector(r * sin(ϑ) * cos(0.3), r * sin(ϑ) * sin(0.3), r * cos(ϑ))

        for r in (0.5spRadius, 2.0spRadius), pole in (0.0, π)

            # on the axis the homogeneous sphere uses the exact limits of πₙ and τₙ
            onAxis = [point(r, pole)]
            for Q in (ElectricField, MagneticField)
                @test relativeDifference(field(sp, ex, Q(onAxis)), field(ref, ex, Q(onAxis))) < 1e-8
            end

            # continuous towards the axis and across the former cutoff
            ϑs = abs.(pole .- (1e-9, ϑc * (1 - 1e-6), ϑc * (1 + 1e-6)))
            for Q in (ElectricField, MagneticField)
                F₀ = field(sp, ex, Q(onAxis))[1]
                F = field(sp, ex, Q([point(r, ϑ) for ϑ in ϑs]))

                @test norm(F[1] - F₀) < 1e-6 * norm(F₀)
                @test norm(F[2] - F[3]) < 1e-6 * norm(F₀)
            end
        end

        # the far field
        for pole in (0.0, π)
            onAxis = [point(1.0, pole)]
            @test relativeDifference(scatteredfield(sp, ex, FarField(onAxis)), scatteredfield(ref, ex, FarField(onAxis))) < 1e-8

            F₀ = scatteredfield(sp, ex, FarField(onAxis))[1]
            F = scatteredfield(sp, ex, FarField([point(1.0, ϑ) for ϑ in abs.(pole .- (1e-9, ϑc * (1 - 1e-6), ϑc * (1 + 1e-6)))]))

            @test norm(F[1] - F₀) < 1e-6 * norm(F₀)
            @test norm(F[2] - F[3]) < 1e-6 * norm(F₀)
        end
    end

    @testset "Negative permittivity" begin

        # a real negative permittivity, as given by the user, without a DomainError
        sp = LayeredSphere(; radii=SVector(0.5spRadius, spRadius), filling=SVector(Medium(-2.0 * 𝜀, 𝜇), Medium(-2.0 * 𝜀, 𝜇)))
        ref = DielectricSphere(; radius=spRadius, filling=medium(-2.0))

        @testset "ka = $ka" for ka in (1.0, 10.0, 30.0)
            ex = planeWave(; frequency=frequencyOf(ka))

            @test all(p -> all(isfinite, p), field(sp, ex, ElectricField(points_in)))
            @test largestDifference(sp, ref, ex) < 1e-8
        end
    end
end
