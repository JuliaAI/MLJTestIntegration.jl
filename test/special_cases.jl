@testset "_filter" begin
    proxies = [
        (name="1", package_name="A", extra="cat"),
        (name="1", package_name="B", extra="mouse"),
        (name="2", package_name="B", extra="dog"),
        (name="1", package_name="C", extra="rat"),
    ]

    bad = [
        (name="1", package_name="A"),
        (name="1", package_name="B"),
    ]

    @test MLJTest._filter(proxies, bad) == [
        (name="2", package_name="B", extra="dog"),
        (name="1", package_name="C", extra="rat"),
    ]
end
