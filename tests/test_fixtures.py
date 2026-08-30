def test_every_stem_has_recorded_facts(fixture_facts, fixture_stems):
    assert sorted(fixture_facts) == sorted(fixture_stems)


def test_fixture_file_exists(bvh_fixture):
    assert bvh_fixture.is_file()
    assert bvh_fixture.read_text().startswith("HIERARCHY")
