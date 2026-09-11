from sokoban_state_modeling.training.sampler import FullCoverageSampler


def test_unique_before_padding_and_resume():
    data = list(range(10))
    sampler = FullCoverageSampler(data, batch_size=4, seed=7)
    iterator = iter(sampler)
    first = [next(iterator) for _ in range(7)]
    saved = sampler.state_dict()
    restored = FullCoverageSampler(data, batch_size=4, seed=7)
    restored.load_state_dict(saved)
    tail = list(restored)
    assert len(first + tail) == 12
    assert len(set((first + tail)[:10])) == 10
    assert (first + tail)[10:] == (first + tail)[:2]


def test_formal_50000_cycle_shape():
    sampler = FullCoverageSampler(range(50_000), batch_size=128, seed=11)
    indices = list(sampler)
    assert len(indices) == 50_048
    assert len(set(indices[:50_000])) == 50_000
    assert indices[50_000:] == indices[:48]
