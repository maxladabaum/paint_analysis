import numpy as np

from origami_analysis import _boundary_template_correlation


def normalized(image):
    image = np.asarray(image, dtype=float)
    image = image - image.mean()
    return image / np.linalg.norm(image)


def corner_template():
    template = np.zeros((11, 13))
    template[2, 3] = template[2, 9] = 1
    template[8, 3] = template[8, 9] = 1
    return template


def test_interior_signal_and_darkness_are_ignored():
    template = corner_template()
    extra_signal = template.copy()
    extra_signal[3:8, 4:9] = 100
    # Unmarked pixels on the rectangle boundary are also interior.
    extra_signal[2, 5] = 50
    for candidate in (template, extra_signal):
        assert np.isclose(_boundary_template_correlation(
            normalized(candidate), normalized(template)), 1)


def test_exterior_signal_lowers_correlation_on_every_side():
    template = corner_template()
    for location in ((1, 6), (9, 6), (5, 2), (5, 10)):
        candidate = template.copy()
        candidate[location] = 2
        score = _boundary_template_correlation(normalized(candidate), normalized(template))
        assert np.isclose(score, 1 / np.sqrt(2))
        candidate[4:7, 4:9] = 100
        assert np.isclose(_boundary_template_correlation(
            normalized(candidate), normalized(template)), score)


def test_missing_expected_bright_signal_lowers_correlation():
    template = corner_template()
    missing = template.copy()
    missing[2, 3] = 0
    assert np.isclose(_boundary_template_correlation(
        normalized(missing), normalized(template)), np.sqrt(3 / 4))
    exterior_only = np.zeros_like(template)
    exterior_only[0, 0] = 10
    assert _boundary_template_correlation(normalized(exterior_only), normalized(template)) == 0


def test_empty_images_have_zero_score():
    template = corner_template()
    assert _boundary_template_correlation(np.zeros_like(template), normalized(template)) == 0
    assert _boundary_template_correlation(template, np.zeros_like(template)) == 0
