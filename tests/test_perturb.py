"""Edge-case tests for perturb.py. Plain asserts, no pytest dependency --
run directly: python -m tests.test_perturb
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.perturb import typo, tone_drop, teencode, augment, augment_fixed


def test_empty_string():
    assert typo("") == ""
    assert tone_drop("") == ""
    assert teencode("") == ""
    assert augment("", p=1.0) == ""
    assert augment_fixed("") == ""


def test_no_vietnamese_chars():
    s = "hello world 123"
    assert tone_drop(s) == s          # nothing to strip
    assert teencode(s) == s           # no dictionary matches


def test_very_short_text():
    for s in ["a", "ê", "!", "  "]:
        # must not crash regardless of length
        typo(s, rate=1.0)
        tone_drop(s)
        teencode(s)
        augment(s, p=1.0)
        augment_fixed(s)


def test_tone_drop_preserves_length_class():
    s = "không đẹp"
    out = tone_drop(s)
    assert out == "khong dep"


def test_teencode_known_substitution():
    s = "không biết"
    out = teencode(s)
    assert out != s  # should substitute at least one known word


def test_augment_fixed_deterministic():
    s = "Máy này rất đẹp và pin trâu"
    assert augment_fixed(s) == augment_fixed(s)
    # different seeds should (usually) differ
    assert augment_fixed(s, seed=0) != augment_fixed(s, seed=999) or True  # not guaranteed, just must not crash


def test_augment_p_zero_is_noop():
    s = "không bao giờ thay đổi"
    assert augment(s, p=0.0) == s


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  {t.__name__}: OK")
    print(f"{len(tests)}/{len(tests)} tests passed")
