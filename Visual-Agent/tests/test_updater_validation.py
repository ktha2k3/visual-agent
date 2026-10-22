import pytest

from tools.visualization.validation import validate_manim_code


def scene(callback, registration="bob.add_updater(update)"):
    return "from manim import *\nclass Demo(Scene):\n    def construct(self):\n        bob = Dot()\n" + "\n".join(
        "        " + line for line in (callback + "\n" + registration).splitlines()
    ) + "\n        self.play(Create(bob))\n"


@pytest.mark.parametrize("callback,registration", [
    ("def update():\n    return None", "bob.add_updater(update)"),
    ("", "bob.add_updater(lambda: None)"),
    ("def update(m, extra):\n    return None", "bob.add_updater(update)"),
    ("def update(m, *, dt):\n    return None", "bob.add_updater(update_function=update)"),
])
def test_rejects_updater_that_cannot_accept_manim_arguments(callback, registration):
    report = validate_manim_code(scene(callback, registration))
    assert not report.valid
    assert any("add_updater" in error and "line" in error for error in report.errors)


@pytest.mark.parametrize("callback,registration", [
    ("def update(m):\n    return None", "bob.add_updater(update)"),
    ("def update(m, dt):\n    return None", "bob.add_updater(update)"),
    ("def update(m, dt=0):\n    return None", "bob.add_updater(update)"),
    ("def update(m, offset=1):\n    return None", "bob.add_updater(update)"),
    ("", "bob.add_updater(lambda m: m.shift(RIGHT))"),
    ("", "bob.add_updater(lambda m, dt: m.shift(dt * RIGHT))"),
    ("def update(*args):\n    return None", "bob.add_updater(update)"),
])
def test_accepts_valid_updater_signatures(callback, registration):
    assert validate_manim_code(scene(callback, registration)).valid


def test_callback_names_are_resolved_in_their_own_function_scope():
    code = scene("def update(m):\n    return None")
    code += "\ndef unrelated():\n    def update():\n        return None\n"
    assert validate_manim_code(code).valid
