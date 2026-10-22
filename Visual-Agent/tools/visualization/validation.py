from __future__ import annotations

import ast
import re

from tools.visualization.models import ValidationReport


_ALLOWED_IMPORT_ROOTS = {"manim", "math", "numpy"}
_BLOCKED_CALLS = {
    "open",
    "eval",
    "exec",
    "compile",
    "__import__",
    "input",
    "globals",
    "locals",
    "getattr",
    "setattr",
    "delattr",
    "vars",
    "breakpoint",
    "SVGMobject",
    "ImageMobject",
    "VideoMobject",
}
_BLOCKED_ATTRIBUTES = {
    "system",
    "popen",
    "spawn",
    "write_text",
    "write_bytes",
    "read_text",
    "read_bytes",
    "fromfile",
    "tofile",
    "savetxt",
    "loadtxt",
    "add_sound",
}
_SCENE_BASES = {"Scene", "MovingCameraScene", "ThreeDScene", "ZoomedScene"}
_FENCED_PYTHON = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


def extract_python_code(raw: str) -> str:
    match = _FENCED_PYTHON.search(raw)
    if match:
        code = match.group(1).strip()
    else:
        code = raw.strip()
    if not code:
        raise ValueError("Model returned empty Manim code")
    if len(code) > 200_000:
        raise ValueError("Generated Manim code exceeds the 200 KB safety limit")
    return code + "\n"


def _base_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _string_constants(tree: ast.AST) -> dict[str, str]:
    """Collect simple string assignments that are never reassigned."""
    constants: dict[str, str] = {}
    reassigned: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        value = node.value
        for target in targets:
            if not isinstance(target, ast.Name):
                continue
            if (
                target.id in constants
                or not isinstance(value, ast.Constant)
                or not isinstance(value.value, str)
            ):
                reassigned.add(target.id)
                constants.pop(target.id, None)
            elif target.id not in reassigned:
                constants[target.id] = value.value
    return constants


def _static_string(node: ast.expr, constants: dict[str, str]) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return constants.get(node.id)
    return None


def _updater_signature_errors(tree: ast.AST) -> list[str]:
    """Check statically resolvable callbacks without executing generated code."""
    errors: list[str] = []

    class CheckScope(ast.NodeVisitor):
        def __init__(self):
            self.scopes: list[dict[str, ast.AST | None]] = []

        def visit_Module(self, node):
            self.visit_scope(node)

        def visit_FunctionDef(self, node):
            self.visit_scope(node)

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_scope(self, node):
            bindings: dict[str, ast.AST | None] = {}

            class Collect(ast.NodeVisitor):
                def visit_FunctionDef(self, child):
                    bindings[child.name] = child

                visit_AsyncFunctionDef = visit_FunctionDef

                def visit_ClassDef(self, child):
                    bindings[child.name] = None

                def visit_Name(self, child):
                    if isinstance(child.ctx, ast.Store):
                        bindings[child.id] = None

            collector = Collect()
            for child in node.body:
                collector.visit(child)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for arg in ast.walk(node.args):
                    if isinstance(arg, ast.arg):
                        bindings[arg.arg] = None
            self.scopes.append(bindings)
            for child in node.body:
                self.visit(child)
            self.scopes.pop()

        def visit_Call(self, node):
            if isinstance(node.func, ast.Attribute) and node.func.attr == "add_updater":
                callback = node.args[0] if node.args else next(
                    (kw.value for kw in node.keywords if kw.arg == "update_function"), None
                )
                definition = callback
                if isinstance(callback, ast.Name):
                    definition = None
                    for scope in reversed(self.scopes):
                        if callback.id in scope:
                            definition = scope[callback.id]
                            break
                if isinstance(definition, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                    args = definition.args
                    positional = args.posonlyargs + args.args
                    names = {arg.arg for arg in positional + args.kwonlyargs}
                    names.update(arg.arg for arg in (args.vararg, args.kwarg) if arg)
                    # Manim selects updater(mobject, dt) when 'dt' is in
                    # inspect.signature; otherwise it calls updater(mobject).
                    supplied = 2 if "dt" in names else 1
                    required = len(positional) - len(args.defaults)
                    required_keywords = any(default is None for default in args.kw_defaults)
                    if (
                        isinstance(definition, ast.AsyncFunctionDef)
                        or required > supplied
                        or (args.vararg is None and supplied > len(positional))
                        or required_keywords
                    ):
                        errors.append(
                            f"add_updater callback at line {node.lineno} cannot accept "
                            f"Manim's {supplied} positional argument(s). Use a synchronous "
                            "def update(mobject) or def update(mobject, dt); "
                            "do not require extra arguments or keyword-only dt."
                        )
            self.generic_visit(node)

    CheckScope().visit(tree)
    return errors


def validate_manim_code(
    code: str, *, required_text_font: str | None = None
) -> ValidationReport:
    errors: list[str] = []
    warnings: list[str] = []
    try:
        tree = ast.parse(code, filename="stem_scene.py")
        compile(tree, "stem_scene.py", "exec")
    except (SyntaxError, ValueError) as exc:
        return ValidationReport(valid=False, errors=[f"Python syntax: {exc}"])

    errors.extend(_updater_signature_errors(tree))
    imported_manim = False
    scene_classes: list[str] = []
    has_play = False
    has_wait = False
    string_constants = _string_constants(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in _ALLOWED_IMPORT_ROOTS:
                    errors.append(f"Blocked import: {alias.name}")
                if root == "manim":
                    errors.append("Use `from manim import ...`; plain `import manim` is blocked")
                imported_manim = imported_manim or root == "manim"
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root not in _ALLOWED_IMPORT_ROOTS:
                errors.append(f"Blocked import: {node.module or '<relative>'}")
            imported_manim = imported_manim or root == "manim"
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in _BLOCKED_CALLS:
                errors.append(f"Blocked call: {node.func.id}()")
            if isinstance(node.func, ast.Attribute):
                if node.func.attr in _BLOCKED_ATTRIBUTES:
                    errors.append(f"Blocked attribute call: {node.func.attr}()")
                has_play = has_play or node.func.attr == "play"
                has_wait = has_wait or node.func.attr == "wait"
            if (
                required_text_font
                and isinstance(node.func, ast.Name)
                and node.func.id == "Text"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
                and any(ord(character) > 127 for character in node.args[0].value)
            ):
                font_value = next(
                    (
                        _static_string(keyword.value, string_constants)
                        for keyword in node.keywords
                        if keyword.arg == "font"
                    ),
                    None,
                )
                if font_value != required_text_font:
                    errors.append(
                        "Vietnamese Text must set "
                        f"font={required_text_font!r} for reliable glyph coverage"
                    )
            for keyword in node.keywords:
                if (
                    isinstance(node.func, ast.Name)
                    and node.func.id == "Text"
                    and keyword.arg == "size"
                ):
                    errors.append(
                        "Text() does not support `size`; use `font_size` or `.scale()`"
                    )
                if keyword.arg == "rate_functions":
                    errors.append(
                        "Use the `rate_func` keyword (singular), not `rate_functions`"
                    )
                if (
                    keyword.arg == "rate_func"
                    and isinstance(keyword.value, ast.Name)
                    and keyword.value.id.startswith("ease_")
                ):
                    errors.append(
                        f"Use rate_functions.{keyword.value.id} instead of bare "
                        f"{keyword.value.id}"
                    )
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            errors.append(f"Blocked dunder access: {node.attr}")
        elif isinstance(node, ast.Name) and node.id.startswith("__"):
            errors.append(f"Blocked dunder name: {node.id}")
        elif isinstance(node, ast.ClassDef):
            if any(_base_name(base) in _SCENE_BASES for base in node.bases):
                scene_classes.append(node.name)

    if not imported_manim:
        errors.append("Scene must import Manim")
    if len(scene_classes) != 1:
        errors.append(
            f"Expected exactly one Manim Scene subclass, found {len(scene_classes)}"
        )
    if not has_play:
        errors.append("Scene contains no animation via self.play()")
    if not has_wait:
        warnings.append("Scene contains no pacing via self.wait()")
    if "TODO" in code or "pass  #" in code:
        errors.append("Scene contains a placeholder")

    return ValidationReport(
        valid=not errors,
        scene_name=scene_classes[0] if len(scene_classes) == 1 else None,
        errors=list(dict.fromkeys(errors)),
        warnings=list(dict.fromkeys(warnings)),
    )
