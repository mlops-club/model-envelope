import json
import os
import shutil
import subprocess
from contextlib import contextmanager
from pathlib import Path
from textwrap import dedent
from typing import Generator, List, Sequence

import pytest

THIS_DIR = Path(__file__).parent
ARTIFACTS_DIR = THIS_DIR / "artifacts"
MODEL_ENVELOPE_PACKAGE_DIR = (THIS_DIR / "../").resolve()


def test__freeze_deps(artifacts_dir: Path):
    deps = ["pandas", "numpy"]

    with create_project(artifacts_dir / "basic", deps) as project_dir:
        frozen_deps = call_freeze_deps(
            project_dir=project_dir,
            artifacts_dir=artifacts_dir,
        )

        assert "pandas" in frozen_deps.keys()
        assert "numpy" in frozen_deps.keys()


def test__freeze_deps__exclude_subtree_that_contains_a_first_degree_dep(
    artifacts_dir: Path,
):
    """
    Ensure that if we exclude a subtree that contains a first-degree dependency,
    then the first-degree dependency is not excluded.

    1. install pandas
    2. install numpy
    3. pandas depends on numpy
    4. remove pandas
    5. assert numpy is still listed
    """
    deps = ["pandas", "numpy"]

    with create_project(
        artifacts_dir / "basic",
        deps,
    ) as project_dir:
        frozen_deps = call_freeze_deps(
            project_dir=project_dir,
            artifacts_dir=artifacts_dir,
            exclude=["pandas"],
        )

        assert "pandas" not in frozen_deps.keys()
        assert "numpy" in frozen_deps.keys()


@pytest.fixture
def artifacts_dir() -> Generator[Path, None, None]:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    yield ARTIFACTS_DIR


@contextmanager
def create_project(
    project_dir: Path,
    dependencies: List[str],
    python_version: str = "3.12",
) -> Generator[Path, None, None]:
    """
    Create a test project with the given dependencies.

    Args:
        project_dir: Directory to create project in
        dependencies: List of dependencies to install
        python_version: Python version to use
    """
    try:
        # Create project structure
        project_dir.mkdir(parents=True, exist_ok=True)

        # Initialize virtual environment
        subprocess.run(
            ["uv", "init", f"--python={python_version}", "--lib"],
            cwd=project_dir,
            check=True,
        )

        dependencies += ["--editable", str(MODEL_ENVELOPE_PACKAGE_DIR)]

        # Install dependencies
        subprocess.run(
            ["uv", "add"] + dependencies,
            cwd=project_dir,
            check=True,
        )

        yield project_dir

    finally:
        if not os.environ.get("KEEP_TEST_ARTIFACTS"):
            shutil.rmtree(project_dir)


def call_freeze_deps(
    project_dir: Path,
    artifacts_dir: Path,
    exclude: Sequence[str] | None = None,
    exclude_common_libs: bool = True,
    exclude_current_package_but_include_its_deps: bool = True,
) -> dict[str, str]:
    freeze_deps_file = project_dir / "freeze_deps.py"

    deps_json_fpath = artifacts_dir / "freeze_deps_output.json"
    freeze_deps_file.write_text(
        dedent(
            f"""\
            from model_envelope.freeze_deps import get_python_deps
            import json
            from pathlib import Path

            deps = get_python_deps(
                exclude={exclude},
                exclude_common_libs={exclude_common_libs},
                exclude_current_package_but_include_its_deps={exclude_current_package_but_include_its_deps},
            )

            output_file = Path("{deps_json_fpath}")
            deps_json = json.dumps(list(deps), indent=2)
            output_file.write_text(deps_json)
            """
        )
    )

    subprocess.run(
        ["uv", "run", str(freeze_deps_file)],
        cwd=project_dir,
        check=True,
    )

    frozen_deps_list = json.loads(deps_json_fpath.read_text())
    split_deps = [dep.split("==") for dep in frozen_deps_list]
    frozen_deps = {name: version for name, version in split_deps}

    deps_json_fpath.unlink(missing_ok=True)

    return frozen_deps
