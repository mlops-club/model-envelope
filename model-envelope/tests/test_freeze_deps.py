import os
import shutil
import subprocess
from contextlib import contextmanager
from pathlib import Path
from textwrap import dedent
from typing import Generator, List

import pytest

THIS_DIR = Path(__file__).parent
ARTIFACTS_DIR = THIS_DIR / "artifacts"
MODEL_ENVELOPE_PACKAGE_DIR = (THIS_DIR / "../").resolve()


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


def test__freeze_deps(artifacts_dir: Path):
    with create_project(artifacts_dir / "basic", ["pandas", "requests"]) as project_dir:
        # 1. create a file that calls freeze_deps and writes the result to a file
        # 2. run the file and check that the output is correct
        # ---

        # 1. create a file that calls freeze_deps and writes the result to a file
        freeze_deps_file = project_dir / "freeze_deps.py"
        freeze_deps_file.write_text(
            dedent(
                f"""\
                from model_envelope.freeze_deps import get_python_deps
                import json
                from pathlib import Path

                output_file = Path("{artifacts_dir}/freeze_deps_output.json")
                deps_json = json.dumps(list(get_python_deps()), indent=2)
                output_file.write_text(deps_json)
                """
            )
        )

        subprocess.run(
            ["uv", "run", str(freeze_deps_file)],
            cwd=project_dir,
            check=True,
        )
