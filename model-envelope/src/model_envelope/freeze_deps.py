import importlib.metadata
import inspect
from pathlib import Path
from typing import Sequence, Set

from pip._vendor.packaging.requirements import Requirement
from pipdeptree._discovery import get_installed_distributions
from pipdeptree._models import PackageDAG

# todo, make this support glob-like wildcard patterns, e.g. pytest*, mypy*, etc.
COMMON_DEV_LIBS = [
    # development
    "pip",
    "setuptools",
    "hatch",
    "poetry",
    "wheel",
    "build",
    "packaging",
    "pre-commit",
    "jupyterlab",
    "ruff",
    "black",
    "isort",
    "mypy",
    "pytest",
    "pytest-cov",
    "pytest-mock",
    "pytest-asyncio",
    "pytest-timeout",
    "pytest-xdist",
    "pytest-mock",
    "pytest-asyncio",
    "pytest-timeout",
    "pytest-xdist",
    # serving
    "fastapi",
    "uvicorn",
    "gunicorn",
    # preprocessing deps that are redundant with serving
    "psycopg",
    "sqlalchemy",
    "snowflake-connector-python",
    # mlflow and tracking
    "mlflow",
    "mlflow-skinny",
    "model-envelope",
    # misc
    "bump2version",
    "isoduration",
    "mpmath",
    "fqdn",
    "uri-template",
    "webcolors",
]


def parse_package_name(package_spec: str) -> str:
    """Extract the package name without version or other qualifiers"""
    try:
        # Use pip's Requirement parser to handle all package spec formats
        return Requirement(package_spec).name
    except Exception:
        # Fallback to simple parsing if the spec can't be parsed
        if " @ " in package_spec:  # git/url installs
            package_spec = package_spec.split(" @ ")[0]
        if "==" in package_spec:  # version specifier
            package_spec = package_spec.split("==")[0]
        return package_spec.strip()


def get_python_deps(
    exclude: Sequence[str] | None = None,
    exclude_common_libs: bool = True,
    exclude_current_package_but_include_its_deps: bool = True,
) -> Set[str]:
    """
    Get Python dependencies, optionally excluding specified packages and their dependency subtrees.

    Args:
        exclude: Sequence of package names to exclude
        exclude_common_libs: Whether to exclude common development libraries
        exclude_current_package_but_include_its_deps: e.g. if train.py lives in a custom package called
            `train-pytorch`, then exclude the name `train-pytorch` but include all of
            its dependencies.

    Returns:
        Set of remaining package specifications (name==version)
    """
    exclude = exclude or []
    exclude = list(exclude)
    if exclude_common_libs:
        exclude.extend(COMMON_DEV_LIBS)

    # Get all installed packages
    pkgs = get_installed_distributions()
    tree = PackageDAG.from_pkgs(pkgs)

    # Get initial set of all packages
    deps = {f"{pkg.project_name}=={pkg.version}" for pkg in tree.keys()}

    # Remove each excluded package and its dependency subtree
    for pkg_to_exclude in exclude:
        deps -= get_dependency_subtree(tree, pkg_to_exclude)

    # Remove current package if requested
    if exclude_current_package_but_include_its_deps:
        current_pkg_name = get_current_package_name()
        if current_pkg_name:
            # Create new set excluding current package
            deps = {dep for dep in deps if parse_package_name(dep) != current_pkg_name}

    return deps


def write_graph_to_text_file(
    path: Path,
    exclude: list[str] | None = None,
    exclude_common_libs: bool = True,
) -> None:
    """
    Write a text representation of the dependency graph to a file.

    The format matches the Unix 'tree' command style.
    """
    exclude = exclude or []
    exclude = list(exclude)
    if exclude_common_libs:
        exclude.extend(COMMON_DEV_LIBS)

    # Get all installed packages
    pkgs = get_installed_distributions()
    tree = PackageDAG.from_pkgs(pkgs)

    # Remove excluded packages and their subtrees
    to_exclude = set()
    for pkg_name in exclude:
        to_exclude.update(get_dependency_subtree(tree, pkg_name))

    # Filter the tree
    tree = tree.filter_nodes(
        include=None, exclude={pkg.split("==")[0] for pkg in to_exclude}
    )

    def render_subtree(pkg, prefix="", is_last=True) -> str:
        """Recursively render a package's dependencies"""
        # Choose the correct prefix characters
        subbranch = "    " if is_last else "│   "

        lines = []
        # Get and sort dependencies
        deps = sorted(tree[pkg], key=lambda x: x.project_name)

        # Add each dependency
        for i, dep in enumerate(deps):
            if dep.dist:  # Only show installed dependencies
                branch = "└── " if i == len(deps) - 1 else "├── "
                lines.append(
                    f"{prefix}{branch}{dep.dist.project_name}=={dep.dist.version}"
                )
                # Add nested dependencies with proper indentation
                subdeps = tree[dep.dist]
                if subdeps:
                    for line in render_subtree(
                        dep.dist,
                        prefix=prefix + subbranch,
                        is_last=(i == len(deps) - 1),
                    ).split("\n"):
                        if line:  # Skip empty lines
                            lines.append(line)

        return "\n".join(lines)

    # Start with root packages (those that aren't dependencies of others)
    all_deps = {
        dep.project_name for pkg in tree.keys() for dep in tree[pkg] if dep.dist
    }
    roots = sorted(
        [pkg for pkg in tree.keys() if pkg.project_name not in all_deps],
        key=lambda x: x.project_name,
    )

    # Build the complete tree text
    text = []
    for root in roots:
        # Add root package name without any prefix
        text.append(f"{root.project_name}=={root.version}")
        # Add its dependency subtree
        subtree = render_subtree(root)
        if subtree:
            text.append(subtree)

    # Write to file
    path.write_text("\n".join(text))


def get_dependency_subtree(tree: PackageDAG, pkg_name: str) -> Set[str]:
    """
    Get a package and its entire dependency subtree from a PackageDAG.

    Args:
        tree: The package dependency graph
        pkg_name: Name of the package to get subtree for

    Returns:
        Set of package specifications (name==version) including the package and all its dependencies
    """
    # Find the package in the tree
    pkg = next((p for p in tree.keys() if p.project_name == pkg_name), None)
    if not pkg:
        return set()

    # Get all dependencies recursively
    to_remove = {f"{pkg.project_name}=={pkg.version}"}
    stack = list(tree[pkg])  # Get direct dependencies

    while stack:
        dep = stack.pop()
        if dep.dist:  # Only process installed dependencies
            dep_spec = f"{dep.dist.project_name}=={dep.dist.version}"
            if dep_spec not in to_remove:  # Avoid cycles
                to_remove.add(dep_spec)
                stack.extend(tree[dep.dist])  # Add this dep's dependencies to process

    return to_remove


def get_current_package_name() -> str | None:
    """
    Get the distribution name of the package that this function is being called from.
    If this function thinks it's being called from model-envelope, it looks one frame
    further back to find the actual calling package.

    Returns:
        Package distribution name (e.g. 'train-pytorch') or None if not found
    """

    def get_package_from_frame(frame) -> str | None:
        if not frame:
            return None

        # Get the module that called this function
        module = inspect.getmodule(frame)
        if not module or not module.__file__:
            return None

        module_path = Path(module.__file__)

        # Look for the src directory
        for parent in module_path.parents:
            if parent.name == "src":
                # Get the next directory after src
                pkg_dir = next(parent.iterdir())
                if pkg_dir.is_dir() and not pkg_dir.name.startswith("_"):
                    # Convert the directory name to distribution name
                    dist_name = get_distribution_name(pkg_dir.name)
                    return dist_name if dist_name else pkg_dir.name

        return None

    # Get the calling frame (skipping this function's frame)
    frame = inspect.currentframe()
    try:
        frame = frame.f_back  # Get the caller's frame
        pkg_name = get_package_from_frame(frame)

        # If we think we're in model-envelope, look one frame further back
        if pkg_name == "model-envelope" and frame:
            pkg_name = get_package_from_frame(frame.f_back)

        return pkg_name

    finally:
        del frame  # Avoid reference cycles


def get_distribution_name(package_name: str) -> str | None:
    try:
        return importlib.metadata.distribution(package_name).metadata["Name"]
    except importlib.metadata.PackageNotFoundError:
        return None


if __name__ == "__main__":
    write_graph_to_text_file(
        Path("requirements-graph-full.txt"), exclude_common_libs=False
    )
    write_graph_to_text_file(
        Path("requirements-graph-model.txt"),
        exclude_common_libs=False,
        exclude=["model-envelope"],
    )

    from rich import print

    # Show what gets removed when excluding 'rich'
    original = get_python_deps()
    after_removal = get_python_deps(exclude=["model-envelope"])
    print(original - after_removal)

    print(get_current_package_name())
