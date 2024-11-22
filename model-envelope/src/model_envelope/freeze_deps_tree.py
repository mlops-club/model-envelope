from copy import deepcopy
from typing import Dict, Sequence, Set

from pipdeptree._discovery import get_installed_distributions
from pipdeptree._models import PackageDAG


def pip_freeze() -> Dict[str, str]:
    """
    Get a dictionary of installed packages and their versions using pipdeptree.

    Returns:
        Dict mapping package names to versions
    """
    pkgs = get_installed_distributions()
    tree = PackageDAG.from_pkgs(pkgs)
    return {pkg.project_name: pkg.version for pkg in tree.keys()}


def pip_graph() -> Dict[str, Dict]:
    """
    Get a nested dictionary representing the dependency graph using pipdeptree.

    Returns:
        Nested dict representing the dependency tree
    """
    # Get all installed packages
    pkgs = get_installed_distributions()
    tree = PackageDAG.from_pkgs(pkgs)

    # Convert pipdeptree format to our dict format
    def convert_node(pkg) -> Dict[str, Dict]:
        pkg_spec = f"{pkg.project_name}=={pkg.version}"
        deps = {}
        for dep in tree[pkg]:
            if dep.dist:  # Only include dependencies that are installed
                dep_spec = f"{dep.dist.project_name}=={dep.dist.version}"
                deps[dep_spec] = convert_node(dep.dist)
        return deps

    graph = {}
    for pkg in tree.keys():
        pkg_spec = f"{pkg.project_name}=={pkg.version}"
        graph[pkg_spec] = convert_node(pkg)

    return graph


def get_package_key(package_spec: str) -> str:
    """Extract the package name without version or other qualifiers"""
    # Handle various package specification formats
    if " @ " in package_spec:  # git/url installs
        package_spec = package_spec.split(" @ ")[0]
    if "==" in package_spec:  # version specifier
        package_spec = package_spec.split("==")[0]
    if ">=" in package_spec:  # minimum version
        package_spec = package_spec.split(">=")[0]
    if "<=" in package_spec:  # maximum version
        package_spec = package_spec.split("<=")[0]
    if "~=" in package_spec:  # compatible release
        package_spec = package_spec.split("~=")[0]
    return package_spec.strip()


def remove_dependency(graph: Dict[str, Dict], package_name: str) -> Dict[str, Dict]:
    """
    Remove a package and its unique dependencies from the dependency graph.
    Dependencies that are required by other packages are preserved.

    Args:
        graph: Dependency graph (treated as immutable)
        package_name: Package name to remove (without version)

    Returns:
        New dependency graph with package and its unique dependencies removed
    """

    def get_all_deps(pkg: str, seen: Set[str] = None) -> Set[str]:
        """Get all dependencies (direct and transitive) for a package"""
        if seen is None:
            seen = set()
        pkg_key = get_package_key(pkg)
        matching_pkg = next((p for p in graph if get_package_key(p) == pkg_key), None)
        if not matching_pkg or matching_pkg in seen:
            return seen
        seen.add(matching_pkg)
        for dep in graph[matching_pkg]:
            get_all_deps(dep, seen)
        return seen

    def get_all_reverse_deps(pkg: str) -> Set[str]:
        """Get all packages that depend on the given package"""
        pkg_key = get_package_key(pkg)
        reverse_deps = set()
        for parent, deps in graph.items():
            if any(get_package_key(d) == pkg_key for d in deps) or any(
                pkg_key == get_package_key(d) for dep in deps for d in get_all_deps(dep)
            ):
                reverse_deps.add(parent)
        return reverse_deps

    # Create a deep copy to ensure no modification of input
    new_graph = deepcopy(graph)

    # Find the full package spec (with version) in the graph
    package_to_remove = next(
        (pkg for pkg in graph if get_package_key(pkg) == package_name), None
    )
    if not package_to_remove:
        return new_graph

    # Get all dependencies of the package to remove
    deps_to_check = get_all_deps(package_to_remove)

    # Remove the target package
    del new_graph[package_to_remove]

    # For each dependency, check if it's safe to remove
    for dep in deps_to_check:
        if get_package_key(dep) == package_name:
            continue

        # Get all packages that depend on this dependency
        reverse_deps = get_all_reverse_deps(dep)

        # Remove package we're removing from reverse deps
        reverse_deps = {
            rd for rd in reverse_deps if get_package_key(rd) != package_name
        }

        # Remove dependency if it's only used by the package we're removing
        if not reverse_deps and dep in new_graph:
            del new_graph[dep]

    return new_graph


def get_python_deps(exclude: Sequence[str] = ()) -> Set[str]:
    """
    Get Python dependencies, optionally excluding specified packages and their unique dependencies.

    Excluding a package will exclude its entire subtree of dependencies.

    Args:
        exclude: Sequence of package names to exclude

    Returns:
        Set of remaining package specifications (name==version)
    """
    deps_graph = pip_graph()

    # Remove each excluded package and its unique dependencies
    for pkg in exclude:
        deps_graph = remove_dependency(deps_graph, pkg)

    def recursively_get_keys(d: Dict[str, Dict]) -> Set[str]:
        """Recursively get all keys from a nested dictionary"""
        keys = set()
        for k, v in d.items():
            keys.add(k)
            keys.update(recursively_get_keys(v))
        return keys

    return recursively_get_keys(deps_graph)


if __name__ == "__main__":
    from rich import print

    # Example: exclude common dev dependencies
    common_dev_libs = [
        "rich",
        "build",
        "packaging",
        "wheel",
        "tomli",
        "setuptools",
        "pip",
        "model-envelope",
    ]
    print(get_python_deps(exclude=common_dev_libs))
