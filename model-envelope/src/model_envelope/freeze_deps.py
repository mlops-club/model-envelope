from copy import deepcopy
from typing import Dict, Set


def get_python_deps(prune_common_dev_libs: bool = True) -> Set[str]:
    """Return Python dependencies."""
    deps_graph = pip_graph()

    common_dev_libs = {"rich", "build", "packaging", "wheel", "tomli"}

    for pkg in common_dev_libs:
        print(deps_graph)
        deps_graph = remove_dependency(deps_graph, pkg)
        print("Removing", pkg)
        print(deps_graph)

    def recursively_get_keys(d: Dict[str, Dict]) -> Set[str]:
        """Recursively get all keys from a nested dictionary"""
        keys = set()
        for k, v in d.items():
            keys.add(k)
            keys.update(recursively_get_keys(v))
        return keys

    return recursively_get_keys(deps_graph)


def pip_freeze() -> Dict[str, str]:
    """
    Get a dictionary of installed packages and their versions.

    Returns:
        Dict mapping package names to versions
    """
    import pkg_resources

    return {dist.key: dist.version for dist in pkg_resources.working_set}


def pip_graph() -> Dict[str, Dict]:
    """
    Get a nested dictionary representing the dependency graph of installed packages.

    Returns:
        Nested dict representing the dependency tree
    """
    import pkg_resources

    tree = {}

    # Get all installed distributions
    for dist in pkg_resources.working_set:
        name = f"{dist.key}=={dist.version}"
        deps = {}

        # Get direct dependencies
        for req in dist.requires():
            # Find the installed version of the dependency
            dep_dist = pkg_resources.working_set.find(req)
            if dep_dist:
                deps[f"{dep_dist.key}=={dep_dist.version}"] = {}

        tree[name] = deps

    return tree


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


def _build_rich_tree(deps: Dict[str, Dict], tree) -> None:
    """Helper to recursively build a rich Tree"""
    for package, subdeps in deps.items():
        branch = tree.add(package)
        if subdeps:
            _build_rich_tree(subdeps, branch)


if __name__ == "__main__":
    from rich import print

    print(get_python_deps())

# if __name__ == "__main__":
#     from rich.console import Console
#     from rich.tree import Tree

#     console = Console()

#     # Print frozen dependencies
#     console.print("\n[bold]Installed Packages:[/bold]")
#     frozen = pip_freeze()
#     for package, version in frozen.items():
#         console.print(f"{package}=={version}")

#     # Print original dependency graph
#     console.print("\n[bold]Original Dependency Graph:[/bold]")
#     graph = pip_graph()
#     tree = Tree("Dependencies")
#     _build_rich_tree(graph, tree)
#     console.print(tree)

#     remove = {"rich", "model-envelope", "build", "packaging", "wheel", "tomli"}
#     updated_graph = deepcopy(graph)
#     for pkg in remove:
#         updated_graph = remove_dependency(updated_graph, pkg)

#     updated_tree = Tree("Dependencies")
#     _build_rich_tree(updated_graph, updated_tree)
#     console.print(updated_tree)
