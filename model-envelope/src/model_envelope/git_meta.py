import subprocess
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse


def _run_git_command(cmd: list[str], cwd: Optional[Path] = None) -> Optional[str]:
    """
    Run a git command and return its output.
    Returns None if the command fails or if not in a git repo.
    """
    try:
        result = subprocess.run(
            ["git"] + cmd,
            capture_output=True,
            text=True,
            cwd=cwd,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, subprocess.SubprocessError):
        return None


def try_get_git_commit() -> Optional[str]:
    """
    Try to get the current git commit hash.

    Returns:
        Full commit hash or None if not in a git repo
    """
    return _run_git_command(["rev-parse", "HEAD"])


def try_get_git_remote() -> Optional[str]:
    """
    Try to get the current git remote URL.

    Returns:
        Remote URL (usually origin) or None if not in a git repo
    """
    if remote := _run_git_command(["remote", "get-url", "origin"]):
        return remote
    return None


def try_get_git_branch() -> Optional[str]:
    """
    Try to get the current git branch name.

    Returns:
        Branch name or None if not in a git repo
    """
    if branch := _run_git_command(["rev-parse", "--abbrev-ref", "HEAD"]):
        return branch
    return None


def try_get_git_diff() -> Optional[str]:
    """
    Try to get a patch of all uncommitted changes.

    Returns:
        Diff output or None if no changes or not in a git repo
    """
    # Get both staged and unstaged changes
    staged = _run_git_command(["diff", "--cached"]) or ""
    unstaged = _run_git_command(["diff"]) or ""

    # Combine staged and unstaged changes
    diff = staged + "\n" + unstaged if staged or unstaged else None
    return diff.strip() if diff else None


def try_get_git_user() -> Optional[str]:
    """
    Try to get the current git user name and email.

    Returns:
        String in format "Name <email>" or None if not in a git repo
    """
    name = _run_git_command(["config", "user.name"])
    email = _run_git_command(["config", "user.email"])

    if name and email:
        return f"{name} <{email}>"
    return None


def try_get_git_web_url(path: Optional[str] = None) -> Optional[str]:
    """
    Try to get a web URL for the current commit/branch.

    Args:
        path: Optional path to file or directory within the repo

    Returns:
        URL to view the code on GitHub/GitLab/etc or None if not possible
    """
    remote = try_get_git_remote()
    if not remote:
        return None

    # Parse the remote URL
    parsed = urlparse(remote)
    if parsed.scheme == "":  # SSH format: git@github.com:user/repo.git
        host = parsed.path.split("@")[-1].split(":")[0]
        path_parts = parsed.path.split(":")[-1].split(".git")[0].split("/")
    else:  # HTTPS format: https://github.com/user/repo.git
        host = parsed.netloc
        path_parts = parsed.path.strip("/").split(".git")[0].split("/")

    if len(path_parts) < 2:  # Need at least user/repo
        return None

    # Get commit or branch
    commit = try_get_git_commit()
    branch = try_get_git_branch()
    ref = commit if commit else branch if branch else "main"

    # Build the URL based on the host
    if "github.com" in host:
        base_url = f"https://github.com/{path_parts[0]}/{path_parts[1]}"
        if path:
            return f"{base_url}/blob/{ref}/{path}"
        return f"{base_url}/tree/{ref}"
    elif "gitlab" in host:
        base_url = f"https://{host}/{path_parts[0]}/{path_parts[1]}"
        if path:
            return f"{base_url}/-/blob/{ref}/{path}"
        return f"{base_url}/-/tree/{ref}"

    return None


def save_git_patch(path: Path) -> Optional[Path]:
    """
    Save the current git diff as a patch file.

    Args:
        path: Path to save the patch file (should end in .patch or .diff)

    Returns:
        Path to the saved patch file or None if no changes or not in a git repo
    """
    if diff := try_get_git_diff():
        # Ensure the path has the correct extension
        path = path.with_suffix(".patch")
        path.write_text(diff)
        return path
    return None


if __name__ == "__main__":
    from rich import print

    print("Git Metadata:")
    print(f"Commit: {try_get_git_commit()}")
    print(f"Remote: {try_get_git_remote()}")
    print(f"Branch: {try_get_git_branch()}")
    print(f"User: {try_get_git_user()}")
    print(f"Web URL: {try_get_git_web_url()}")

    print("\nUncommitted Changes:")
    if patch_path := save_git_patch(Path("changes.patch")):
        print(f"Patch saved to: {patch_path}")
    else:
        print("No changes")
