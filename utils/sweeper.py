# Project sweeper cleaning Python cache artifacts and printing a project tree.
# Author: Shengning Wang

import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Set, Union

PathLike = Union[str, Path]

BYTECODE_PATTERNS = ("*.pyc", "*.pyo")
DEFAULT_IGNORE_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    ".pytest_cache", ".mypy_cache", ".tox", ".idea", ".vscode",
    "dist", "build", "*.egg-info",
}
DEFAULT_IGNORE_PATTERNS = {"*.pyc", "*.pyo", ".DS_Store", "*.log"}
APP_TITLE = "WSNET Project Utilities"
BANNER_WIDTH = 60


# ============================================================
# Project Root
# ============================================================

def setup_project_root(relative_depth: int = 2) -> Path:
    """
    Add the inferred project root to sys.path and return it.

    Args:
        relative_depth (int): Number of parent levels to step up from this file.

    Returns:
        Path: Resolved project root path.
    """
    current_file = Path(__file__).resolve()
    project_root = current_file.parents[relative_depth - 1]

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    return project_root


# ============================================================
# Cleanup
# ============================================================

def _remove_matching_files(base_path: Path, patterns: List[str], verbose: bool) -> List[Path]:
    removed_items: List[Path] = []

    for pattern in patterns:
        for file_path in base_path.rglob(pattern):
            try:
                file_path.unlink()
                removed_items.append(file_path)
                if verbose:
                    print(f"  🗑️  Removed file: {file_path.relative_to(base_path)}")
            except (OSError, PermissionError) as error:
                if verbose:
                    print(f"  ⚠️  Failed to remove {file_path}: {error}")

    return removed_items


def _remove_cache_dirs(base_path: Path, verbose: bool) -> List[Path]:
    removed_items: List[Path] = []

    for cache_dir in base_path.rglob("__pycache__"):
        if not cache_dir.is_dir():
            continue
        try:
            shutil.rmtree(cache_dir)
            removed_items.append(cache_dir)
            if verbose:
                print(f"  🗑️  Removed dir:  {cache_dir.relative_to(base_path)}/")
        except (OSError, PermissionError) as error:
            if verbose:
                print(f"  ⚠️  Failed to remove {cache_dir}: {error}")

    return removed_items


def clean_python_artifacts(target_dir: PathLike = ".", verbose: bool = True) -> List[Path]:
    """
    Remove Python bytecode files and cache directories under target_dir.

    Args:
        target_dir (PathLike): Root directory to clean.
        verbose (bool): Whether to print cleanup progress.

    Returns:
        List[Path]: Removed files and directories.
    """
    base_path = Path(target_dir).resolve()
    removed_items: List[Path] = []

    removed_items.extend(_remove_matching_files(base_path, list(BYTECODE_PATTERNS), verbose))
    removed_items.extend(_remove_cache_dirs(base_path, verbose))

    if verbose:
        print(f"\n✅ Cleanup complete: {len(removed_items)} items removed from {base_path.name}/")

    return removed_items


# ============================================================
# Tree Rendering
# ============================================================

def _normalize_ignore_dirs(ignore_dirs: Optional[Set[str]]) -> Set[str]:
    return set(DEFAULT_IGNORE_DIRS) if ignore_dirs is None else ignore_dirs


def _normalize_ignore_patterns(ignore_patterns: Optional[Set[str]]) -> Set[str]:
    return set(DEFAULT_IGNORE_PATTERNS) if ignore_patterns is None else ignore_patterns


def _matches_ignore_pattern(name: str, ignore_patterns: Set[str]) -> bool:
    for pattern in ignore_patterns:
        stripped_pattern = pattern.replace("*", "")
        if name.endswith(stripped_pattern) or name == stripped_pattern:
            return True
    return False


def _should_skip_item(item: Path, ignore_dirs: Set[str], ignore_patterns: Set[str]) -> bool:
    name = item.name

    if name.startswith(".") and name != ".env":
        return True

    if item.is_dir() and name in ignore_dirs:
        return True

    return _matches_ignore_pattern(name, ignore_patterns)


def _list_tree_items(path: Path, ignore_dirs: Set[str], ignore_patterns: Set[str]) -> List[Path]:
    items = list(path.iterdir())
    filtered_items = [item for item in items if not _should_skip_item(item, ignore_dirs, ignore_patterns)]
    filtered_items.sort(key=lambda item: (not item.is_dir(), item.name.lower()))
    return filtered_items


def _build_tree_lines(
    path: Path,
    prefix: str,
    ignore_dirs: Set[str],
    ignore_patterns: Set[str],
    max_depth: Optional[int],
    current_depth: int,
) -> List[str]:
    lines: List[str] = []

    if max_depth is not None and current_depth > max_depth:
        return lines

    try:
        tree_items = _list_tree_items(path, ignore_dirs, ignore_patterns)
    except PermissionError:
        return [f"{prefix}[Permission Denied]"]

    for item_index, item in enumerate(tree_items):
        is_last = item_index == len(tree_items) - 1
        connector = "└── " if is_last else "├── "
        child_prefix = prefix + ("    " if is_last else "│   ")

        lines.append(f"{prefix}{connector}{item.name}")

        if item.is_dir():
            lines.extend(
                _build_tree_lines(
                    item,
                    child_prefix,
                    ignore_dirs,
                    ignore_patterns,
                    max_depth,
                    current_depth + 1,
                )
            )

    return lines


def generate_tree(
    directory: PathLike,
    prefix: str = "",
    ignore_dirs: Optional[Set[str]] = None,
    ignore_patterns: Optional[Set[str]] = None,
    max_depth: Optional[int] = None,
    current_depth: int = 0,
) -> str:
    """
    Generate a formatted tree string for a directory.

    Args:
        directory (PathLike): Root directory to render.
        prefix (str): Prefix used by recursive calls.
        ignore_dirs (Optional[Set[str]]): Directory names to skip.
        ignore_patterns (Optional[Set[str]]): File patterns to skip.
        max_depth (Optional[int]): Maximum traversal depth.
        current_depth (int): Current recursion depth.

    Returns:
        str: Directory tree content without the root line.

    Raises:
        ValueError: If directory does not exist or is not a directory.
    """
    path = Path(directory)

    if not path.exists():
        raise ValueError(f"Directory does not exist: {directory}")

    if not path.is_dir():
        raise ValueError(f"Path is not a directory: {directory}")

    ignore_dirs = _normalize_ignore_dirs(ignore_dirs)
    ignore_patterns = _normalize_ignore_patterns(ignore_patterns)
    lines = _build_tree_lines(path, prefix, ignore_dirs, ignore_patterns, max_depth, current_depth)
    return "\n".join(lines)


def print_tree(
    directory: Optional[PathLike] = None,
    root_name: Optional[str] = None,
    max_depth: Optional[int] = None,
) -> str:
    """
    Print and return the directory tree.

    Args:
        directory (Optional[PathLike]): Directory to render. Defaults to the project root.
        root_name (Optional[str]): Root display name override.
        max_depth (Optional[int]): Maximum traversal depth.

    Returns:
        str: Tree string including the root line.
    """
    if directory is None:
        directory = setup_project_root(relative_depth=2)

    path = Path(directory).resolve()
    display_name = root_name or path.name
    tree_content = generate_tree(path, max_depth=max_depth)
    full_tree = f"📁 {display_name}/\n{tree_content}"

    print(full_tree)
    return full_tree


# ============================================================
# Clipboard And Entry Point
# ============================================================

def _resolve_clipboard_command() -> Optional[List[str]]:
    if sys.platform == "darwin":
        return ["pbcopy"]

    if sys.platform == "win32":
        return ["clip"]

    if sys.platform.startswith("linux"):
        if shutil.which("wl-copy"):
            return ["wl-copy"]
        if shutil.which("xclip"):
            return ["xclip", "-selection", "clipboard"]

        print("  ⚠️  Clipboard tool not found. Install 'wl-copy' or 'xclip'.")
        return None

    print(f"  ⚠️  Clipboard not supported on platform: {sys.platform}")
    return None


def _run_clipboard_command(command: List[str], text: str) -> bool:
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            close_fds=True,
            universal_newlines=True,
        )
        process.communicate(input=text)
    except Exception as error:
        print(f"  ⚠️  Failed to copy to clipboard: {error}")
        return False

    if process.returncode == 0:
        return True

    print(f"  ⚠️  Clipboard command failed with code: {process.returncode}")
    return False


def copy_to_clipboard(text: str) -> bool:
    """
    Copy text to the system clipboard using a native command.

    Args:
        text (str): Text to copy.

    Returns:
        bool: Whether the copy command succeeded.
    """
    command = _resolve_clipboard_command()
    if command is None:
        return False

    return _run_clipboard_command(command, text)


def _print_banner() -> None:
    line = "=" * BANNER_WIDTH
    print(line)
    print(APP_TITLE)
    print(line)


def _setup_project_root_step(relative_depth: int) -> Path:
    print(f"\n📍 Setting up project root (depth={relative_depth})...")
    project_root = setup_project_root(relative_depth=relative_depth)
    print(f"   Project root: {project_root}")
    return project_root


def _cleanup_step(project_root: Path) -> List[Path]:
    print("\n🧹 Cleaning Python artifacts...")
    removed_items = clean_python_artifacts(target_dir=project_root, verbose=True)
    print(f"   Total removed: {len(removed_items)} items")
    return removed_items


def _tree_step(project_root: Path, max_tree_depth: Optional[int], copy_clipboard: bool) -> str:
    print("\n📂 Generating project structure...")
    tree_string = print_tree(directory=project_root, max_depth=max_tree_depth)

    if not copy_clipboard:
        return tree_string

    print("\n📋 Copying to clipboard...")
    if copy_to_clipboard(tree_string):
        print("   ✅ Successfully copied to clipboard!")
        return tree_string

    print("   ❌ Failed to copy to clipboard")
    print("   💡 Tree is printed above - manually copy if needed")
    return tree_string


def _print_footer() -> None:
    line = "=" * BANNER_WIDTH
    print(f"\n{line}")
    print("Done!")
    print(line)


def main(
    relative_depth: int = 2,
    auto_clean: bool = True,
    print_structure: bool = True,
    copy_clipboard: bool = True,
    max_tree_depth: Optional[int] = None,
) -> None:
    """
    Run the sweep workflow.

    Args:
        relative_depth (int): Parent depth used to infer the project root.
        auto_clean (bool): Whether to remove Python artifacts.
        print_structure (bool): Whether to print the project tree.
        copy_clipboard (bool): Whether to copy the tree to the clipboard.
        max_tree_depth (Optional[int]): Maximum depth for the tree view.
    """
    _print_banner()
    project_root = _setup_project_root_step(relative_depth)

    if auto_clean:
        _cleanup_step(project_root)

    if print_structure:
        _tree_step(project_root, max_tree_depth, copy_clipboard)

    _print_footer()


if __name__ == "__main__":
    main(
        relative_depth=2,
        auto_clean=True,
        print_structure=True,
        copy_clipboard=True,
        max_tree_depth=None,
    )
