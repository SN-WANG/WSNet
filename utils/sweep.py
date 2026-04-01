"""
Project sweep utilities for WSNet.

This module can clean Python cache artifacts and print a project tree.
"""

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

    if max_depth is not None and current_depth > max_depth:
        return ""

    ignore_dirs = _normalize_ignore_dirs(ignore_dirs)
    ignore_patterns = _normalize_ignore_patterns(ignore_patterns)
    lines: List[str] = []

    try:
        filtered_items = _list_tree_items(path, ignore_dirs, ignore_patterns)
    except PermissionError:
        return f"{prefix}[Permission Denied]\n"

    for item_index, item in enumerate(filtered_items):
        is_last = item_index == len(filtered_items) - 1
        connector = "└── " if is_last else "├── "
        next_prefix = "    " if is_last else "│   "

        lines.append(f"{prefix}{connector}{item.name}")

        if item.is_dir():
            sub_tree = generate_tree(
                item,
                prefix + next_prefix,
                ignore_dirs,
                ignore_patterns,
                max_depth,
                current_depth + 1,
            )
            if sub_tree:
                lines.append(sub_tree.rstrip())

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


def copy_to_clipboard(text: str) -> bool:
    """
    Copy text to the system clipboard using a native command.

    Args:
        text (str): Text to copy.

    Returns:
        bool: Whether the copy command succeeded.
    """
    cmd = _resolve_clipboard_command()
    if cmd is None:
        return False

    try:
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            close_fds=True,
            universal_newlines=True,
        )
        process.communicate(input=text)

        if process.returncode == 0:
            return True

        print(f"  ⚠️  Clipboard command failed with code: {process.returncode}")
        return False

    except Exception as error:
        print(f"  ⚠️  Failed to copy to clipboard: {error}")
        return False


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
    print("=" * 60)
    print("WSNET Project Utilities")
    print("=" * 60)

    print(f"\n📍 Setting up project root (depth={relative_depth})...")
    project_root = setup_project_root(relative_depth=relative_depth)
    print(f"   Project root: {project_root}")

    if auto_clean:
        print("\n🧹 Cleaning Python artifacts...")
        removed = clean_python_artifacts(target_dir=project_root, verbose=True)
        print(f"   Total removed: {len(removed)} items")

    tree_string = ""
    if print_structure:
        print("\n📂 Generating project structure...")
        tree_string = print_tree(directory=project_root, max_depth=max_tree_depth)

        if copy_clipboard:
            print("\n📋 Copying to clipboard...")
            if copy_to_clipboard(tree_string):
                print("   ✅ Successfully copied to clipboard!")
            else:
                print("   ❌ Failed to copy to clipboard")
                print("   💡 Tree is printed above - manually copy if needed")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main(
        relative_depth=2,
        auto_clean=True,
        print_structure=True,
        copy_clipboard=True,
        max_tree_depth=None,
    )
