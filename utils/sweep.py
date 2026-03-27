"""
Sweep for wsnet, Combined functionality for:
1. Cleaning Python cache files (__pycache__, *.pyc, *.pyo)
2. Generating project directory tree with copy-to-clipboard support

Author: Shengning Wang
"""

import sys
import pathlib
import shutil
import subprocess
from typing import List, Union, Optional, Set
from pathlib import Path


def setup_project_root(relative_depth: int = 2) -> pathlib.Path:
    """
    Configures the system path to include the project root directory.
    
    Args:
        relative_depth (int): The number of levels to traverse up from current 
            file to find project root. 
            Default is 2 (e.g., from wsnet/utils/project_utils.py to wsnet/).

    Returns:
        project_root (pathlib.Path): The absolute path to the project root.
    """
    current_file: pathlib.Path = pathlib.Path(__file__).resolve()
    project_root: pathlib.Path = current_file.parents[relative_depth - 1]

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    return project_root


def clean_python_artifacts(
    target_dir: Union[str, pathlib.Path] = ".",
    verbose: bool = True
) -> List[pathlib.Path]:
    """
    Recursively removes Python compilation artifacts and cache directories.

    Targets:
        1. *.pyc and *.pyo files (compiled bytecode).
        2. __pycache__ directories.

    Args:
        target_dir (Union[str, pathlib.Path]): The root directory to start cleaning.
            Default is the current working directory.
        verbose (bool): Whether to print cleanup progress. Default is True.

    Returns:
        removed_items (List[pathlib.Path]): List of all removed files and directories.
    """
    base_path: pathlib.Path = pathlib.Path(target_dir).resolve()
    removed_items: List[pathlib.Path] = []

    # 1. Remove compiled bytecode files
    # Shape: N files matching the glob pattern
    bytecode_patterns: List[str] = ["*.pyc", "*.pyo"]
    for pattern in bytecode_patterns:
        for file_path in base_path.rglob(pattern):
            try:
                file_path.unlink()
                removed_items.append(file_path)
                if verbose:
                    print(f"  🗑️  Removed file: {file_path.relative_to(base_path)}")
            except (OSError, PermissionError) as e:
                if verbose:
                    print(f"  ⚠️  Failed to remove {file_path}: {e}")

    # 2. Remove cache directories
    # Shape: M directories matching the name "__pycache__"
    for cache_dir in base_path.rglob("__pycache__"):
        if cache_dir.is_dir():
            try:
                shutil.rmtree(cache_dir)
                removed_items.append(cache_dir)
                if verbose:
                    print(f"  🗑️  Removed dir:  {cache_dir.relative_to(base_path)}/")
            except (OSError, PermissionError) as e:
                if verbose:
                    print(f"  ⚠️  Failed to remove {cache_dir}: {e}")

    if verbose:
        print(f"\n✅ Cleanup complete: {len(removed_items)} items removed from {base_path.name}/")

    return removed_items


def generate_tree(
    directory: Union[str, pathlib.Path],
    prefix: str = "",
    ignore_dirs: Optional[Set[str]] = None,
    ignore_patterns: Optional[Set[str]] = None,
    max_depth: Optional[int] = None,
    current_depth: int = 0
) -> str:
    """
    Generate a tree structure string of the directory.

    Args:
        directory: Root directory to generate tree from
        prefix: Prefix string for current line (used for recursion)
        ignore_dirs: Set of directory names to ignore (default: common cache/env dirs)
        ignore_patterns: Set of file patterns to ignore (e.g., "*.pyc")
        max_depth: Maximum depth to traverse (None for unlimited)
        current_depth: Current depth in recursion (internal use)

    Returns:
        tree_string (str): Formatted tree structure

    Raises:
        ValueError: If directory does not exist or is not accessible
    """
    path = Path(directory)

    if not path.exists():
        raise ValueError(f"Directory does not exist: {directory}")

    if not path.is_dir():
        raise ValueError(f"Path is not a directory: {directory}")

    # Default ignore patterns
    if ignore_dirs is None:
        ignore_dirs = {
            ".git", "__pycache__", "node_modules", ".venv", "venv",
            ".pytest_cache", ".mypy_cache", ".tox", ".idea", ".vscode",
            "dist", "build", "*.egg-info"
        }

    if ignore_patterns is None:
        ignore_patterns = {"*.pyc", "*.pyo", ".DS_Store", "*.log"}

    # Check depth limit
    if max_depth is not None and current_depth > max_depth:
        return ""

    lines: List[str] = []

    # Get directory contents
    try:
        items = list(path.iterdir())
    except PermissionError:
        return f"{prefix}[Permission Denied]\n"

    # Filter items
    filtered_items = []
    for item in items:
        name = item.name

        # Skip hidden files (except .env files which are config)
        if name.startswith(".") and name != ".env":
            continue

        # Skip ignored directories
        if item.is_dir() and name in ignore_dirs:
            continue

        # Skip patterns
        skip = False
        for pattern in ignore_patterns:
            if name.endswith(pattern.replace("*", "")) or name == pattern.replace("*", ""):
                skip = True
                break
        if skip:
            continue

        filtered_items.append(item)

    # Sort: directories first, then files, both alphabetically
    filtered_items.sort(key=lambda x: (not x.is_dir(), x.name.lower()))

    # Generate tree
    for i, item in enumerate(filtered_items):
        is_last = (i == len(filtered_items) - 1)
        connector = "└── " if is_last else "├── "
        next_prefix = "    " if is_last else "│   "

        lines.append(f"{prefix}{connector}{item.name}")

        if item.is_dir():
            # Recurse into subdirectory
            sub_tree = generate_tree(
                item,
                prefix + next_prefix,
                ignore_dirs,
                ignore_patterns,
                max_depth,
                current_depth + 1
            )
            if sub_tree:
                lines.append(sub_tree.rstrip())

    return "\n".join(lines)


def print_tree(
    directory: Optional[Union[str, pathlib.Path]] = None,
    root_name: Optional[str] = None,
    max_depth: Optional[int] = None
) -> str:
    """
    Print and return the directory tree structure.

    Args:
        directory: Directory to print tree for (default: project root)
        root_name: Custom name for root directory (default: actual folder name)
        max_depth: Maximum depth to display

    Returns:
        full_tree (str): Complete tree string including root
    """
    if directory is None:
        directory = setup_project_root(relative_depth=2)

    path = Path(directory).resolve()
    display_name = root_name or path.name

    # Generate tree
    tree_content = generate_tree(path, max_depth=max_depth)

    # Combine root and tree
    full_tree = f"📁 {display_name}/\n{tree_content}"

    print(full_tree)
    return full_tree


def copy_to_clipboard(text: str) -> bool:
    """
    Copy text to system clipboard using native tools.

    Supports:
        - macOS (pbcopy)
        - Linux (xclip or wl-copy)
        - Windows (clip)

    Args:
        text (str): Text to copy to clipboard

    Returns:
        success (bool): True if successful, False otherwise
    """
    # Detect platform and use appropriate command
    if sys.platform == "darwin":  # macOS
        cmd = ["pbcopy"]
    elif sys.platform == "win32":  # Windows
        cmd = ["clip"]
    elif sys.platform.startswith("linux"):  # Linux
        # Try wl-copy (Wayland) first, then xclip (X11)
        if shutil.which("wl-copy"):
            cmd = ["wl-copy"]
        elif shutil.which("xclip"):
            cmd = ["xclip", "-selection", "clipboard"]
        else:
            print("  ⚠️  Clipboard tool not found. Install 'wl-copy' or 'xclip'.")
            return False
    else:
        print(f"  ⚠️  Clipboard not supported on platform: {sys.platform}")
        return False

    try:
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            close_fds=True,
            universal_newlines=True
        )
        process.communicate(input=text)

        if process.returncode == 0:
            return True
        else:
            print(f"  ⚠️  Clipboard command failed with code: {process.returncode}")
            return False

    except Exception as e:
        print(f"  ⚠️  Failed to copy to clipboard: {e}")
        return False


def main(
    relative_depth: int = 2,
    auto_clean: bool = True,
    print_structure: bool = True,
    copy_clipboard: bool = True,
    max_tree_depth: Optional[int] = None
) -> None:
    """
    Main entry point for sweep.

    Performs the following operations in order:
    1. Sets up project root in sys.path
    2. Cleans Python cache artifacts (if auto_clean=True)
    3. Generates and prints directory tree (if print_structure=True)
    4. Copies tree to clipboard (if copy_clipboard=True)

    Args:
        relative_depth (int): Levels to traverse up to find project root (default: 2)
        auto_clean (bool): Whether to clean cache files automatically (default: True)
        print_structure (bool): Whether to print tree structure (default: True)
        copy_clipboard (bool): Whether to copy tree to clipboard (default: True)
        max_tree_depth (Optional[int]): Maximum depth for tree display (default: None)

    Example:
        # From wsnet/utils/project_utils.py, clean and print tree
        >>> main(relative_depth=2)

        # From deeper location, adjust depth
        >>> main(relative_depth=3)

        # Only clean, don"t print tree
        >>> main(auto_clean=True, print_structure=False)
    """
    print("=" * 60)
    print("WSNET Project Utilities")
    print("=" * 60)

    # Step 1: Setup project root
    print(f"\n📍 Setting up project root (depth={relative_depth})...")
    project_root = setup_project_root(relative_depth=relative_depth)
    print(f"   Project root: {project_root}")

    # Step 2: Clean cache files
    if auto_clean:
        print(f"\n🧹 Cleaning Python artifacts...")
        removed = clean_python_artifacts(target_dir=project_root, verbose=True)
        print(f"   Total removed: {len(removed)} items")

    # Step 3 & 4: Generate tree and copy to clipboard
    tree_string = ""
    if print_structure:
        print(f"\n📂 Generating project structure...")
        tree_string = print_tree(
            directory=project_root,
            max_depth=max_tree_depth
        )

        if copy_clipboard:
            print(f"\n📋 Copying to clipboard...")
            if copy_to_clipboard(tree_string):
                print("   ✅ Successfully copied to clipboard!")
            else:
                print("   ❌ Failed to copy to clipboard")
                print("   💡 Tree is printed above - manually copy if needed")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    """
    Default execution: Run from <project_name>/wsnet/utils/sweep.py

    Adjust relative_depth based on file location:
    - From <project_name>/wsnet/utils/sweep.py: relative_depth=3
    - From <project_name>/wsnet/utils/tools/sweep.py: relative_depth=4
    """
    main(
        relative_depth=3,           # Adjust based on file location
        auto_clean=True,            # Clean __pycache__ and *.pyc
        print_structure=True,       # Print directory tree
        copy_clipboard=True,        # Copy tree to clipboard
        max_tree_depth=None         # No depth limit (or set to 3, 4, etc.)
    )
