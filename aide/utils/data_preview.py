"""
Contains functions to manually generate a textual preview of some common file types (.csv, .json,..) for the agent.
"""

import json
from pathlib import Path

import humanize
import pandas as pd
from genson import SchemaBuilder
from pandas.api.types import is_numeric_dtype

# these files are treated as code (e.g. markdown wrapped)
code_files = {".py", ".sh", ".yaml", ".yml", ".md", ".html", ".xml", ".log", ".rst"}
# we treat these files as text (rather than binary) files
plaintext_files = {".txt", ".csv", ".json", ".tsv"} | code_files


def get_file_len_size(f: Path) -> tuple[int, str]:
    """
    Calculate the size of a file (#lines for plaintext files, otherwise #bytes)
    Also returns a human-readable string representation of the size.
    """
    if f.suffix in plaintext_files:
        # Try multiple encodings to handle non-UTF-8 files
        num_lines = None
        for encoding in ['utf-8', 'latin-1', 'iso-8859-1']:
            try:
                num_lines = sum(1 for _ in open(f, encoding=encoding))
                break
            except (UnicodeDecodeError, LookupError):
                continue
        
        # If all encodings fail, fall back to byte size
        if num_lines is None:
            s = f.stat().st_size
            return s, humanize.naturalsize(s)
        
        return num_lines, f"{num_lines} lines"
    else:
        s = f.stat().st_size
        return s, humanize.naturalsize(s)


def file_tree(path: Path, depth=0) -> str:
    """Generate a tree structure of files in a directory"""
    result = []
    files = [p for p in Path(path).iterdir() if not p.is_dir()]
    dirs = [p for p in Path(path).iterdir() if p.is_dir()]
    max_n = 4 if len(files) > 30 else 8
    for p in sorted(files)[:max_n]:
        result.append(f"{' ' * depth * 4}{p.name} ({get_file_len_size(p)[1]})")
    if len(files) > max_n:
        result.append(f"{' ' * depth * 4}... and {len(files) - max_n} other files")

    for p in sorted(dirs):
        result.append(f"{' ' * depth * 4}{p.name}/")
        result.append(file_tree(p, depth + 1))

    return "\n".join(result)


def _walk(path: Path):
    """Recursively walk a directory (analogous to os.walk but for pathlib.Path)"""
    for p in sorted(Path(path).iterdir()):
        if p.is_dir():
            yield from _walk(p)
            continue
        yield p


def preview_csv(p: Path, file_name: str, simple=True) -> str:
    """Generate a textual preview of a csv file

    Args:
        p (Path): the path to the csv file
        file_name (str): the file name to use in the preview
        simple (bool, optional): whether to use a simplified version of the preview. Defaults to True.

    Returns:
        str: the textual preview
    """
    # Try to read CSV with error handling for malformed data
    # Limit rows to prevent huge data previews (especially for text-heavy datasets)
    MAX_PREVIEW_ROWS = 10  # Heavily reduced to prevent 413 errors with text-heavy data like essays
    
    try:
        # First try with default settings, limit rows for preview
        df = pd.read_csv(p, nrows=MAX_PREVIEW_ROWS)
    except (pd.errors.ParserError, ValueError) as e:
        # If parsing fails, try with more robust settings
        try:
            # Use Python engine which is more forgiving, and handle quoting
            df = pd.read_csv(p, engine='python', on_bad_lines='skip', encoding_errors='ignore', nrows=MAX_PREVIEW_ROWS)
        except Exception as e2:
            # If still fails, return error message instead of crashing
            return f"-> {file_name}: Unable to parse CSV file (error: {str(e)[:100]})"

    out = []

    out.append(f"-> {file_name} has {df.shape[0]} rows and {df.shape[1]} columns.")

    if simple:
        cols = df.columns.tolist()
        sel_cols = 15
        cols_str = ", ".join(cols[:sel_cols])
        res = f"The columns are: {cols_str}"
        if len(cols) > sel_cols:
            res += f"... and {len(cols) - sel_cols} more columns"
        out.append(res)
    else:
        out.append("Here is some information about the columns:")
        for col in sorted(df.columns):
            dtype = df[col].dtype
            name = f"{col} ({dtype})"

            nan_count = df[col].isnull().sum()

            if dtype == "bool":
                v = df[col][df[col].notnull()].mean()
                out.append(f"{name} is {v * 100:.2f}% True, {100 - v * 100:.2f}% False")
            elif df[col].nunique() < 10:
                out.append(
                    f"{name} has {df[col].nunique()} unique values: {df[col].unique().tolist()}"
                )
            elif is_numeric_dtype(df[col]):
                out.append(
                    f"{name} has range: {df[col].min():.2f} - {df[col].max():.2f}, {nan_count} nan values"
                )
            elif dtype == "object":
                # Truncate long text values to prevent huge previews (especially essays/reviews)
                example_values = df[col].value_counts().head(2).index.tolist()  # Reduced from 4 to 2 examples
                truncated_examples = [str(v)[:50] + '...' if len(str(v)) > 50 else str(v) for v in example_values]  # Reduced from 100 to 50 chars
                out.append(
                    f"{name} has {df[col].nunique()} unique values. Example: {truncated_examples[0] if truncated_examples else 'N/A'}"
                )

    return "\n".join(out)


def preview_json(p: Path, file_name: str):
    """Generate a textual preview of a json file using a generated json schema.
    Supports both standard JSON and JSONL (JSON Lines) formats.
    """
    builder = SchemaBuilder()
    
    try:
        with open(p, 'r', encoding='utf-8') as f:
            # Try to detect JSONL format (JSON Lines - one object per line)
            first_line = f.readline().strip()
            if not first_line:
                return f"-> {file_name}: Empty file"
            
            # Try parsing first line as JSON
            try:
                first_obj = json.loads(first_line)
                # Check if there's a second line (indicates JSONL format)
                second_line = f.readline().strip()
                is_jsonl = bool(second_line)
                
                if is_jsonl:
                    # JSONL format: parse each line as a separate JSON object
                    f.seek(0)  # Reset to beginning
                    line_count = 0
                    max_lines = 100  # Limit number of lines to process for schema generation
                    
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                            builder.add_object(obj)
                            line_count += 1
                            if line_count >= max_lines:
                                break
                        except json.JSONDecodeError:
                            continue
                    
                    schema = builder.to_json(indent=2)
                    return f"-> {file_name} is a JSONL file (JSON Lines format) with auto-generated json schema:\n{schema}"
                else:
                    # Single JSON object on first line, but no second line - treat as single object
                    builder.add_object(first_obj)
            except json.JSONDecodeError:
                # First line is not valid JSON, try as standard JSON file
                f.seek(0)
                try:
                    data = json.load(f)
                    builder.add_object(data)
                except json.JSONDecodeError as e:
                    return f"-> {file_name}: Unable to parse JSON file (error: {str(e)[:100]})"
        
        return f"-> {file_name} has auto-generated json schema:\n" + builder.to_json(indent=2)
    except Exception as e:
        return f"-> {file_name}: Error reading file (error: {str(e)[:100]})"


def generate(base_path, include_file_details=True, simple=True):
    """
    Generate a textual preview of a directory, including an overview of the directory
    structure and previews of individual files
    """
    tree = f"```\n{file_tree(base_path)}```"
    out = [tree]

    if include_file_details:
        for fn in _walk(base_path):
            file_name = str(fn.relative_to(base_path))

            if fn.suffix == ".csv":
                out.append(preview_csv(fn, file_name, simple=simple))
            elif fn.suffix == ".json":
                out.append(preview_json(fn, file_name))
            elif fn.suffix in plaintext_files:
                # Read all plaintext files (no arbitrary line limit)
                try:
                    with open(fn, encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        if fn.suffix in code_files:
                            content = f"```\n{content}\n```"
                        out.append(f"-> {file_name} has content:\n\n{content}")
                except Exception as e:
                    out.append(f"-> {file_name}: Error reading file ({str(e)[:50]})")

    result = "\n\n".join(out)

    # if the result is very long we generate a simpler version
    if len(result) > 6_000 and not simple:
        return generate(
            base_path, include_file_details=include_file_details, simple=True
        )

    return result
