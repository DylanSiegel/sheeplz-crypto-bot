from pathlib import Path
from multiprocessing import Pool, cpu_count
import os
import pandas as pd
import pyarrow.parquet as pq  # Import pyarrow for Parquet support

def process_file(file_path, project_root):
    """Reads file content and returns formatted string for output."""
    relative_path = file_path.relative_to(project_root)
    try:
        if file_path.suffix == ".csv":
            # Read a snippet of the CSV (first few rows)
            df = pd.read_csv(file_path, nrows=5)  # Read only first 5 rows for context
            snippet = df.to_string(index=False)
            return f"// File: {relative_path}\n// Snippet (CSV):\n{snippet}\n\n"
        elif file_path.suffix == ".parquet":
            # Read a snippet of the Parquet file (first few rows)
            try:
                table = pq.read_table(file_path)  # Read Parquet table
                df = table.slice(0, 5).to_pandas()  # Get first 5 rows and convert to pandas
                snippet = df.to_string(index=False)  # Get string representation
                return f"// File: {relative_path}\n// Snippet (Parquet):\n{snippet}\n\n"
            except Exception as e:
                return f"// File: {relative_path} (Error reading Parquet: {e})\n\n"
        else:
            with file_path.open("r", encoding="utf-8") as infile:
                return f"// File: {relative_path}\n{infile.read()}\n\n"
    except UnicodeDecodeError:
        return f"// File: {relative_path} (binary file)\n\n"
    except Exception as e:
        print(f"Error processing file {relative_path}: {e}")
        return None

def combine_code(project_root, output_file, ignore_dirs, ignore_files):
    """Combines code from the project into a single output file, ignoring specified directories and files."""
    relevant_extensions = (".py", ".yaml", ".yml", ".csv", ".parquet")  # Relevant file extensions
    project_root = Path(project_root)  # Convert to Path object

    file_paths = []  # Collect file paths for both tree and processing
    file_tree_output = ["**File Tree (Relevant Files Only)**\n"]

    for root, dirs, files in os.walk(project_root):
        dirs[:] = [d for d in dirs if d not in ignore_dirs]
        root_path = Path(root)
        relevant_files = [file for file in files if file not in ignore_files and file.endswith(relevant_extensions)]

        if relevant_files:
            file_tree_output.append(f"  {root_path.relative_to(project_root)}\n")
            for file in relevant_files:
                file_tree_output.append(f"    - {file}\n")
                file_paths.append(root_path / file)

    with open(output_file, "w", encoding="utf-8") as outfile:
        outfile.writelines(file_tree_output)  # Write the file tree first
        with Pool(processes=cpu_count()) as pool:  # Use multiprocessing.Pool
            results = pool.starmap(process_file, [(file_path, project_root) for file_path in file_paths])  # starmap for multiple args
            for result in results:
                if result:
                    outfile.write(result)

if __name__ == "__main__":
    project_root = Path(r"C:\Users\dylan\Desktop\sheeplz-crypto-bot")  # Your project root
    output_file = project_root / "combined_code.txt"

    # Directories and files to ignore
    ignore_dirs = [".venv", "venv", "__pycache__", "visualizations", "outputs", "cache", "logs"]
    ignore_files = [".env", "isympy.py", ".conda", ".log", ".txt", "coderator.py", ".gitignore", "readme.md"]

    try:
        combine_code(project_root, output_file, ignore_dirs, ignore_files)
        print(f"\nCombined code saved to: {output_file}")
    except Exception as e:
        print(f"\nAn error occurred: {e}")