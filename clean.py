import subprocess
import os

def clean_project(project_dir):
  """
  Cleans a Python project by removing unused imports and variables using Autoflake.

  Args:
    project_dir: The path to the root directory of the Python project.
  """

  exclude_paths = [
      "venv",
      "migrations",
      ".mypy_cache",
      "__pycache__",
      # Add any other directories or file names to exclude
  ]

  for root, _, files in os.walk(project_dir):
      for file in files:
          if file.endswith(".py") and not any(exclude in root for exclude in exclude_paths):
              file_path = os.path.join(root, file)
              try:
                  subprocess.run(
                      [
                          "autoflake",
                          "--in-place",
                          "--remove-unused-variables",
                          "--remove-all-unused-imports",
                          file_path,
                      ],
                      check=True,
                  )
                  print(f"Cleaned: {file_path}")
              except subprocess.CalledProcessError as e:
                  print(f"Error cleaning {file_path}: {e}")

if __name__ == "__main__":
  project_directory = r"C:\Users\dylan\Desktop\sheeplz-crypto-bot"  # Your project directory
  clean_project(project_directory)