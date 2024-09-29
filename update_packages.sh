#!/bin/bash

# Ensure pyenv is initialized
eval "$(pyenv init -)"

# Activate the correct Python version (replace 3.x.x with your specific version)
pyenv shell 3.x.x

# Update pip itself
pip install --upgrade pip

# Update all packages
pip install --upgrade -r requirements.txt

# Optional: Generate a new requirements.txt with updated versions
pip freeze > requirements_updated.txt

echo "All packages have been updated. A new 'requirements_updated.txt' file has been created with the latest versions."