from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="crypto_trading_bot",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A cryptocurrency trading bot using reinforcement learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/crypto_trading_bot",
    packages=find_packages(include=['src', 'src.*']),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy",
        "pandas",
        "torch",
        "gym",
        "ccxt",
        "hydra-core",
        "pytorch-lightning",
        "optuna",
        "mlflow",
        "matplotlib",
        "seaborn",
        "talib",
        "transformers",
        # Add any other dependencies your project needs
    ],
    entry_points={
        "console_scripts": [
            "crypto_bot=src.main:main",
        ],
    },
)