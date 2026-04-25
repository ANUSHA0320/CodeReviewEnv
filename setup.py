"""
setup.py
========
Install CodeReviewEnv as a local package so imports work from any directory.

    pip install -e .

This also registers the gymnasium environment ID  ``CodeReviewEnv-v0``
automatically when the package is imported.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read long description from README
long_description = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")

setup(
    name="code-review-env",
    version="1.0.0",
    description="OpenAI Gym / Gymnasium environment for AI pull-request code review",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="project contributors",
    python_requires=">=3.9",
    packages=find_packages(exclude=["tests*", "baseline*", "app*"]),
    include_package_data=True,
    package_data={
        "code_review_env": ["../data/*.json"],
    },
    install_requires=[
        "gymnasium>=0.29.0",
        "numpy>=1.23.0",
        "pydantic>=2.0.0",
        "scikit-learn>=1.2.0",
        "PyYAML>=6.0",
    ],
    extras_require={
        "api": [
            "fastapi>=0.100.0",
            "uvicorn[standard]>=0.21.0",
            "httpx>=0.24.0",
        ],
        "ui": [
            "gradio>=4.0.0",
        ],
        "llm": [
            "openai>=1.0.0",
        ],
        "dev": [
            "pytest>=7.0.0",
        ],
        "all": [
            "fastapi>=0.100.0",
            "uvicorn[standard]>=0.21.0",
            "httpx>=0.24.0",
            "gradio>=4.0.0",
            "openai>=1.0.0",
            "pytest>=7.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "code-review-agent=baseline.run_agent:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords=[
        "reinforcement-learning",
        "gymnasium",
        "openai-gym",
        "code-review",
        "pull-request",
        "ai",
        "llm",
    ],
)
