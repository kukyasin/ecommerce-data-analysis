"""Setup configuration for E-Commerce Data Analysis package."""

from setuptools import setup, find_packages

with open("README_GITHUB.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ecommerce-data-analysis",
    version="1.0.0",
    author="Yasin Kuk",
    author_email="your.email@example.com",
    description="Professional business intelligence dashboards with statistical analysis and ML insights",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kukyasin/ecommerce-data-analysis",
    license="MIT",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Office/Business",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.22.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.12.0",
        "scikit-learn>=1.0.0",
        "scipy>=1.8.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "black>=22.0",
            "flake8>=4.0",
            "isort>=5.0",
        ],
    },
    keywords=[
        "data-analysis",
        "business-intelligence",
        "visualization",
        "machine-learning",
        "statistics",
        "pandas",
        "matplotlib",
        "seaborn",
        "e-commerce",
        "analytics",
    ],
    project_urls={
        "Bug Reports": "https://github.com/kukyasin/ecommerce-data-analysis/issues",
        "Source": "https://github.com/kukyasin/ecommerce-data-analysis",
        "Documentation": "https://github.com/kukyasin/ecommerce-data-analysis#readme",
    },
)
