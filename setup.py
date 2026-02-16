from setuptools import setup, find_packages

with open("README.md") as f:
    long_description = f.read()

with open("requirements.txt") as requirements_file:
    requirements = requirements_file.read().splitlines()

setup(
    name="flai-tree-stats-lidar",
    version="0.1",
    python_requires="<=3.13",
    description="Extract tree and forestry statistics from LiDAR point clouds.",
    license="MIT",
    keywords="",
    author="Flai d.o.o.",
    author_email="info@flai.ai",
    url="https://github.com/flai-ai/flai-tree-stats-lidar",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["tests", "examples"]),
    entry_points={
        "console_scripts": [
            "flai_forest=flai_tree_stats_lidar.cli.cli_forest:cli",
        ]
    },
    install_requires=requirements,
)