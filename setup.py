from setuptools import setup, find_packages

setup(
    name="quantum-neural-blockchain-system",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.19.0",
        "scipy>=1.7.0",
    ],
    author="Quantum Systems Team",
    author_email="quantum@example.com",
    description="A hybrid quantum-classical system combining quantum computing, neural networks, blockchain, and robotics",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/quantum-neural-blockchain-system",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.8",
)
