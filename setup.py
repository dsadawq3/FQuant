from setuptools import setup, find_packages

setup(
    name="fquant",
    version="1.0.0",
    description="FQuant: High-Precision Post-Training LLM Quantization Framework by F-Labs",
    author="F-Labs",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "safetensors>=0.4.0",
        "numpy>=1.22.0"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
)
