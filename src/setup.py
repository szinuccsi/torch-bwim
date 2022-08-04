import setuptools


setuptools.setup(
    name="torch_bwim",
    version="0.0.2",
    author="Bence Szinyéri",
    author_email="szinyeribence@edu.bme.hu",
    description="Neural Network Framework",
    long_description="A Python implementation framework to work with neural networks, implemented in PyTorch.",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "numpy",
        "torch",
        "matplotlib",
        "scipy",
        "git+https://github.com/szinuccsi/persistance_helper.git#subdirectory=src"
    ],
    python_requires='>=3.8',
)