from setuptools import setup, find_packages

setup(
    name="TIL_bihead",
    version="0.0.2",
    author="Vishwesh Ramanathan",
    author_email="vishweshramanathan@gmail.com",
    packages=find_packages(),
    license="LICENSE.txt",
    install_requires=[
        "tqdm==4.62.3",
        "torch==1.10.2",
        "torchvision==0.11.3",
        "scikit-image==0.19.2",
        "opencv-python==4.5.5.64",
        "albumentations==1.1.0"
    ],
)
