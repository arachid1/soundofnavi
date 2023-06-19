# setup.py
from setuptools import setup, find_packages


# REQUIRED_PACKAGES = ["keras",
#                      "h5py<3.0.0",
#                      "tensorflow-gpu",
#                      "tensorflow",
#                      "tensorflow-estimator",
#                      "google-cloud-storage",
#                      "cloudstorage",
#                      "google",
#                      "tensorboard",
#                      "gast==0.3.3",
#                      "numba==0.43.0",
#                      "llvmlite==0.32.1",
#                      "librosa==0.7.2",
#                      "matplotlib",
#                      "requests",
#                      "nlpaug"]

REQUIRED_PACKAGES = ["keras==2.4.3",
                     "h5py<3.0.0",
                     "tensorflow-gpu",
                     "tensorflow==2.3.1",
                     "tensorflow-estimator==2.3.0",
                     "google-cloud-storage==1.29.0",
                     "cloudstorage",
                     "google",
                     "tensorboard",
                     "gast==0.3.3",
                     "numba==0.43.0",
                     "llvmlite==0.32.1",
                     "librosa==0.7.2",
                     "matplotlib",
                     "nlpaug"]

setup(
    name="mel",
    version="0.1",
    packages=find_packages(),
    description="classification of abnormal lung sounds",
    author="Ali Rachidi",
    author_email="rachidiali10@gmail.com",
    license="MIT",
    install_requires=[
        REQUIRED_PACKAGES
    ],
    zip_safe=False,
)
