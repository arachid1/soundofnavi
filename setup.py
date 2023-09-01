import setuptools

version = "0.1"

setupkwargs = dict(
    name="soundofnavi",
    version=version,
    description="audio data modelling and ai package",
    url="https://github.com/arachid1/soundofnavi",
    author="Ali Rachidi",
    author_email="ali.rachidi73@gmail.com",
    # license="MIT",
    # packages=['openmsimodel'],
    packages=setuptools.find_packages(include=["soundofnavi*"]),
    # packages=find_packages(),
    zip_safe=False,
    entry_points={"console_scripts": []},
    python_requires=">=3.8,<3.9",
    install_requires=["numpy", "pandas", "matplotlib", "tensorflow", 
                        "scikit-learn", "librosa"],
    extras_require={},
)

setupkwargs["extras_require"]["all"] = sum(setupkwargs["extras_require"].values(), [])

setuptools.setup(**setupkwargs)
