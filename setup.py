import setuptools

setuptools.setup(
    name="exodusii",
    version="1.0.0",
    description="Python wrappers to the ExodusII finite element database model",
    packages=["exodusii"],
    package_dir={"exodusii": "exodusii"},
    scripts=["bin/exoread.py"],
)
