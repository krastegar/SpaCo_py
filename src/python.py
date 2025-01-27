import pkg_resources

try:
    setuptools_version = pkg_resources.get_distribution("setuptools").version
    print(f"Setuptools version: {setuptools_version}")
except pkg_resources.DistributionNotFound:
    print("Setuptools is not installed.")
