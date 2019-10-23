
from os import listdir
from os.path import join

from setuptools import setup
try: # for pip >= 10
    from pip._internal.req import parse_requirements
except ImportError: # for pip <= 9.0.3
    from pip.req import parse_requirements


PACKAGE_NAME = "aidictive"


# Creates a __version__ variable.
with open(PACKAGE_NAME + "/_version.py") as file:
    exec(file.read())

# Read requirements.
req = parse_requirements("requirements.txt", session="hack")
REQUIREMENTS = [str(ir.req) for ir in req]
print("Requirements:", REQUIREMENTS)

setup(name=PACKAGE_NAME,
      version=__version__,
      description="Addictive AI Library.",
      long_description=open("README.md").read(),
      long_description_content_type="text/markdown",
      keywords="ml framework",
      author="guiferviz",
      license="Copyright guiferviz",
      packages=[PACKAGE_NAME],
      install_requires=REQUIREMENTS,
      entry_points={
          "console_scripts": [
              "{} = {}.__main__:main".format(PACKAGE_NAME, PACKAGE_NAME)
          ]
      },
      zip_safe=False)
