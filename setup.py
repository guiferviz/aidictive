
from setuptools import setup, find_packages
from pip._internal.req import parse_requirements


PACKAGE_NAME = "aidictive"
DESCRIPTION = "An addictive AI library."
KEYWORDS = "python ai ml framework dl machine deep learning deep pytorch"
AUTHOR = "guiferviz"
AUTHOR_EMAIL = "guiferviz@gmail.com"
LICENSE = "Copyright " + AUTHOR
URL = "https://github.com/guiferviz/aidictive"


print("Project:", PACKAGE_NAME)

# Creates a __version__ variable.
with open(PACKAGE_NAME + "/_version.py") as file:
    exec(file.read())
print("Version:", __version__)

# Read requirements.
req = parse_requirements("requirements.txt", session="hack")
for i in req:
    print(dir(i))
    print(i.requirement)
REQUIREMENTS = [str(ir.req) for ir in req]
print("Requirements:", REQUIREMENTS)

# Install all packages in the current dir except tests.
PACKAGES = find_packages(exclude=["tests", "tests.*"])
print("Packages:", PACKAGES)

setup(name=PACKAGE_NAME,
      version=__version__,
      description=DESCRIPTION,
      long_description=open("README.md").read(),
      long_description_content_type="text/markdown",
      url=URL,
      keywords=KEYWORDS,
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      license=LICENSE,
      packages=PACKAGES,
      install_requires=REQUIREMENTS,
      entry_points={
          "console_scripts": [
              "{} = {}.__main__:main".format(PACKAGE_NAME, PACKAGE_NAME)
          ]
      },
      zip_safe=False)

