import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()
setuptools.setup(
     name='PermutationImportancePhysics',
     version='0.111',
     packages=['permutationimportancephysics',],
     author='A Ghosh',
     description="Permutation Importance for Physics",
     long_description=long_description,
   long_description_content_type="text/markdown",
     url='https://github.com/aghoshpub/permutationImportancePhysics',
 )
