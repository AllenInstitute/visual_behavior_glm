from setuptools import setup

setup(name='visual-behavior-glm-strategy',
      packages=['visual_behavior_glm_strategy'],
      version='0.0.1',
      description='GLM for Visual Behavior ophys data',
      url='https://github.com/AllenInstitute/visual_behavior_glm_strategy',
      author='Alex Piet, Nick Ponvert, Doug Ollerenshaw, Marina Garrett',
      author_email="alex.piet@alleninstitute.org, nick.ponvert@alleninstitute.org, dougo@alleninstitute.org, marinag@alleninstitute.org",
      license='Allen Institute',
      dependency_links=['https://github.com/AllenInstitute/visual_behavior_analysis.git'],
      install_requires=[
        "allensdk",
        "h5py",
        "matplotlib",
        "plotly",
        "pandas==0.25.3",
        "seaborn",
        "numpy",
        "pymongo",
        "xarray==0.15.1",
        "xarray_mongodb",
      ],
     )
