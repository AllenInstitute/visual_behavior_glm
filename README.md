# visual_behavior_glm
Fits a kernel regression model to neural traces during visual behavior. 

# Installing and setting up the package

## Set up an environment

Before installing, it's recommended to set up a new Python environment with Python 3.7:

For example, using Conda:

    conda create -n visual_behavior_glm python=3.7.9

Then activate the environment:

    conda activate visual_behavior_glm

## Installation

To facilitate development, it is recommended to set up the package in 'editable' mode:

    git clone https://github.com/AllenInstitute/visual_behavior_glm.git
    cd visual_behavior_glm
    pip install -e .

An additional dependency of the package is `visual_behavior_analysis` (VBA)  
Assuming that most users of this package will also be contributing to VBA, it should also be installed in 'editable' mode:

    git clone https://github.com/AllenInstitute/visual_behavior_analysis.git
    cd visual_behavior_analysis
    pip install -e .

Alternatively, the current master branch could be installed with:

    pip install git+https://github.com/AllenInstitute/visual_behavior_analysis.git

Test that the package was installed properly by importing the GLM class from outside of the visual_behavior_glm directory:

    cd ~
    python
    >>> from visual_behavior_glm.glm import GLM

Please report issues at https://github.com/AllenInstitute/visual_behavior_glm/issues
