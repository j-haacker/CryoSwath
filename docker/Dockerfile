FROM quay.io/jupyter/minimal-notebook:latest

RUN mkdir $HOME/project_dir
WORKDIR $HOME/project_dir
RUN git clone https://github.com/j-haacker/cryoswath.git
RUN conda create -n main_env python=3.13 pip numpy GDAL distributed
SHELL ["conda", "run", "-n", "main_env", "/bin/bash", "-c"]
RUN pip install --editable "./cryoswath"
RUN cryoswath-init
WORKDIR $HOME/project_dir/scripts
