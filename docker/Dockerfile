FROM quay.io/jupyter/minimal-notebook:latest

RUN mkdir $HOME/cryoswath
RUN git clone https://github.com/j-haacker/cryoswath.git $HOME/cryoswath
WORKDIR $HOME/cryoswath
RUN sed -i '/- python .*/d' environment.yml
RUN conda env update -n base -f environment.yml
RUN mkdir -p $HOME/project_dir/data
WORKDIR $HOME/project_dir
