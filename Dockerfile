FROM jupyter/scipy-notebook

COPY environment.yml /tmp/
RUN conda env update -n base -f /tmp/environment.yml

COPY environment-r.yml /tmp/
RUN conda env update -n base -f /tmp/environment-r.yml

COPY env-dev.yml /tmp/
RUN conda env update -n base -f /tmp/env-dev.yml


RUN jupyter labextension install @axlair/jupyterlab_vim @jupyterlab/toc @ryantam626/jupyterlab_code_formatter
# RUN pip install jupyterlab_code_formatter
# jupyter labextension install jupyterlab-jupytext@0.19
RUN jupyter serverextension enable --py jupyterlab_code_formatter
RUN jupyter lab build --name='ros'
RUN jupyter nbextension install --py jupytext --user
RUN jupyter nbextension enable jupytext --user --py
RUN jupyter serverextension enable jupytext

# Script to add "collapsibleNotebooks": false
RUN mkdir -p "/home/jovyan/.jupyter/lab/user-settings/@jupyterlab/toc/"
RUN echo '{"collapsibleNotebooks": true}' > /home/jovyan/.jupyter/lab/user-settings/\@jupyterlab/toc/plugin.jupyterlab-settings


# Add this stuff to files
# RUN pip install dscontrib==20200728134544
WORKDIR /home/jovyan/ros/


# Run commands
# docker run -p 8888:8888 jupyter/scipy-notebook -v ~/Dropbox/ws/data/ros:/ros
# run pip install git+git://github.com/wcbeard/dscontrib@9127e99
