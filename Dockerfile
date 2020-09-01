FROM jupyter/scipy-notebook

COPY environment.yml /tmp/
RUN conda env update -n base -f /tmp/environment.yml

COPY environment-r.yml /tmp/
RUN conda env update -n base -f /tmp/environment-r.yml

RUN jupyter labextension install @axlair/jupyterlab_vim @jupyterlab/toc @ryantam626/jupyterlab_code_formatter
# RUN pip install jupyterlab_code_formatter
RUN jupyter serverextension enable --py jupyterlab_code_formatter
RUN jupyter lab build --name='ros'

COPY environment-r.yml /tmp/
RUN conda env update -n base -f /tmp/environment-r.yml

# Script to add "collapsibleNotebooks": false
RUN mkdir -p "/home/jovyan/.jupyter/lab/user-settings/@jupyterlab/toc/"
RUN echo '{"collapsibleNotebooks": true}' > /home/jovyan/.jupyter/lab/user-settings/\@jupyterlab/toc/plugin.jupyterlab-settings

# Add this stuff to files
# RUN pip install dscontrib==20200728134544
WORKDIR /home/jovyan/ros/


# Run commands
# docker run -p 8888:8888 jupyter/scipy-notebook -v ~/Dropbox/ws/data/ros:/ros
