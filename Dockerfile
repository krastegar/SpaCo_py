FROM rocker/rstudio:4.4

# Set environment variables 
# non interactive mode is for dealing with arguements that require response 
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies for RStudio Server
RUN apt-get update && apt-get install -y \
    psmisc \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libhdf5-dev \
    libfftw3-dev \
    libgsl-dev \
    libclang-dev \
    libharfbuzz-dev \
    libfreetype6-dev \
    libpng-dev \
    libtiff5-dev \
    libjpeg-dev \
    libfontconfig1-dev \
    libpng-dev \
    libfribidi-dev \
    libglpk40 \
    git \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install R packages

# Install devtools package in R
RUN R -e "install.packages('remotes')"

# Installing Necessary R packages
RUN R -e "library(remotes); remotes::install_version('ggplot2', version='3.5.2')"
RUN R -e "library(remotes); remotes::install_version('Seurat', version='5.2.1')"
RUN R -e "library(remotes); remotes::install_version('Matrix', version='1.7.3')"
RUN R -e "library(remotes); remotes::install_github('IMSBCompBio/SpaCo')"
RUN R -e "library(remotes); remotes::install_github('xzhoulab/SPARK')" 

# Home Directory
WORKDIR /home/rstudio

# load data 
COPY R_scripts/brain_seuratobject_raw.RData /home/rstudio/

# Expose RStudio Server port
EXPOSE 8787

# Set default user and password
ENV USERNAME=rstudio
ENV PASSWORD=seurat

# Keep the container running
CMD ["/init"]