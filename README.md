# SParse-sensing Experimental Demonstration

This repo is designed to be a playground for testing sparse sensing and it's various applicaitons. There will be several examples that are heavily based on exercises designed by Steve Brunton. However, the ultimate goal will be to expand the repo to apply sparse sensing to other fields aswell. 

## Topics Covered

* Recreation

* Compression

* Tailored sensing

* Robust Algorithm generation
 
* Classification

## Envrionment and install
First clone this repo through the following shell command:

```
git clone git@github.com:thomaskost17/SPED.git
```
This will make a local copy of the repository on your system. As this project is python based, we will be managing our envrionment using conda. To replicate the envrionment first install miniconda locally. Afterwards run the following command to create the envrionment:

```
conda env create --file SPED.yml
```

Now that you have created the desired envrionment, you can activate the envrionment by running:

```
conda activate SPED
```

At this point you can execute any of the code in the repository! Note, if you are modifying the code for your own purpose, our envrionment uses python 3.6.13. This is due to a stability limitation of cvxpy.
