### Detecting Copy-Move Forgery Using Mantranet with Morphology Based Segmentation
This Repository contains the code for copy-move forgery Detection in images which was done as a course project for ECE 278A,UCSB. The backbone network was taken from Mantranet (https://github.com/ISICV/ManTraNet). We just use this backbone for the inference pupose. We added morphology based post processing over the output from ManTraNet for our project's purpose.
   
For cloning the repo type:

```
git clone https://github.com/ASMIftekhar/COPY_MOVE.git

```

This will copy the whole repository in your current directory

To download all the precomputed results, you need to run the following command. This will download and put all the precomputed results in your current working directory
 
```
bash setup.sh

```
### Requirements
All dependencies will be installed in a python3 virtual environment.

## Step 1: Create a python virtual environment

```
virtualenv -p python3.6 <path_to_new_virtual_env>
```
## Step 2: Activate the environment


```
source <path_to_new_virtual_env>/bin/activate
```

## Step 3: Install the dependencies

run 
```
pip3 install -r requirements
```
