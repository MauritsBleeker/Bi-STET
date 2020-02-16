# Bi-STET

This is the repository for 'Bidirectional Scene Text Recognition with a Single Decoder', by Maurits Bleeker and Maarten de Rijke [[pdf](https://arxiv.org/pdf/1912.03656.pdf)]

The base source-code for this project comes from: http://nlp.seas.harvard.edu/2018/04/03/attention.html

I have tried to keep te code as general as possible. However, some elements of the pipeline are specially for the environment I worked with. 

## Model weights and reproducibility

To reproduce the results of the paper, please use the final model parameters. 

https://drive.google.com/file/d/1OwJ3iVpRhnjIZyOi7aOQIeLv7N1DHZkC/view?usp=sharing

In the folder data_utils/, all the scripts to generate the train and test sets as used for this paper are provided.

# Python and package versions

* Python 3.7 
* Pillow	5.4.1
* nltk	3.4.5	
* numpy	1.17.1	
* scipy	1.2.0	
* seaborn	0.9.0	
* tensorboard-logger	0.1.0	
* tensorboardX	1.7	
* torch	1.1.0.post2	
* torchvision	0.2.1	
* transformers	2.1.1	

# Run
 
To run the code, just run ```main.py```, and set all the condifurations in the Config.py. The configutations to reproduce the results are set in the ```Config.py``` file.
 
# Training

There are two options to load the training/test data:

- From disk. This can be done by using the annotation file(s).
- From a pickle file. The pickle file should contain a python dict with the following data format.

```
{
image_id : { 
    'data' : 'binary image string',
    'label' : 'word'
    }
}

```

## Test and train annotations

The annotations text files are formatted as 'path/to/image.jpg annotation'. The path to image is always relative to a root folder.

Example root folder: User/Documents/Project/data/IIITK/

In User/Documents/Project/data/IIITK/, we have an annotation.txt and the images.

An example of annotation the file: 
```
test/1002_1.png private

```

# Data processing

All the files to process the original provided train datasets are given in /data_utils.


# Reference 
If you found this code useful, please cite the following paper:
```
@article{bleeker2019bidirectional,
  title={Bidirectional Scene Text Recognition with a Single Decoder},
  author={Bleeker, Maurits and de Rijke, Maarten},
  journal={arXiv preprint arXiv:1912.03656},
  year={2019}
}
```
