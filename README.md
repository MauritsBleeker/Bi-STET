# Bi-STET

This is the project repository for Bidirectional Scene Text Recognition with a Single Decoder, by Maurits Bleeker and Maarten de Rijke

The base source code of this project comes from: http://nlp.seas.harvard.edu/2018/04/03/attention.html

I tried to keep te code as general as possible. But some elements of the pipeline are specially written for the environment I worked with. 

## Model weights

To reproduce the results of the paper, please use the final model parameters. 

https://drive.google.com/file/d/1OwJ3iVpRhnjIZyOi7aOQIeLv7N1DHZkC/view?usp=sharing

## Reproduce the results

I added the processed valiation datasets and the model paramaters to reproduce the results. 

I worked with a config class. The idea was to store the class after training en reload this during validation. However, tit tunred out to be 

# Run
 
To run the code, just run ```main.py```, and set all the condifurations in the Config.py. The configutations to reproduce the results are in the ```Config.py```.
 
# Training

There are two options to load the training/test data

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

The annotations files are formatted as 'path/to/image.jpg annotation'. The path to image is always relative to a root folder.

Example root folder: User/Documents/Project/data/IIITK/

In User/Documents/Project/data/IIITK/, we have an annotation.txt and the images.

An example of annotation the file: 
```
test/1002_1.png private

```

# Data processing

All the files to process the the original provided datasets are given in /data_utils.

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
