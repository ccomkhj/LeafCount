## Challenge Overview
Welcome to the Leaf Counting Challenge

This is the Jülich Challenges version of the Leaf Counting Challenge (LCC) from CVPPP2017, the third workshop on Computer Vision Problems in Plant Phenotyping held in conjunction with ICCV2017. For further information please refer to our dataset page.
https://gitlab.version.fz-juelich.de/MLDL_FZJ/MLDL_FZJ_Wiki/-/wikis/Juelich%20Challenges%20Hackathon


## About the data

The provided data has been collected in our laboratories (datasets A1 -- A3) or derived from a public dataset (A4, public data kindly shared by Dr Hannah Dee from Aberystwyth) of top-view images of rosette plants. All images were hand labelled. We share images of tobacco plants and arabidopsis plants via (https://www.plant-phenotyping.org/datasets). Tobacco images were collected using a camera which contained in its field of view a single plant. Arabidopsis images were collected using a camera with a larger field of view encompassing many plants, which were cropped. The images released are either from mutants or wild types and have been taken in a span of several days. Plant images are encoded as tiff files.

## How I solved

The given label is only the number of leafs, but the size of images and the portion of plants are different. 
Therefore, detecting the plant without plants are the most difficult challenges.
I have come up with the several Computer vision algorithm.

First, I tried to detect the vase of plant by using circular *hough transformation*. However, vase's shape is not only circle. So, it failed.<br>
Second, I used clouding method. *K-means* was applied to detect. My intuition was that most of green color of plants will gather in one mode. I also converted RGB into HSV, HSL , but it was not working.<br>
Third, I tested pre-trained *YOLO* to detect the location of leaf, but it was not effective. There was not publicly pre-trained weight only for the leave only.<br>
Fourth, I utilized *Grab-cut* which is semi-supervised learning. In the begining I used Open-CV to simply implement the algorithm, but it supports only one forground and one background image, but multiple images are necessary to cover the diversity of plant and background. Therefore, I developed the customized grab-cut algorithm which takes 6 foregrounds and backgrounds each.<br>
Lastly, croped images will be feed into deep learning model (here we used Densenet.)


> **Feel free to contact us**
* [Huijo Kim](mailto:huijo.kim@rwth-aachen.de)
* [Praise Thampi](mailto:praise.thampi@rwth-aachen.de)
* [Ankit Patnala](mailto:ankit.patnala@rwth-aachen.de)
