Challenge Overview
Welcome to the Leaf Counting Challenge

This is the Jülich Challenges version of the Leaf Counting Challenge (LCC) from CVPPP2017, the third workshop on Computer Vision Problems in Plant Phenotyping held in conjunction with ICCV2017. The Leaf Segmentation Challenge (LSC) based on the same data is already online available at CodaLab under challenge number 18405. We set up this LCC version to meet the communities interest in this challenge, as e.g. visible in the still high download numbers of the dataset (see Fig.1). For further information please refer to our dataset page.
https://gitlab.version.fz-juelich.de/MLDL_FZJ/MLDL_FZJ_Wiki/-/wikis/Juelich%20Challenges%20Hackathon


About the data

The provided data has been collected in our laboratories (datasets A1 -- A3) or derived from a public dataset (A4, public data kindly shared by Dr Hannah Dee from Aberystwyth) of top-view images of rosette plants. All images were hand labelled. We share images of tobacco plants and arabidopsis plants via (https://www.plant-phenotyping.org/datasets). Tobacco images were collected using a camera which contained in its field of view a single plant. Arabidopsis images were collected using a camera with a larger field of view encompassing many plants, which were cropped. The images released are either from mutants or wild types and have been taken in a span of several days. Plant images are encoded as tiff files.

All images were hand labelled to obtain ground truth masks for each leaf in the scene. These masks are image files encoded in PNG where each segmented leaf is identified with a unique integer value, starting from 1, where 0 is background. For the counting problem, annotations are provided in the form of a png image where each leaf center is denoted by a single pixel. Additionally a CSV file with image name and number of leaves is provided.

For further information on the ground truth annotation process, please refer to:

M. Minervini, A. Fischbach, H.Scharr, and S.A. Tsaftaris. Finely-grained annotated datasets for image-based plant phenotyping. Pattern Recognition Letters, pages 1-10, 2015, doi:10.1016/j.patrec.2015.10.013 [PDF] [BibTex]
Hanno Scharr, Massimo Minervini, Andreas Fischbach, Sotirios A. Tsaftaris. Annotated Image Datasets of Rosette Plants. Technical Report No. FZJ-2014-03837, Forschungszentrum Jülich, 2014
Bell, Jonathan, & Dee, Hannah M. (2016). Aberystwyth Leaf Evaluation Dataset [Data set]. Zenodo. http://doi.org/10.5281/zenodo.168158
or the 2017 challenge documents on LSC 2017 or LCC 2017.

https://data-challenges.fz-juelich.de/web/challenges/challenge-page/85/overview
