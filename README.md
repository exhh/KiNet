# KiNet
This code is used to locate and classify individual cells or nuclei in histological images. The code is implemented with PyTorch (version 0.4.1, https://pytorch.org/) on a Ubuntu Linux machine. 

### Cell/Nucei Recognition
**Training:** The input is RGB images and corresponding labels (please find below an example training image and its corresponding label). For each training image, the label is K proximity maps (here K=3), each of which corresponds to one type of cell/nucleus. Given the gold standard annotation of each training image, the label for immunopositive tumor nuclei, immunonegative tumor nuclei and non-tumor nuclei is generated using Equation (1) in the paper [1].

<br />
<img src="results/example_training.png" width="1200"><br/>
<br />

**Testing:** The input is RGB images. The output is predicted positions and categories of cells/nuclei, which will be stored in .mat files and also overlaid on the RGB images (as shown in the "Prediction" column in the figure below). You might want to download a pre-trained model from this link, https://www.dropbox.com/s/sl2l5z3d65l983t/ki67net-best.pth?dl=0, and then put the pre-trained model into the directory: learned_models/PNET-ki67net/

<br />
<img src="results/example_result.png" width="1200"><br/> 
<br />

**Usage (Linux command line):** \
Model training: &nbsp; ./train.sh \
Model inference/prediction: &nbsp; ./eval.sh  

<br /><br />
### Joint Cell/Nuclei Recognition and Tumor Region Segmentation      
For cell/nuclei recognition, the label for each training image is K proximity maps (here K=3), each of which corresponds to one type of cell/nucleus. For tumor region segmentation, a binary mask is proided for model training.

**Usage (Linux command line):** \
Model training: &nbsp; ./train_roi.sh \
Model inference/prediction: &nbsp; ./eval_roi.sh 

<br /> <br /> 
*Some codes are based on this following code repository: \
Hoffman et al. CyCADA: Cycle-Consistent Adversarial Domain Adaptation. ICML 2018. https://github.com/jhoffman/cycada_release

<br /> 
Relevant References:<br /> 
[1] Xing et al. Pixel-to-pixel Learning with Weak Supervision for Single-stage Nucleus Recognition in Ki67 Images. IEEE Transactions on Biomedical Engineering, 2019.
