## Prediction of therapy outcomes in an internet-based intervention: towards precision therapy?

![Bild_traurig](https://user-images.githubusercontent.com/50407361/104200051-7bf4d500-5428-11eb-82b0-f059bed2c02f.jpg)
*Photo by Nik Shuliahin on Unsplash*

Machine learning (ML) is considered a promising approach to overcome long-standing challenges in clinical psychological research and bring us closer towards personalized mental health care. While ML-based outcome prediction in medicine already demonstrates clinical utility, applications in psychotherapy research yield mostly modest accuracies, possibly reflecting the high variance induced by dynamic personal (e.g. relationship to therapist, diagnosis), procedural (e.g. therapy content) and contextual conditions. Beyond that, sample sizes are often far from big-data dimensions. Internet-based interventions allow for a higher standardization and control of those factors, while being scalable to larger numbers of subjects. 

In this project, we aim to determine if the application of ML models to pre- or early-treatment characteristics of patients undergoing an online CBT program for depression enables clinically beneficial outcome prediction. If so, we want to extract a set of easily accessible, cost-efficient prognostic indices allowing informed decisions on treatment change (e.g. to face-to-face therapy) or adaptation (e.g. increasing dosage of sessions/ feedback). 

The dataset consists of baseline clinical and sociodemographic as well as treatment-related (e.g. short-term responses) variables of 1278 mildly to moderately depressed individuals attending a 6-week guided cognitive behavioral Online-Intervention in Germany (TK Depressionscoach).

## How to use the code
Install all the packages defined in requirements.txt

Use config.py to setup your analysis (e.g. name of outcome variable, handling of missing values). Important: you have to change the PATH variable! 

The whole pipeline is started via main.py. You may add extra preprocessing skripts in line 57.  

Contact me if you have troubles running the code. 
