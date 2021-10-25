# GSNV-Classifier
Description
    
We all have about 40-100 new single nucleotide variants in our genomes referred as germline variants that are not present in the genomes of our parents. Psychiatric disorders such as Autism, Schizophrenia, & ADHD etc. have been shown to associated with germline variants when they disrupt the function of an important gene expressed in the brain. Two separate classifiers to extract the germline variants from the genomes of dad, mom, and proband (affected kid) are presented here. Classifiers are written in Python language. First classifier is based on machine learning (ML), several different ML libraries were tested from ‘scikit-learn’ tool kit, and I found that Random Forest library yielded best results when evaluated against a ‘Gold Standard’ call set. Second independent classifier was also developed employing Neural Networks from Google’s AI library ‘TensorFlow’. 


An example of Germline variant from a familial VCF file is shown here.
     
     chr3:71,021,785	  GG(Father)	 GG(Mother)	  AG(Child)

‘cleaned_published_data_matrix_training_set’ file is a training data set employed for both ML and TensorFlow based classifiers. This data set was generously provided by Professor Jonathan Sebat from Department of Psychiatry at University of California at San Diego. Details of this data set and accompanying research are published in this artcle:  Jacob J. Michaelson, Yujian Shi et al. & Jonathan Sebat (2012). Whole-Genome Sequencing in Autism Identifies Hot Spots for De Novo Germline Mutation. Cell, 151, 1431–1442 .

A small validation data set ‘smallValidationData’ is provided here to make predictions.

Both the classfiers need Python 3.6 or higher version.

transformingVCFFile.py transforms VQS recalibrated SNV Haplotype caller output to a file format that will serve as validation data set can be read by both the classifiers.

forestDNMClassifierModPred.py is ML based classifier and for it to work training data set and validation data sets need to be provided. User will need to update two fields in the code ‘dataframe’ and ‘pred_dataframe’.

AI based classifier ‘tensorflowDNMFinal.py’ also needs training data set and validation data sets. Two fields ‘TT_Data’ and ‘predData’ are file paths for training and validation data respectively and user will need to update these two fields in the code.
