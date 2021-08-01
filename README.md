# GSNV-Classifier
Germline Single Nucleotide Variant classifiers:
    
We all have about 40-100 new single nucleotide variants in our genome referred as germline variants that are not present in the genomes of our parents. Germline variants have been linked to psychiatric diseases such as Autism, Schizophrenia, & ADHD etc. when they disrupt the function of an important gene expressed in the brain. Two separate classifiers to extract the germline variants from the genomes of dad, mom, and proband (affected kid) are presented here.  First classifier is based on machine learning (ML), several different ML libraries were tested from ‘scikit-learn’ tool kit, and I found that Random Forest library yielded best results when evaluated against a ‘Gold Standard’ call set. Second independent classifier was also developed employing Neural Networks from Google’s AI library ‘TensorFlow’. 


An example of Germline variant from a VCF file is shown here.
     chr3:71,021,785	  GG(Father)	 GG(Mother)	  AG(Child)

‘cleaned_published_data_matrix_training_set’ is a training data set for both ML and TensorFlow based classifiers.

A small validation data set ‘smallValidationData’ is provided here to make predictions.

transformingVCFFile.py transforms Haplotype caller output to a file format that will serve as validation data set can be read by both the classifiers.

forestDNMClassifierModPred.py is ML based classifier and for it to work training data set and validation data sets need to be provided. User will need to update two fields in the code ‘dataframe’ and ‘pred_dataframe’.

AI based classifier ‘tensorflowDNMFinal.py’ also needs training data set and validation data sets. Two fields ‘TT_Data’ and ‘predData’ are file paths for training and validation data respectively and user will need to update the fields in the code.
![image](https://user-images.githubusercontent.com/88251461/127777510-f718de62-7bff-4065-8195-a9266ba69cd5.png)
Germline Single Nucleotide Variant Classifiers:
    
We all have about 40-100 new single nucleotide variants in our genome referred as germline variants that are not present in the genomes of our parents. Germline variants have been linked to psychiatric diseases such as Autism, Schizophrenia, & ADHD etc. when they disrupt the function of an important gene expressed in the brain. Two separate classifiers to extract the germline variants from the genomes of dad, mom, and proband (affected kid) are presented here.  First classifier is based on machine learning (ML), several different ML libraries were tested from ‘scikit-learn’ tool kit, and I found that Random Forest library yielded best results when evaluated against a ‘Gold Standard’ call set. Second independent classifier was also developed employing Neural Networks from Google’s AI library ‘TensorFlow’. 


An example of Germline variant from a VCF file is shown here.
     chr3:71,021,785	  GG(Father)	 GG(Mother)	  AG(Child)

‘cleaned_published_data_matrix_training_set’ is a training data set for both ML and TensorFlow based classifiers.

A small validation data set ‘smallValidationData’ is provided here to make predictions.

transformingVCFFile.py transforms Haplotype caller output to a file format that will serve as validation data set can be read by both the classifiers.

forestDNMClassifierModPred.py is ML based classifier and for it to work training data set and validation data sets need to be provided. User will need to update two fields in the code ‘dataframe’ and ‘pred_dataframe’.

AI based classifier ‘tensorflowDNMFinal.py’ also needs training data set and validation data sets. Two fields ‘TT_Data’ and ‘predData’ are file paths for training and validation data respectively and user will need to update the fields in the code.
![image](https://user-images.githubusercontent.com/88251461/127777511-96240e51-d4af-4947-9d7d-a0ec75a134a7.png)
