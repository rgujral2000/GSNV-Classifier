#!/usr/bin/python
# file: transformingVCFFile.py

# This application is used to extact the likely denovo, homozygous alt, and other
# SVs present in child like the ones, which indicate the possiblity of loss of
# heterozygosity etc.

import re, sys
##import numpy as np

#############################################################################
## Author: Rohan Gujral
## Project: Machine Learning Germline variant classifier
## This code converts GATK trio VCF file generated using Haplotype caller
## to a format that can be read by 'Germline variant classifier'.
## The trio vcf input for this code assume that data is in father, mother,
## and child order. Else one will have to convert data in that format
## for subsequent classifier code to work.
#############################################################################

TAB = '\t'
POUND = "#"
NEWLINE = '\n'
DASH = '-'

countNumOfRecs = 0
countDenovo = 0
baseCol = 8

FILESET = open("Path to HAPLOTYPE caller trio SNPs only recalibrated VCF file", "r")


def isHomoRef(gtCall):
    """
    :rtype: boolean
    """
    if(gtCall == "0/0"):
        return True
    else:
        return False

def isHet(gtCall):
    """
    :rtype: boolean
    """
    if(gtCall == "0/1"):
        return True
    else:
        return False


def getValType(perRec, pos):
    vals = re.split(':', perRec.strip()) 
    return vals[pos]

def getMaxVal(valueOne, valueTwo):
    if(valueOne == valueTwo):
        return valueOne
    elif (valueOne > valueTwo):
        return valueOne
    else:
        return valueTwo

def getMinVal(valueOne, valueTwo):
    if(valueOne == valueTwo):
        return valueOne
    elif (valueOne > valueTwo):
        return valueTwo
    else:
        return valueOne

def getDifferentElements(colEight):
    vals = re.split(';', colEight.strip())
    allElements = []
    for i in range(0, len(vals), 1):
        if("=" in vals[i] and "culprit" not in vals[i]):
            parts = re.split('=', vals[i].strip())
            allElements.append(parts[1].strip())
	    ##print(parts[0] + " " + parts[1])

    ##elements = list(np.float_(allElements))## This is a simple way to convert a string list to float list
    elements = []
    ##for index in range(0, len(allElements)):
    ##for item in allElements:
        ##elements.append(round(float(item), 3))
    elements.append(str(float(allElements[15])))
    elements.append(allElements[4])
    elements.append(allElements[3])
    elements.append(allElements[6])
    elements.append(allElements[14])
    elements.append(allElements[9])
    elements.append(allElements[11])
    elements.append(allElements[12])
    elements.append(allElements[13])
    ##allElements = []
    ##print("List: ", elements)
    return "	".join(elements)

def getAlleleRatio(alleleVals):
    
    alleles = re.split(',', alleleVals.strip()) 

    return  round(float(alleles[1]) / float(int(alleles[0]) + 1), 3)


def getMiddlePhredScore(phredScore):
    vals = re.split(',', phredScore.strip())
    return int(vals[1])


def getFirstPhredScore(phredScore):
    vals = re.split(',', phredScore.strip())
    return int(vals[0])

counter = 0

for lineWithFNames in FILESET:
    if not lineWithFNames.startswith(POUND) and not lineWithFNames.startswith('X') and not lineWithFNames.startswith('Y') and not lineWithFNames.startswith('GL') and not lineWithFNames.startswith('MT') and "ReadPosRankSum" in lineWithFNames :
        ##print(lineWithFNames)
    	linePieces = re.split('\t+', lineWithFNames.strip())	

        chrm =  linePieces[0] + DASH + linePieces[1]

      

    	dadCol = linePieces[9]
    	##momCol = int(linePieces[10])
    	momCol = linePieces[10]
    	childCol = linePieces[11]
        dadVals = re.split(':', dadCol.strip())
        dadGen = isHomoRef(getValType(dadCol, 0))
        momGen = isHomoRef(getValType(momCol, 0))
        childGen = isHet(getValType(childCol, 0))

        if (dadGen and momGen and childGen and '.' not in dadCol and '.' not in momCol and '.' not in childCol):

            flag = linePieces[6]
            if(flag == 'PASS'):
                flag = 1
            elif(flag == 'VQSRTrancheSNP99.90to100.00'):
                flag = -1
            else:
                flag = 0

            overallQScore = float(linePieces[5])

            dadAlleles = getValType(dadCol, 1)
            momAlleles = getValType(momCol, 1)
            childAlleles = getValType(childCol, 1)

            
            dadAlleleRatio = getAlleleRatio(dadAlleles)
            momAlleleRatio = getAlleleRatio(momAlleles)
            childAlleleRatio = getAlleleRatio(childAlleles)

            minParentAlleleRatio = getMinVal(dadAlleleRatio, momAlleleRatio)
            maxParentAlleleRatio = getMaxVal(dadAlleleRatio, momAlleleRatio)

            dadMiddlePScore = getMiddlePhredScore(getValType(dadCol, 4))
            momMiddlePScore = getMiddlePhredScore(getValType(momCol, 4))
            childMiddlePScore = getMiddlePhredScore(getValType(childCol, 4))

            minMiddleScore = getMinVal(dadMiddlePScore, momMiddlePScore)
            maxMiddleScore = getMaxVal(dadMiddlePScore, momMiddlePScore)

            dadFirstPScore = getFirstPhredScore(getValType(dadCol, 4))
            momFirstPScore = getFirstPhredScore(getValType(momCol, 4))
            childFirstPScore = getFirstPhredScore(getValType(childCol, 4))

            minFirstScore = getMinVal(dadFirstPScore, momFirstPScore)
            maxFirstScore = getMaxVal(dadFirstPScore, momFirstPScore)

            dadGQScore = int(getValType(dadCol, 3))
            momGQScore = int( getValType(momCol, 3))
            childGQScore = int(getValType(childCol, 3))

            minGQScore = getMinVal(dadGQScore, momGQScore) 
            maxGQScore = getMaxVal(dadGQScore, momGQScore) 

            ##print("GQScore: " + str(dadGQScore) + TAB + str(momGQScore) + TAB + str(childGQScore))

            ##print("Middle P-Score: " + str(maxMiddleScore) + TAB + str(minMiddleScore) + TAB + str(childMiddlePScore) + TAB + str(maxFirstScore) + TAB + str(minFirstScore) + TAB + str(childFirstPScore) )
            
            ##print("" + str(dadAlleleRatio) + TAB + str(momAlleleRatio) + TAB + str(childAlleleRatio))
            ##print("" + str(maxParentAlleleRatio) + TAB + str(minParentAlleleRatio) + TAB + str(childAlleleRatio))
    	    ##print(linePieces[0] + TAB + dadCol + TAB + momCol + TAB + childCol + TAB + getGenoType(dadCol)+ TAB + getGenoType(momCol) + TAB + getGenoType(childCol) )
            variousElements = getDifferentElements(linePieces[7])
            
            print("" + str(maxParentAlleleRatio) + TAB + str(minParentAlleleRatio) + TAB + str(childAlleleRatio) +TAB + str(maxMiddleScore) + TAB + str(minMiddleScore) + TAB + str(childMiddlePScore) + TAB + str(maxFirstScore) + TAB             + str(minFirstScore) + TAB + str(childFirstPScore) + TAB + str(flag) + TAB + str(overallQScore) + TAB +  variousElements + TAB + str(minGQScore) + TAB + str(maxGQScore) + TAB + str(childGQScore) + TAB + chrm)

            counter = counter + 1
            ##print("counter: ", counter)
"""
            if(counter < 1):
                print("counter: " + str(counter))
                print("Flag: ", flag)
                print("QScore: ", overallQScore)
            else :
                exit() 
"""        
