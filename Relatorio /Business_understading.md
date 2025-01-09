# Kaggle 

Mild forgetfulness is often a normal part of ageing. But for some people, memory and thinking issues can become more serious as they get older. Cognitive impairment and dementia are increasingly frequent worldwide, impacting the quality of life of millions of patients and their families. These conditions can be caused by various factors such as genetics, lifestyle, and health conditions.

Mild Cognitive Impairment (MCI) is a condition characterized by a cognitive decline that is greater than what would be expected for an individual's age and education level but does not significantly interfere with daily activities. This condition is distinct from dementia, in which cognitive deficits are more severe and widespread and significantly impact daily functioning. However, MCI with memory complaints and deficits has a high risk of progressing to dementia, particularly Alzheimer's disease (AD).

This disease is a neurodegenerative disorder characterized by a gradual decline in chronic primary memory and cognitive impairment. The increasing incidence, high rate of disability, and high cost of treatment have made AD one of the most serious diseases affecting humanity. MCI is a transitional state between normal ageing and AD. People with MCI have a higher risk of developing AD. Therefore, in addition to researching this disease, it is also important to predict its onset in time for potential treatment to reduce the number of people with dementia in the long term.

Many studies have been carried out to develop classification models for MCI, AD and diagnostic purposes. More recently, research has focused on predicting the progression of MCI to AD, which is what we aim to do with this assignment.


# Enunciado 

Mild forgetfulness is often a normal part of ageing. But for some people, memory and thinking issues can become more 
serious  as  they  get  older.  Cognitive  impairment  and  dementia  are  increasingly  frequent  worldwide,  impacting  the 
quality  of  life  of  millions  of  patients  and  their  families.  These  conditions  can  be  caused  by  various  factors  such  as 
genetics, lifestyle, and health conditions. Mild Cognitive Impairment (MCI) is a condition characterized by a cognitive 
decline that is greater than what would be expected for an individual's age and education level but does not significantly 
interfere with daily activities. This condition is distinct from dementia, in which cognitive deficits are more severe and 
widespread and significantly impact daily functioning. However, MCI with memory complaints and deficits has a high 
risk  of  progressing  to  dementia,  particularly  Alzheimer's  disease  (AD).  This  disease  is  a  neurodegenerative  disorder 
characterized by a gradual decline in chronic primary memory and cognitive impairment. The increasing incidence, high 
rate of disability, and high cost of treatment have made AD one of the most serious diseases affecting humanity. MCI is 
a transitional state between normal ageing and AD. People with MCI have a higher risk of developing AD. Therefore, in 
addition to researching this disease, it is also important to predict its onset in time for potential treat ment to reduce 
the number of people with dementia in the long term. 
Through the use of Magnetic Resonance Imaging (MRI) of the brain, we can observe differences that are linked to MCI 
such  as  shrinkage  of  the  hippocampus,  an  essential  region  of  the  brain  responsible  for  memory,  enlargement  of  the 
fluid-filled spaces (ventricles) in the brain, and reduced use of glucose.  
Radiomics is a quantitative approach to medical imaging that provides textural information through the mathematical 
extraction of the spatial distribution of signal intensities and pixel interrelationships. After radiomics feature extraction, 
Machine Learning (ML) or advanced statistical methods are used to analyse the features.
Many  studies  have  been  carried  out  to  develop  classification  models  for  MCI,  AD  and  diagnostic  purposes.  More 
recently,  research  has  focused  on  predicting  the  progression  of  MCI  to  AD,  which  is  what  we  aim  to  do  with  this 
assignment. Fig. 1 illustrates that MCI symptoms can remain stable for years, progress to Alzheimer's disease or another 
type of dementia, or improve over time. 
For this assignment, two datasets  were created (Table 2), each representing the extraction of radiomic features from 
distinct brain regions: the hippocampus and the occipital lobe. The hippocampus dataset (DShippo) was selected due to 
its significant relevance in Alzheimer's research, while the occipital lobe dataset (DSocc) serves as a control, as this region 
is not typically associated with dementia. The hypothesis is that radiomic features in the postulated region of interest 
will demonstrate differences  that will consequently support delineating patients with MCI that will  evolve to AD and 
those who won’t. 





## Submits Inicial


Inicialmente realizamos a visualização e o pre processamento do dataset do hippocampus.


Como o dataset é muito extenso decidimos que a melhor forma do analisamos era através de CSV's.


Sendo que o primeiro csv que nós criamos foi o de Analise da correlação das Colunas com o Target.


Percebemos assim que o data set estava dividido por grupos:




### Grupos


| Coluna 1        | Coluna 2        | Coluna 3       |
|------------------|-----------------|----------------|
| original         | wavelet-LLH    | wavelet-HHL    |
| wavelet-LHL      | wavelet-LHH    | wavelet-HHH    |
| wavelet-HLL      | wavelet-HLH    | wavelet-LLL    |
| log-sigma-1-0    | log-sigma-2-0  | log-sigma-3-0  |
| log-sigma-4-0    | log-sigma-5-0  | square_        |
| square_          | squareroot     | logarithm      |
| exponential      | gradient       | lbp-2D         |
| lbp-3D-k         | lbp-3D-m1      | lbp-3D-m2      |




E cada SubGrupo está dividido pela aplicação da biblioteca python PyRadiomics.


| Category                                 | Description                            |
|-----------------------------------------|----------------------------------------|
| First Order Statistics                  | Extracts statistical features.         |
| Shape-based (3D)                        | Captures 3D geometric characteristics.|
| Shape-based (2D)                        | Captures 2D geometric characteristics.|
| Gray Level Co-occurrence Matrix         | Measures spatial gray-level relations.|
| Gray Level Run Length Matrix            | Analyzes runs of similar gray levels. |
| Gray Level Size Zone Matrix             | Examines zones of similar gray levels.|
| Neighbouring Gray Tone Difference Matrix| Evaluates differences between neighbors.|
| Gray Level Dependence Matrix            | Captures gray-level dependencies.     |


