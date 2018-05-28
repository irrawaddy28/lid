##  HMM based Language ID Detector 
This recipe splits a long utterance, containing sentences spoken in multiple languages, into multiple smaller segments.   
Each split segment is expected to have speech pertaining to only a specific language.   

Thus for e.g.,   
if long utterance = [\<music\> \<amharic\> \<english\> \<amharic\> \<english\>],  
then running the HMM based language ID detector on the long utterance will generate,  

short utt 1 = [\<music\>],  
short utt 2 = [\<amharic\>],  
short utt 3 = [\<english\>],  
short utt 4 = [\<amharic\>],  
short utt 5 = [\<english\>],  

The long utterances are sourced from SBS corpus. See local/make_sbs.sh  
The SBS corpus is located at "/ws/rz-cl-2/hasegawa/amitdas/corpus"
