import numpy as np
import argparse
import string 
import json
import sys
import csv
import os

# Provided wordlists.
FIRST_PERSON_PRONOUNS = {
    'i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours'}
SECOND_PERSON_PRONOUNS = {
    'you', 'your', 'yours', 'u', 'ur', 'urs'}
THIRD_PERSON_PRONOUNS = {
    'he', 'him', 'his', 'she', 'her', 'hers', 'it', 'its', 'they', 'them',
    'their', 'theirs'}
SLANG = {
    'smh', 'fwb', 'lmfao', 'lmao', 'lms', 'tbh', 'rofl', 'wtf', 'bff',
    'wyd', 'lylc', 'brb', 'atm', 'imao', 'sml', 'btw', 'bw', 'imho', 'fyi',
    'ppl', 'sob', 'ttyl', 'imo', 'ltr', 'thx', 'kk', 'omg', 'omfg', 'ttys',
    'afn', 'bbs', 'cya', 'ez', 'f2f', 'gtr', 'ic', 'jk', 'k', 'ly', 'ya',
    'nm', 'np', 'plz', 'ru', 'so', 'tc', 'tmi', 'ym', 'ur', 'u', 'sol', 'fml'}

# os.chdir('./Wordlists')
os.chdir('/u/cs401/Wordlists')

bgl_words = []
bgl_AoA = []
bgl_IMG = []
bgl_FAM = []

with open('BristolNorms+GilhoolyLogie.csv', 'r') as File:
    BGLfile = csv.reader(File)
    for row in BGLfile:
        bgl_words.append(row[1]) 
        bgl_AoA.append(row[3])
        bgl_IMG.append(row[4])
        bgl_FAM.append(row[5])

w_words = []
w_V = []
w_A = []
w_D = []

with open('Ratings_Warriner_et_al.csv', 'r') as File:
    Warrinerfile = csv.reader(File)
    for row in Warrinerfile:
        w_words.append(row[1]) 
        w_V.append(row[2])
        w_A.append(row[5])
        w_D.append(row[8])


#os.chdir('./feats')
os.chdir('/u/cs401/A1/feats')

AltDict = {}
Alt = np.load('Alt_feats.dat.npy', 'r')
with open('Alt_IDs.txt', 'r') as File:
    ID = File.read().split()
for i, num in enumerate(ID):
    AltDict[num] = Alt[i,:] 

CenterDict = {}
Center = np.load('Center_feats.dat.npy', 'r')
with open('Center_IDs.txt', 'r') as File:
    ID = File.read().split()
for i, num in enumerate(ID):
    CenterDict[num] = Center[i,:] 

LeftDict = {}
Left = np.load('Left_feats.dat.npy', 'r')
with open('Left_IDs.txt', 'r') as File:
    ID = File.read().split()
for i, num in enumerate(ID):
    LeftDict[num] = Left[i,:] 

RightDict = {}
Right = np.load('Right_feats.dat.npy', 'r')
with open('Right_IDs.txt', 'r') as File:
    ID = File.read().split()
for i, num in enumerate(ID):
    RightDict[num] = Right[i,:] 


def extract1(comment):
    ''' This function extracts features from a single comment

    Parameters:
        comment : string, the body of a comment (after preprocessing)

    Returns:
        feats : numpy Array, a 173-length vector of floating point features 
        (only the first 29 are expected to be filled, here)
    '''    
    feats = np.zeros(173)
    comment_list = comment.split()
    tokens = []
    tags = []
    
    for word in comment_list: 
        if word != "\n":
            wordlist = word.rsplit('/', 1)
            tokens.append(wordlist[0])
            tags.append(wordlist[1])
   
    total_token_length = 0
    num_tokens = 0
    AoA = []; IMG = []; FAM = []
    V = []; A = []; D = []
    for i, word in enumerate(tokens):
        # checks if word is a punctuation
        punc = True
        for j in range (len(word)):
            if ((word[j] in string.punctuation) == False):
                punc = False
                break

        # 1. words in uppercase 
        if (word == word.upper()) and (len(word) >= 3) and punc == False:
            feats[0] += 1
        
        word = word.lower()
        
        # 2. first person pronouns 
        if (word in FIRST_PERSON_PRONOUNS):
            feats[1] += 1
        
        # 3. second person pronouns 
        if (word in SECOND_PERSON_PRONOUNS):
            feats[2] += 1
        
        # 4. third person pronouns 
        if (word in THIRD_PERSON_PRONOUNS):
            feats[3] += 1
        
        # 5. coord conjunctions 
        if (tags[i] == 'CC'):
            feats[4] += 1
        
        # 6. past tense 
        if (tags[i] == 'VBD'):
            feats[5] += 1

        # 7. future tense
        if (word in ("'ll", "will", "gonna") or (i < len(tokens)-2 and \
            word == 'go' and tokens[i+1].lower() == 'to' and tags[i+2] == 'VB')):
            feats[6] += 1
        
        # 8. commas 
        if (tags[i] == ','):
            feats[7] += 1
        
        # 9. multi-char punc tokens 
        if punc and (len(word) > 1):
            feats[8] += 1
        
        # 10. common nouns
        if (tags[i] in ("NN", "NNS")):
            feats[9] += 1
        
        # 11. proper nouds
        if (tags[i] in ("NNP", "NNPS")):
            feats[10] += 1
        
        # 12. adverbs
        if (tags[i] in ("RB", "RBR", "RBS")):
            feats[11] += 1
        
        # 13. wh- words
        if (tags[i] in ("WDT", "WP", "WP$", "WRB")):
            feats[12] += 1
        
        # 14. slag
        if (word in SLANG):
            feats[13] += 1

        # count total number of characters in this comment 
        if punc == False: 
            total_token_length += len(word)
            num_tokens += 1
            if word in bgl_words:
                index = bgl_words.index(word)
                AoA.append(bgl_AoA[index])
                IMG.append(bgl_IMG[index])
                FAM.append(bgl_FAM[index])
            if word in w_words:
                index = w_words.index(word)
                V.append(w_V[index])
                A.append(w_A[index])
                D.append(w_D[index])

    # 17. number of sentences 
    if comment.count('\n') == 0: #if sentence has no period 
        feats[16] = 1
    else:
        feats[16] = comment.count('\n') 
    # 15. average length of sentences, in tokens 
    feats[14] = len(tokens)/feats[16]
    # 16. average length of tokens, excluding punc-only tokens, in char
    if num_tokens != 0:
        feats[15] = total_token_length/num_tokens
    else:
        feats[15] = 0

    AoA = [float(i) for i in AoA]
    IMG = [float(i) for i in IMG]
    FAM = [float(i) for i in FAM]
    if AoA:
        # 18. Average of AoA
        feats[17] = np.mean(AoA)
        # 19. Average of IMG
        feats[18] = np.mean(IMG)
        # 20. Average of FAM
        feats[19] = np.mean(FAM)
        # 21. SD of AoA
        feats[20] = np.std(AoA)
        # 22. SD of IMG
        feats[21] = np.std(IMG)
        # 23. SD of FAM
        feats[22] = np.std(FAM)
        
    V = [float(i) for i in V]
    A = [float(i) for i in A]
    D = [float(i) for i in D]
    if V:
        # 24. Average of V.Mean.Sum
        feats[23] = np.mean(V)
        # 25. Average of A.Mean.Sum
        feats[24] = np.mean(A)
        # 26. Average of D.Mean.Sum
        feats[25] = np.mean(D)
        # 27. SD of V.Mean.Sum
        feats[26] = np.std(V)
        # 28. SD of A.Mean.Sum
        feats[27] = np.std(A)
        # 29. SD of D.Mean.Sum
        feats[28] = np.std(D)


    return feats
    # TODO: Extract features that rely on capitalization.
    # TODO: Lowercase the text in comment. Be careful not to lowercase the tags. (e.g. "Dog/NN" -> "dog/NN").
    # TODO: Extract features that do not rely on capitalization.
 
    
def extract2(feats, comment_class, comment_id):
    ''' This function adds features 30-173 for a single comment.

    Parameters:
        feats: np.array of length 173
        comment_class: str in {"Alt", "Center", "Left", "Right"}
        comment_id: int indicating the id of a comment

    Returns:
        feats : numpy Array, a 173-length vector of floating point features (this 
        function adds feature 30-173). This should be a modified version of 
        the parameter feats.
    '''    
    
    if comment_class == "Alt":
        feats[29:173] = AltDict[comment_id] 
    if comment_class == "Center":
        feats[29:173] = CenterDict[comment_id] 
    if comment_class == "Left":
        feats[29:173] = LeftDict[comment_id] 
    if comment_class == "Right":
        feats[29:173] = RightDict[comment_id] 
       
    return feats
    
def main(args):
    # switch to the directory that contains preproc.json
    os.chdir("/h/u11/c0/00/cheny163/Desktop/A1")

    data = json.load(open(args.input))
    feats = np.zeros((len(data), 173+1))
    i = 0
    for datum in data:
        # Use extract1 to find the first 29 features for each 
        # data point. Add these to feats.
        features1 = extract1(datum['body'])
        feats[i, 0:28] = features1 [0:28]
        # Use extract2 to copy LIWC features (features 30-173)
        # into feats. (Note that these rely on each data point's class,
        # which is why we can't add them in extract1).
        features2 = extract2(features1, datum['cat'], datum['id'])
        feats[i, 29:173] = features2[29:173]
        if (datum['cat'] == "Left"):
            feats[i, 173] = 0
        if (datum['cat'] == "Center"):
            feats[i, 173] = 1
        if (datum['cat'] == "Right"):
            feats[i, 173] = 2
        if (datum['cat'] == "Alt"):
            feats[i, 173] = 3
        i+=1

    np.savez_compressed(args.output, feats)

    
if __name__ == "__main__": 
    
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("-i", "--input", help="The input JSON file, preprocessed as in Task 1", required=True)
    parser.add_argument("-p", "--a1_dir", help="Path to csc401 A1 directory. By default it is set to the cdf directory for the assignment.", default="/u/cs401/A1/")
    args = parser.parse_args()        
   
    main(args)

