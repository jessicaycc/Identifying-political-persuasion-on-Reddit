import sys
import argparse
import os
import json
import re
import spacy
from html import unescape


nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
sentencizer = nlp.create_pipe("sentencizer")
nlp.add_pipe(sentencizer)


def preproc1(comment , steps=range(1, 5)):
    ''' This function pre-processes a single comment

    Parameters:                                                                      
        comment : string, the body of a comment
        steps   : list of ints, each entry in this list corresponds to a preprocessing step  

    Returns:
        modComm : string, the modified comment 
    '''
    modComm = comment
  
    if 1 in steps:  # replace newlines with spaces
        modComm = re.sub(r"\n{1,}", " ", modComm)
    if 2 in steps:  # unescape html
        modComm = unescape(modComm)
    if 3 in steps:  # remove URLs
        modComm = re.sub(r"(http|www)\S+", "", modComm)
    if 4 in steps:  # remove duplicate spaces
        modComm = re.sub(' +', ' ', modComm)

    # TODO: get Spacy document for modComm
    doc = nlp(modComm)
    # TODO: use Spacy document for modComm to create a string.
    newComm = ""
    for sentence in doc.sents:
        # print(sentence.text)
        for token in sentence: 
            if token.tag_ == ".": 
                if token.lemma_ == "-PRON-":
                    newToken = token.text + "/" + token.tag_ + "\n"
                else: 
                    newToken = token.lemma_ + "/" + token.tag_ + "\n"
            else: 
                if token.lemma_ == "-PRON-":
                    newToken = token.text + "/" + token.tag_ + " "
                # account for future tense
                elif (token.lemma_ == "going to"):
                    g = 'going'
                    t = 'to'
                    newToken = "going/VBG to/TO "
                # account for S.C. during testing
                elif (token.text == "S.C." ):
                    newToken = "SouthCarolina/" + token.tag_ + " "
                else:
                    newToken = token.lemma_ + "/" + token.tag_ + " "
            newComm = newComm + newToken
        
    # Make sure to:
    #    * Insert "\n" between sentences.
    #    * Split tokens with spaces.
    #    * Write "/POS" after each token.
    modComm = newComm
    # modComm = re.sub(r"(?<![A-Z][a-z])([!?.])(?=\s*[A-Z])\s*",r"\1\n",modComm)
    # print(modComm)

    return modComm


def main(args):
    allOutput = []
   
    for subdir, dirs, files in os.walk(indir):
        for file in files:
            fullFile = os.path.join(subdir, file)
            
            print( "Processing " + fullFile)

            # TODO: select appropriate args.max lines
            data = json.load(open(fullFile))
            start = int(args.ID[0])%len(data)
            max_lines = int(args.max)

            if start + max_lines > len(data):
                data_lines = data[start:] + data[:start + max_lines - len(data)]
            else:
                data_lines = data[start:start + max_lines]

            result = []
 
            for line in data_lines:
                result.append({'id':json.loads(line)['id'], 'body':preproc1(json.loads(line)['body']),'cat':file})
               
            allOutput.extend(result)
            # TODO: read those lines with something like `j = json.loads(line)`
            # TODO: choose to retain fields from those lines that are relevant to you
            # TODO: add a field to each selected line called 'cat' with the value of 'file' (e.g., 'Alt', 'Right', ...) 
            # TODO: process the body field (j['body']) with preproc1(...) using default for `steps` argument
            # TODO: replace the 'body' field with the processed text
            # TODO: append the result to 'allOutput'
            
    fout = open(args.output, 'w')
    fout.write(json.dumps(allOutput))
    fout.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument('ID', metavar='N', type=int, nargs=1,
                        help='your student ID')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("--max", type=int, help="The maximum number of comments to read from each file", default=10000)
    parser.add_argument("--a1_dir", help="The directory for A1. Should contain subdir data. Defaults to the directory for A1 on cdf.", default='/u/cs401/A1')
    
    args = parser.parse_args()

    if (args.max > 200272):
        print( "Error: If you want to read more than 200,272 comments per file, you have to read them all." )
        sys.exit(1)
    
    indir = os.path.join(args.a1_dir, 'data')
    #indir = './samples_outputs/'
    #indir = './data/'
    main(args)
