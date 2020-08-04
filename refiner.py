import io, re, string
from dateutil import parser
import datetime
try:
    import nltk
except:
    import os
    print("nltk not found \ntrying to install nltk...")
    os.system("pip install nltk")

"""Abbreviation	        Meaning
    CC	            coordinating conjunction
    CD	            cardinal digit
    DT	            determiner
    EX	            existential there
    FW	            foreign word
    IN	            preposition/subordinating conjunction
    JJ	            adjective (large)
    JJR	            adjective, comparative (larger)
    JJS	            adjective, superlative (largest)
    LS	            list market
    MD	            modal (could, will)
    NN	            noun, singular (cat, tree)
    NNS	            noun plural (desks)
    NNP	            proper noun, singular (sarah)
    NNPS	    proper noun, plural (indians or americans)
    PDT	            predeterminer (all, both, half)
    POS	            possessive ending (parent\ 's)
    PRP	            personal pronoun (hers, herself, him,himself)
    PRP$	    possessive pronoun (her, his, mine, my, our )
    RB	            adverb (occasionally, swiftly)
    RBR	            adverb, comparative (greater)
    RBS	            adverb, superlative (biggest)
    RP	            particle (about)
    TO	            infinite marker (to)
    UH	            interjection (goodbye)
    VB	            verb (ask)
    VBG	            verb gerund (judging)
    VBD	            verb past tense (pleaded)
    VBN	            verb past participle (reunified)
    VBP	            verb, present tense not 3rd person singular(wrap)
    VBZ	            verb, present tense with 3rd person singular (bases)
    WDT	            wh-determiner (that, what)
    WP	            wh- pronoun (who)
    WRB	            wh- adverb (how)"""
                           

#date = r'\d+\S\d+\S\d+'
def punctSep(text):
    def customSplit(text, sep):
        sp = text.split(sep)
        return [sp[0], f"".join(sp[1::])]
    pfound = ""
    for t in text:
        if t in punct2:
            pfound = t
            return (True, customSplit(text, pfound))
    return (False, None)

def rmvDup(data):
    rmv = []
    for d in data:
        if d not in rmv:
            rmv.append(d)
    return rmv

def detect(text, multiM = False):
    ### multiM = allow multiple matches
    if not multiM:

        if punctSep(text)[0]:
            return punctSep(text)[1]
        
        if True:
            try:
                date = parser.parse(text, fuzzy=True)
                return ('date', f"{date.day}/{date.month}/{date.year}")
            except Exception:
                pass
            
        if re.search(r'[\w\.-]+@[\w\.-]+', text):
            email = re.search(r'[\w\.-]+@[\w\.-]+', text)
            return ("email", email[0])

        if re.search(r'\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4}', text):
            number = re.search(r'\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4}', text)
            return ("phone number", number[0])

        if True:
            found = ''
            greater = [0,0] #len , index
            for i, c in enumerate(text.split(" ")):
                if len(c) > greater[0]:
                    greater = [len(c), i]
            found = text.split(" ")[greater[1]]
            
            return (found, text)

        
    else:
        multimatch = []

        if True:
            try:
                date = parser.parse(text, fuzzy=True)
                multimatch.append(('date', f"{date.day}/{date.month}/{date.year}"))
            except Exception:
                pass
        if re.search(r'[\w\.-]+@[\w\.-]+', text):
            email = re.search(r'[\w\.-]+@[\w\.-]+', text)
            multimatch.append(("email", email[0]))
            
        if re.search(r'\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4}', text):
            number = re.search(r'\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4}', text)
            multimatch.append(("phone number", number[0]))
        return multimatch

def endswith_punct(text) -> bool:
    for p in punct:
        if text.endswith(p):
            return True
    return False

def smartCompile(data) -> list:
    refine = ['']

    try:
        pre = data[0]
        for d in data[1::]:
            d = d.strip()
            if d!="":
                if endswith_punct(pre):
                    t = pre+" "+d
                    refine.append(t)
                    pre = t
                elif (pre[-1].isdigit() and (d[0].isdigit() or d[0].isalpha()) and not endswith_punct(d))or(pre[-1].isalpha() and d[0].isdigit() and not endswith_punct(d)):
                    
                    t = pre+" "+d
                    refine.append(t)
                    pre = t
                else:
                    refine.append(pre)
                    pre = d
                '''else:
                    refine.append(d)'''
    except Exception as ex:
        print(ex)
        pass
    return refine
        

raw_data = ["Rajasthan Technical University\nRawatbhata Road, Kota - 324010 (Ph:0744-2473014,2473931)\nENROLLMENT CARD\nENROLLMENT NO 17\nIA CS M\n3.\nP 042\
        \nName :\nMANISH CHOUDHARY\nFather's Name: RAMAPRAKASH CHOUDHARY\nDate of Birth\n6/3/2002\nCourse :\nВ. ТЕСН\nBERLIM\nBranch :\nComputer Science \
        and Engineering\nCollege :\nInstitute of Engineering & Technology, Alwar\nINICAL\nUni\n2006\n(Satya Pal Yadav)\nAuthorised Signatory\nSignature of\
        Student\n"]

punct = [x for x in string.punctuation]
punct2 = [":", ">", "<", "=", "-", "/", "\\"]

### main function ###
def structMyData(raw_data):
    raw_data = [x.split("\n") for x in raw_data]

    tokens = []

    table = str.maketrans('', '', string.punctuation)

    for n, i in enumerate(raw_data):
        for j in range(len(i)):
            tokens.append(nltk.tokenize.word_tokenize(raw_data[n][j]))

    sentences = [" ".join(x) for x in tokens]

    refine = rmvDup(smartCompile(sentences))
    data = {}
    for r in refine:
        d = detect(r)
        try:
            data[d[0]] = d[1]
        except:
            pass
    return data

if __name__ == "__main__":
    print(structMyData(raw_data))

