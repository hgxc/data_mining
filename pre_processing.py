from nltk.corpus import stopwords
import nltk
from nltk.text import Text
from  nltk import word_tokenize,pos_tag
from nltk import FreqDist
import os
import re
from nltk.stem import WordNetLemmatizer
if __name__ == '__main__':
    file_path='./Dataset/'
    save_path='./SData/'
    file_class=['Training-800/Crude','Training-800/Grain','Training-800/Interest','Training-800/Trade','Testing-40']
    for i,x in enumerate(file_class):
        file_name = os.listdir(file_path+str(x))
        file_name.sort(key=lambda x:int(x[:-4]))
        for file in file_name:
            file_ = open(file_path+str(x)+'/'+file)
            text = file_.read()
            result = re.sub(r'\d+','',text)
            # print(result)
            tokens = word_tokenize(result)
            token_words = pos_tag(tokens)

            words_lematizer = []
            wordnet_lematizer = WordNetLemmatizer()
            for word, tag in token_words:
                if tag.startswith('NN'):
                    word_lematizer =  wordnet_lematizer.lemmatize(word, pos='n')  # n代表名词
                elif tag.startswith('VB'):
                    word_lematizer =  wordnet_lematizer.lemmatize(word, pos='v')   # v代表动词
                elif tag.startswith('JJ'):
                    word_lematizer =  wordnet_lematizer.lemmatize(word, pos='a')   # a代表形容词
                elif tag.startswith('R'):
                    word_lematizer =  wordnet_lematizer.lemmatize(word, pos='r')   # r代表代词
                else:
                    word_lematizer =  wordnet_lematizer.lemmatize(word)
                words_lematizer.append(word_lematizer)
            cleaned_words = [word for word in words_lematizer if word not in stopwords.words('english')]
            characters = [',', '.','DBSCAN', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$',"'", '%','--','-','...','^','{','}',"''","``"]
            words_list = [word for word in cleaned_words if word not in characters]
            words_lists = [x.lower() for x in words_list ]

            freq = FreqDist(words_lists)
            save_file=save_path+str(x)+'/'+file
            print(save_file)
            with open(save_file,'w') as sfile:
                for key,val in freq.items():
                    sfile.write(str(key) + ' ' + str(val)+'\n')
            print('laala')