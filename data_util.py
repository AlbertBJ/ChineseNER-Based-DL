__author__ = 'Albert Wang'
import codecs
import os
from typing import Optional
import pandas as pd
import numpy as np
import re

# data sourse: https://bosonnlp.com/dev/resource  NER


def tagWords(sPath: str, tPath: str):
    '''
    word-based tag
    tag BIEO
    sPath: source file path
    tPath: target file path after tagged
    '''

    # del target file if existed
    if os.path.exists(tPath):
        os.remove(tPath)

    with codecs.open(
            sPath, 'r', encoding='utf-8') as r, codecs.open(
                tPath, 'w', encoding='utf-8') as w:
        for line in r:  # process per line
            line = re.sub(' ', '', line.strip())
            i = 0
            while i < len(line.rstrip()):
                if line[i] == '{':  # begin to tag entity
                    i += 2  # double { , valid entity
                    tmpEntity = []
                    while line[i] != '}':  #extract entity untill }
                        tmpEntity.append(line[i])
                        i += 1  # next word
                    ent = ''.join(
                        tmpEntity)  # ent value is product_name:浙江在线杭州
                    word = ent.split(':')
                    w.write('{}/B_{} '.format(word[1][0], word[0]))
                    for k in range(1, len(word[1]) - 1):
                        w.write('{}/I_{} '.format(word[1][k], word[0]))
                    w.write('{}/E_{} '.format(word[1][-1], word[0]))
                    i += 2  # next valid word
                else:
                    w.write('{}/O '.format(line[i]))
                    i += 1
            w.write('\n')


def splitSentences(sPath: str,
                   splitSenPattern: str,
                   splitPath: Optional[str] = None,
                   dicPath: Optional[str] = None):
    '''
    split into sentences from per text(the line of target file)
    
    sPath: file path needing to split
    tPath: optional, file path where all dics to be stored
    '''
    if not os.path.exists(sPath):
        raise Exception(' could not find the file path')
    # allSentences = []  # record all sentence
    words = {'UNK': 0}  # k: word  v:frequency
    tags = {'UNK': 0}  # k: tag  v:frequency
    data = []
    labels = []
    if splitPath:
        w = codecs.open(splitPath, 'w', encoding='utf-8')
    with codecs.open(sPath, 'r', encoding='utf-8') as r:
        for line in r:
            sentences = re.split(
                splitSenPattern,
                line)  # using regular expression to split text into sentences
            for sen in sentences:  # sentences format :word/tag word/tag ...
                linedata = []
                linelabel = []
                sen = sen.strip()
                if sen is None or not sen:
                    continue
                if splitPath:
                    w.write('{}\n'.format(sen))
                for w_t in sen.split():  # w_t formatL word/tag
                    wt = w_t.split('/')
                    # record frequency for words and tags
                    if wt[0] is None or not wt[0] or wt[1] is None or not wt[1]:
                        continue
                    linedata.append(wt[0])
                    linelabel.append(wt[1])
                    if wt[0] in words:
                        words[wt[0]] += 1
                    else:
                        words[wt[0]] = 1
                    if wt[1] in tags:
                        tags[wt[1]] += 1
                    else:
                        tags[wt[1]] = 1
                data.append(linedata)
                labels.append(linelabel)
    if splitPath:
        w.close()
    id2word = id2word_tag(words)
    id2tag = id2word_tag(tags)
    l = [id2word, id2tag, tag_word2id(id2word), tag_word2id(id2tag)]
    if dicPath:
        import pickle
        with open(dicPath, 'wb') as out:
            pickle.dump(l, out)
    return l, pd.DataFrame({'words': data, 'tags': labels})


def id2word_tag(dic: dict) -> dict:
    '''
    format dict{id:word/tag}
    '''
    sorteddict = sorted(dic.items(), key=lambda x: x[1], reverse=True)
    import pandas as pd
    s = pd.Series(list(dict(sorteddict).keys()))
    return s.to_dict()


def tag_word2id(dic: dict) -> dict:
    '''
    format dict{word/tag:id}
    
    dic: {id:word/tag}
    '''
    return {v: k for k, v in dic.items()}


def getValuesBulk(dic: dict, keys: list) -> list:
    '''
    
    '''
    import pandas as pd
    s = pd.Series(dic, index=dic.keys())
    return list(s[keys])


def processData(df: pd.DataFrame, tag2id: dict, word2id: dict,
                sentenceMaxLen: int):
    def x_padding(words):
        wordids = []
        wordids = getValuesBulk(word2id, words)
        if len(wordids) < sentenceMaxLen:
            wordids.extend([0] * (sentenceMaxLen - len(wordids)))

        else:
            wordids = wordids[:sentenceMaxLen]
        return wordids

    def y_padding(tags):
        tagids = []
        tagids = getValuesBulk(tag2id, tags)
        if len(tagids) < sentenceMaxLen:
            tagids.extend([0] * (sentenceMaxLen - len(tagids)))
        else:
            tagids = tagids[:sentenceMaxLen]
        return tagids

    df['x'] = df['words'].apply(x_padding)
    df['y'] = df['tags'].apply(y_padding)

    # this field wii be used in sequence_length in the next steps
    df['wordslen'] = df['words'].apply(lambda x: len(x) if len(x) <
                                       sentenceMaxLen else sentenceMaxLen)

    from sklearn.model_selection import train_test_split
    x_train, x_test = train_test_split(
        df.loc[:, :], test_size=0.3, random_state=40)

    return np.asarray(list(x_train['x'].values)), np.asarray(
        list(x_test['x'].values)), np.asarray(list(
            x_train['y'].values)), np.asarray(
                list(x_test['y'].values)), np.asarray(
                    list(x_train['wordslen'])), np.asarray(
                        list(x_test['wordslen']))


def process_input(inputText,word2id, dict_config):
    '''
    process inputting text

    Return:
          input_processed: all valid input
          input_wordIds: input data for model
          senquence_lengths: the vector of true sequence length
    '''

    sentences = re.split(dict_config['splitSenPattern'][0:-2], inputText)
    input_processed = []
    input_wordIds = []
    senquence_lengths = []

    for sen in sentences:
        sen = re.sub(' ', '', sen.strip())
        if sen is None or not sen:
            continue
        input_processed.append(sen) # remove empty line
        line_ids = []
        for word in sen:
            if word in word2id:
                line_ids.append(word2id[word])
            else:
                line_ids.append(word2id["UNK"])
        senquence_lengths.append(len(line_ids))
        input_wordIds.append(
            padding_line(line_ids, dict_config['sentenceMaxLen']))

    return input_processed, np.asarray(input_wordIds), senquence_lengths


def padding_line(line_ids, senMaxLen):
    '''
    input text padding

    Args:
        line_ids: wordid lists per sentence
        senMaxLen: max length of the config setting
    '''
    if len(line_ids) < senMaxLen:
        line_ids.extend([0] * (senMaxLen - len(line_ids)))
    else:
        line_ids = line_ids[:senMaxLen]
    return line_ids


def usingPreEmbedding(id2word: dict,
                      spreVecPath: str,
                      embedding_dim: int = 300) -> np.array:
    '''
    merge embedding vector using pre-trained vector
    
    word2id: current corpus word2id dict
    spreVecPath: the path of pre-trained embedding vector
    embedding_dim: embedding dim
    
    return: pre-trained vector
    '''
    import numpy as np
    # read pre-trained embedding vector
    pre = {}
    with codecs.open(spreVecPath, 'r', encoding='utf-8') as r:
        for line in r:
            row = line.rstrip().split()
            if len(row) == embedding_dim + 1:  # valid vector
                pre[row[0]] = row[1:]

    # gen current embedding vector
    new_vector = []
    for i in range(len(id2word)):
        word = id2word[i]
        if word in pre:  # find an exists vector
            #             print(type(pre[word]))
            new_vector.append(pre[word])
        else:
            z = np.random.random(size=embedding_dim)
            #             print(type(z))
            new_vector.append(z)
            pass

    return np.asarray(new_vector, dtype=np.float32)


if __name__ == '__main__':
    tagWords(sPath=sPath, tPath=tPath)
    l, df = splitSentences(sPath=tPath, splitPath=splitPath)
    id2word, id2tag, word2id, tag2id = l
    x_train, x_test, y_train, y_test = processData(df, tag2id, word2id)
    print('x_train, x_test, y_train, y_test shape: {} {} {} {}'.format(
        x_train, x_test, y_train, y_test))
