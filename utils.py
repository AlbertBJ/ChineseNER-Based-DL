import tensorflow as tf
import data_util as data
from Model import Model
import re
import numpy as np
import json
import os


def pre_process(dict_config):

    sPath = dict_config['sPath']
    tPath = dict_config['tPath']
    splitPath = dict_config['splitPath']
    usePreTrained = dict_config['usePreTrained']
    spreVecPath = dict_config['sPreVector']
    sentenceMaxLen = dict_config['sentenceMaxLen']
    splitSenPattern = dict_config['splitSenPattern']
    dicPath = dict_config['dicPath']

    data.tagWords(sPath=sPath, tPath=tPath)
    dic_list, df = data.splitSentences(
        sPath=tPath,
        splitPath=splitPath,
        splitSenPattern=splitSenPattern,
        dicPath=dicPath)
    id2word, id2tag, word2id, tag2id = dic_list
    x_train, x_test, y_train, y_test, sequence_train, sequence_test = data.processData(
        df, tag2id, word2id, sentenceMaxLen)
    if usePreTrained:
        dict_config['embedding'] = data.usingPreEmbedding(
            id2word, spreVecPath=spreVecPath)

    dict_config['voc_size'] = len(id2word)
    dict_config['tag_size'] = len(id2tag)

    saveJson(dict_config)

    return x_train, x_test, y_train, y_test, sequence_train, sequence_test, id2word, id2tag, word2id, tag2id


def createModel(session, dict_config):
    # x_train, x_test, y_train, y_test, sequence_train, sequence_test, id2word, id2tag, word2id, tag2id = pre_process(
    #     dict_config)

    model = Model(dict_config)
    model.buildNN()

    if not dict_config['is_train']:  # restore latest model
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(dict_config['sModelFile'])

        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            saver.restore(session, ckpt.model_checkpoint_path)

    return model


def calculateResult(x, y, senquence_lengths, id2tag, id2word):
    '''
    calculate result including precision,recall and F1
    Args:
        x:input data, shape is [batch_size,senMaxLen]
        y:tag labels,shape is [batch_size,senMaxLen]        
        senquence_lengths: senquence_lengths is a vector of true length
        id2tag: dict,
        id2word: dict
    '''
    result = []
    entity = []
    for i in range(x.shape[0]):  #for every sentence
        for j in range(senquence_lengths[i]):  #for every valid word
            if id2tag[y[i][j]][0] == 'B':
                entity = [id2word[x[i][j]] + '/' + id2tag[y[i][j]]]
            elif id2tag[y[i][j]][0] == 'I' and len(entity) != 0 and entity[
                    -1].split('/')[1][1:] == id2tag[y[i][j]][1:]:
                entity.append(id2word[x[i][j]] + '/' + id2tag[y[i][j]])
            elif id2tag[y[i][j]][0] == 'E' and len(entity) != 0 and entity[
                    -1].split('/')[1][1:] == id2tag[y[i][j]][1:]:
                entity.append(id2word[x[i][j]] + '/' + id2tag[y[i][j]])
                entity.append(str(i))
                entity.append(str(j))
                result.append(entity)
                entity = []
            else:
                entity = []
    return result


def calculatePRF(x, y, pre, senquence_lengths, id2tag, id2word):
    '''
    calculate precision, recall and F1 
    Args:
        x:input data, shape is [batch_size,senMaxLen]
        y:tag labels,shape is [batch_size,senMaxLen]  
        pre: predict result, shape is [batch_size,senMaxLen] 
        senquence_lengths: senquence_lengths is a vector of true length
        id2tag: dict,
        id2word；dict
    '''
    FN_TP_list = calculateResult(x, y, senquence_lengths, id2tag, id2word)
    FP_TP_list = calculateResult(x, pre, senquence_lengths, id2tag, id2word)
    TP_list = [i for i in FP_TP_list if i in FN_TP_list]
    FN_TP = len(FN_TP_list)
    FP_TP = len(FP_TP_list)
    TP = len(TP_list)
    p = precision(TP, FP_TP)
    r = recall(TP, FN_TP)
    return p, r, F1(p, r)


def precision(tp, fp_tp):
    '''
    calculate  precision
    '''
    return 0 if fp_tp == 0 else tp / fp_tp
    # return tp / fp_tp


def recall(tp, fn_tp):
    '''
    calculate  recall
    '''
    return 0 if fn_tp == 0 else tp / fn_tp
    # return tp / fn_tp


def F1(p, r):
    '''
    calculate  F1
    Args:
        p: precision
        r:recall
    '''
    return 0 if p + r == 0 else 2 * p * r / (p + r)
    # return 2 * p * r / (p + r)


def process_input(model, session, inputText, word2id, id2tag, dict_config):
    '''
    process input text
        
    '''
    input_text, input_wordIds, sequence_lengths = data.process_input(
        inputText, word2id, dict_config)

    dic = {
        model.inputdata: input_wordIds,
        model.sequence_length: sequence_lengths
    }
    pre = session.run([model.decode_tags], dic)
    return getEntities(pre[0], input_text, id2tag, sequence_lengths)


def getEntities(result_decode, input_text, id2tag, senquence_lengths):
    '''
    get entity list

    Return:
        entity list, shape[[tagType,entityName],...,[tagType,entityName]]
    '''

    entities = []
    for i in range(len(input_text)):  #for every sen
        name = []
        entity = []
        for j in range(senquence_lengths[i]):  #for every word
            tag = id2tag[result_decode[i][j]]  # get tag
            if tag[0] == 'B':
                tagType = tag[2:]
                name.append(input_text[i][j])
                entity.append(tagType)
            elif tag[0] == 'I' and len(entity) != 0:
                name.append(input_text[i][j])
            elif tag[0] == 'E' and len(entity) != 0:
                name.append(input_text[i][j])
                entity.append(''.join(name))
                entities.append(entity)
                entity = []
                name = []
            else:
                entity = []  # for next step
                name = []
    return entities


def saveJson(dict_config):
    with open(os.path.join(os.path.dirname(__file__), 'config/conf.json'), 'w') as json_file:
        json_file.write(json.dumps(dict_config))


def loadJson():
    jsonfile=os.path.join(os.path.dirname(__file__), 'config/conf.json')
    if not os.path.exists(jsonfile):
        return None
    with open() as json_file:
        data = json.load(json_file)
    return data