import tensorflow as tf
import math
from Model import Model
from batchManager import BatchManager
from utils import pre_process, createModel, calculatePRF, process_input, loadJson
import os
import pickle
from log_util import Log
import argparse

curr = os.path.dirname(__file__)

parser = argparse.ArgumentParser(description='Process some params.')
parser.add_argument(
    '--sPath',
    default=os.path.join(
        curr, 'dataset/BosonNLP_NER_6C/BosonNLP_NER_6C/BosonNLP_NER_6C.txt'),
    help='original train dataset path')
parser.add_argument(
    '--tPath',
    default=os.path.join(
        curr, 'dataset/BosonNLP_NER_6C/BosonNLP_NER_6C/wordTagged.txt'),
    help='the  path of tagged dataset')

parser.add_argument(
    '--splitSenPattern',
    default='[，。《》“”‘’？！#【】]/O',
    help='regular pattern for spliting sentences')

parser.add_argument(
    '--splitPath',
    default=os.path.join(curr,
                         'dataset/BosonNLP_NER_6C/BosonNLP_NER_6C/split.txt'),
    help='the  path of splited dataset')
parser.add_argument(
    '--usePreTrained',
    default=True,
    help='whether to use pre-trained embedding matrix')

parser.add_argument(
    '--sPreVector',
    default=os.path.join(curr, 'dataset/sgns.renmin.word/sgns.renmin.word'),
    help='the path pre-trained embedding matrix')
parser.add_argument(
    '--dicPath',
    default=os.path.join(curr, 'map/dic.pkl'),
    help='the path saving dictionary')
parser.add_argument(
    '--sentenceMaxLen', default=20, help='the max length per sentence')
parser.add_argument(
    '--sModelFile',
    default=os.path.join(curr, 'model'),
    help='the path saving models')
parser.add_argument(
    '--log', default=os.path.join(curr, 'log'), help='the path saving log')
parser.add_argument('--batch_size', default=256, help='batch size')
parser.add_argument('--lr', default=0.01, help='learning rate')
parser.add_argument('--epoch', default=200, help='epoch')

parser.add_argument(
    '--embedding_dim', default=300, help='embedding matrix length of axis=1')
parser.add_argument(
    '--is_train',
    default=True,
    help='when true ,execute training, otherwise inference')
parser.add_argument('--model_name', default='ner', help='model name')
parser.add_argument(
    '--model_save_step', default=100, help='the frequency saving model')
parser.add_argument('--max_to_keep', default=5, help='max model count')
parser.add_argument(
    '--rate', default=0.5, help='drop out rate,rate=1-drop_out_keep')

params = parser.parse_args()

F = 0


def train(dict_config):
    log = Log(dict_config['log'])
    log.info('execute train')

    x_train, x_test, y_train, y_test, sequence_train, sequence_test, id2word, id2tag, word2id, tag2id = pre_process(
        dict_config)
    log.info(' data processed')
    batchManager = BatchManager(x_train, y_train, sequence_train)
    log.info(' begin train ')
    tf.reset_default_graph()
    with tf.Session() as sess:
        model = createModel(sess, dict_config)
        batch_num = math.ceil(x_train.shape[0] / model.batch_size)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=dict_config['max_to_keep'])
        step = 1
        for epoch in range(dict_config['epoch']):
            for batch in range(batch_num):
                model.rate = dict_config['rate']
                x, y, sequence_length = batchManager.next_batch(
                    model.batch_size)

                dic = {
                    model.inputdata: x,
                    model.tagdata: y,
                    model.sequence_length: sequence_length
                }
                step += 1

                sqe, loss, _ = sess.run(
                    [model.decode_tags, model.loss, model.train_op], dic)

                if step % dict_config['model_save_step'] == 0:
                    log.info(' step : {}, and loss is {}'.format(step, loss))

                    model.rate = 0
                    P_, R_, F1_ = evaluate(sess, model, x_train, y_train,
                                           sequence_train, id2tag, id2word)
                    log.info('evaluate model:')
                    train_s = 'train data P value:{},R value:{},F1 value:{}'.format(
                        P_, R_, F1_)
                    log.info(train_s)

                    P, R, F1 = evaluate(sess, model, x_test, y_test,
                                        sequence_test, id2tag, id2word)
                    test_s = 'Test data P value:{},R value:{},F1 value:{}'.format(
                        P, R, F1)
                    log.info(test_s)

                    saver.save(
                        sess,
                        os.path.join(dict_config['sModelFile'],
                                     dict_config['model_name']),
                        global_step=step)
                    log.info(' model saved. epoch:{}, step:{}'.format(
                        epoch, step))

                    # global F
                    # if F1 > F:
                    #     F = F1
                    #     saver.save(
                    #         sess,
                    #         os.path.join(dict_config['sModelFile'],
                    #                      dict_config['model_name']),
                    #         global_step=step)
                    #     log.info(' model saved. epoch:{}, step:{}'.format(
                    #         epoch, step))


def evaluate(session, model, x_test, y_test, sequence_test, id2tag, id2word):
    dic = {
        model.inputdata: x_test,
        model.tagdata: y_test,
        model.sequence_length: sequence_test
    }

    # return shape  :batch_size*senMaxLen
    decode_tags = session.run([model.decode_tags], dic)
    P, R, F1 = calculatePRF(x_test, y_test, decode_tags[0], sequence_test,
                            id2tag, id2word)
    return P, R, F1


def inference(dict_config):
    log = Log(dict_config['log'])
    log.info("begin to inference")
    with open(dict_config['dicPath'], "rb") as f:
        id2word, id2tag, word2id, tag2id = pickle.load(f)
    with tf.Session() as sess:
        model = createModel(sess, dict_config)
        while True:
            inputText = input("Please enter your input: ")
            entities = process_input(model, sess, inputText, word2id, id2tag,
                                     dict_config)
            logresu = []
            logresu.append(inputText)
            length = len(entities)
            if length <= 0:
                print("Sorry, we cannot find entity from input!")
                continue
            print("we find result as bellow:")
            for i in range(length):
                r = "{}:{}".format(entities[i][0], entities[i][1])
                logresu.append(r)
                print(r)
            log.info("input text:{}, result:{}".format(
                logresu[0], logresu[1:] if length > 0 else ''))


def run():
    # convert to dict
    dic = vars(params)
    dic['embedding'] = None

    if dic['is_train']:
        train(dic)
    else:
        # load config
        saved = loadJson()
        # merging config, using dic to overwrite saved when same key
        dic = {**saved, **dic}
        inference(dic)


def inference_online(dict_config, string):
    tf.reset_default_graph()
    with open(dict_config['dicPath'], "rb") as f:
        id2word, id2tag, word2id, tag2id = pickle.load(f)
    with tf.Session() as sess:
        model = createModel(sess, dict_config)

        # inputText = input("Please enter your input: ")
        entities = process_input(model, sess, string, word2id, id2tag,
                                 dict_config)
        return entities


def predict(string):
    dic = vars(params)
    dic['embedding'] = None
    dic['is_train'] = False
    saved = loadJson()
    # merging config, using dic to overwrite saved when same key
    dic = {**saved, **dic}
    return inference_online(dic, string)


class SingleModel:
    model = None

    def __init__(self):
        raise SyntaxError("can not instance, please use get_model")

    @staticmethod
    def get_model():

        if SingleModel.model is None:
            pass
        # load model
        return SingleModel.model


if __name__ == "__main__":
    run()
