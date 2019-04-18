'''
Author:Albert
Date:  2019-4-18
describe: api起始页
'''
__author__ = 'Albert'
# from datetime import date
import tornado.escape
import tornado.ioloop
import tornado.web
# import pandas as pd
# import loyalLevel as loyal
# import loyalLevel_v2 as loyal_v2
# import coursePredict as course
# import associateCourse as asso
# import userSimilarity as user
# import associateRules as rules
# import userSimilarity_v2 as user_v2
# import userSimilarity_v3 as user_v3
from tornado.options import define, options
# import coursePredict_v2 as course2
# import dataProcess.multiLabelEncoder
import main
import json

define("port", default=9000, help="run on the given port", type=int)


def returnResult(data, msg=None, status=1):
    '''
    返回结果
    
    Args:
         data:如果status=0,则data为空
         msg:返回信息，如果status=0，则返回具体错误信息
         status：0：处理失败  1：处理成功
       
    Returns:
         str:返回json串   
    '''
    dic = {}
    dic["data"] = data
    dic["msg"] = msg
    dic["status"] = status

    return json.dumps(dic, ensure_ascii=False)


class ProcessHandler(tornado.web.RequestHandler):
    '''
    process api
    '''

    def post(self, version):
        '''
        post
        
        Returns:
              json:
        '''
        try:
            param = self.request.body.decode('utf-8')
            v_num = int(version[1:])
            if v_num == 1:
                r = main.predict(param)
            # elif v_num == 2:
            #     loyalmodel = loyal_v2.loyal_v2()
            else:
                self.finish(returnResult(None, "无对应版本，请核实版本号", 0))

            self.finish(returnResult(r))
        except BaseException as err:
            self.finish(returnResult(None, "系统异常:" + str(err.args[0]), 0))


application = tornado.web.Application([(r"/NER/(?P<version>v\d+)",
                                        ProcessHandler)])

# tornado.options.parse_command_line()

if __name__ == "__main__":
    # tornado.options.parse_command_line()
    application.listen(options.port)
    tornado.ioloop.IOLoop.instance().start()