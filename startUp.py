'''
Author:Albert
Date:  2019-4-18
describe: api起始页
'''
__author__ = 'Albert'
import tornado.escape
import tornado.ioloop
import tornado.web
from tornado.options import define, options
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