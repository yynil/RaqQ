import msgpack

class LLMClient:
    def __init__(self,url) -> None:
        import zmq
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(url)
        self.socket.setsockopt(zmq.RCVTIMEO, 60000)

    def encode(self,texts):
        cmd = {"cmd": "GET_EMBEDDINGS", "texts": texts}
        self.socket.send(msgpack.packb(cmd, use_bin_type=True))
        msg = self.socket.recv()
        resp = msgpack.unpackb(msg, raw=False)
        return resp
    
    def cross_encode(self,texts_0,texts_1):
        cmd = {"cmd": "GET_CROSS_SCORES", "texts_0": texts_0,"texts_1": texts_1}
        self.socket.send(msgpack.packb(cmd, use_bin_type=True))
        msg = self.socket.recv()
        resp = msgpack.unpackb(msg, raw=False)
        return resp
    
    def generate(self,instruction,input_text,token_count=100):
        cmd = {"cmd": "GENERATE_TEXTS", "instruction": instruction,"input_text":input_text,"token_count": token_count}
        self.socket.send(msgpack.packb(cmd, use_bin_type=True))
        msg = self.socket.recv()
        resp = msgpack.unpackb(msg, raw=False)
        return resp
    def generate_beam(self,instruction,input_text,token_count=100):
        cmd = {"cmd": "BEAM_GENERATE", "instruction": instruction,"input_text":input_text,"token_count": token_count}
        self.socket.send(msgpack.packb(cmd, use_bin_type=True))
        msg = self.socket.recv()
        resp = msgpack.unpackb(msg, raw=False)
        return resp
    
if __name__ == '__main__':
    llm_client = LLMClient("tcp://localhost:7781")
    print(llm_client.encode(["大家好","我来自北京"]))
    print(llm_client.cross_encode(["大家好","我是一个机器人"],["你好","小猫咪真可爱"]))
    instruction ='根据给定的短文，回答问题： 《庆余年》是谁主演的？'
    input_text = '《庆余年》是由腾讯影业、新丽电视、天津深蓝影视、上海阅文影视等出品，王倦、猫腻担任编剧，孙皓执导，张若昀、李沁领衔主演，陈道明、吴刚、李小冉、辛芷蕾、李纯、宋轶联合主演的古装剧。身世神秘的少年——范闲，自小跟随奶奶生活在海边小城澹州，随着一位老师的突然造访，他看似平静的生活开始直面重重的危机与考验。在神秘老师和一位蒙眼守护者的指点下，范闲熟识药性药理，修炼霸道真气并精进武艺，而后接连化解了诸多危局。因对身世之谜的好奇，范闲离开澹州，前赴京都。 在京都，范闲饱尝人间冷暖并坚守对正义、良善的坚持，历经家族、江湖、庙堂的种种考验与锤炼，书写了光彩的人生传奇。。该剧于2019年11月26日在腾讯视频、爱奇艺首播；该剧播出后斩获了第26届上海电视节最佳中国电视剧提名，第九届大学生电视节大学生赏析推荐电视剧以及2020年度优秀海外传播作品。'
    
    print(llm_client.generate(instruction,input_text,token_count=100))
    print(llm_client.generate_beam(instruction,input_text,token_count=100))