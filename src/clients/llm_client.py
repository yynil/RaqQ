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
    
    def generate(self,ctx,token_count=100):
        cmd = {"cmd": "GENERATE_TEXTS", "ctx": ctx,"token_count": token_count}
        self.socket.send(msgpack.packb(cmd, use_bin_type=True))
        msg = self.socket.recv()
        resp = msgpack.unpackb(msg, raw=False)
        return resp
    
if __name__ == '__main__':
    llm_client = LLMClient("tcp://localhost:7781")
    print(llm_client.encode(["大家好","我来自北京"]))
    print(llm_client.cross_encode(["大家好","我是一个机器人"],["你好","小猫咪真可爱"]))
    print(llm_client.generate("User: 我有一个数学表达式，请为我算出结果。\nBot: 好的。请出题。\nUser:(25-8)*12\nBot:",token_count=100))