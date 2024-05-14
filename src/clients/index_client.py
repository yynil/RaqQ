import msgpack
import uuid
class IndexClient:
    def __init__(self,frontend_url) -> None:
        import zmq
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(frontend_url)
        self.socket.setsockopt(zmq.RCVTIMEO, 60000)

    def index_texts(self,texts,keys=None):
        if keys is None or isinstance(keys, list) == False or len(keys) != len(texts):
            keys = [str(uuid.uuid4()) for i in range(len(texts))]
        cmd = {"cmd": "INDEX_TEXTS", "texts": texts,"keys": keys}
        self.socket.send(msgpack.packb(cmd, use_bin_type=True))
        msg = self.socket.recv()
        resp = msgpack.unpackb(msg, raw=False)
        resp["keys"] = keys
        return resp
    
    def search_nearby(self,text):
        cmd = {"cmd": "SEARCH_NEARBY", "text": text}
        self.socket.send(msgpack.packb(cmd, use_bin_type=True))
        msg = self.socket.recv()
        resp = msgpack.unpackb(msg, raw=False)
        return resp
    
    
    
if __name__ == '__main__':
    index_client = IndexClient("tcp://localhost:7783")
    # print(index_client.index_texts(["大家好","我来自北京"]))
    # print(index_client.index_texts(["你好","小猫咪真可爱"]))
    # print(index_client.index_texts(["User: 我有一个数学表达式，请为我算出结果。\nBot: 好的。请出题。\nUser:(25-8)*12\nBot:"]))
    # print(index_client.index_texts(["美丽突额隆头鱼（学名：Semicossyphus pulcher）是东太平洋特有的一种隆头鱼科鱼类，分布于美国加利福尼亚州的蒙特利湾至墨西哥的加利福尼亚湾。[2]美丽突额隆头鱼为日行性肉食鱼类，主要猎物为各种底栖无脊椎动物。其和其他隆头鱼科物种一样是雌性先熟顺序性雌雄同体，即所有鱼出生时均为雌性，后可能转变为雄性。此外，该鱼也可食用。 "]))
    query = "美丽突额隆头鱼以什么为主要猎物？"
    results = index_client.search_nearby(query)['value']
    print(results)
    documents = results["documents"][0]
    print(documents)
    from llm_client import LLMClient
    llm_client = LLMClient("tcp://localhost:7781")  
    results = llm_client.cross_encode([query for i in range(len(documents))],documents)
    print(results)
    #select max score index
    max_score_index = results["value"].index(max(results["value"]))
    print(f"Max score index {max_score_index}")
    selected_text = documents[max_score_index]
    print(f"Selected text {selected_text}")

    #generate answer
    prompt_text = f"User: 给你一段文本，请根据这段文本回答我的问题。\nBot: 好的，请问文本是什么？\nUser: {selected_text}\nBot:问题是什么？\nUser: {query}\nBot:好的，根据你提供的文本，你的问题的答案是："
    print(prompt_text)
    print("------------------------------")
    result = llm_client.generate(prompt_text,token_count=100)['value']
    next_round_index = result.find("User:")
    print(result[0:next_round_index])

    print(result)