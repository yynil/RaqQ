import msgpack
class CacheClient:
    def __init__(self, read_url, write_url):
        import zmq
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(read_url)
        self.socket.setsockopt(zmq.RCVTIMEO, 60000)
        self.writer_socket = self.context.socket(zmq.REQ)
        self.writer_socket.connect(write_url)
        self.writer_socket.setsockopt(zmq.RCVTIMEO, 60000)

    def get(self, key):
        cmd = {"cmd": "GET", "key": key}
        self.socket.send(msgpack.packb(cmd, use_bin_type=True))
        msg = self.socket.recv()
        resp = msgpack.unpackb(msg, raw=False)
        return resp
    
    def set(self, key, value):
        cmd = {"cmd": "SET", "key": key, "value": value}
        self.writer_socket.send(msgpack.packb(cmd, use_bin_type=True))
        msg = self.writer_socket.recv()
        resp = msgpack.unpackb(msg, raw=False)
        return resp
    
    def delete(self, key):
        cmd = {"cmd": "DELETE", "key": key}
        self.writer_socket.send(msgpack.packb(cmd, use_bin_type=True))
        msg = self.writer_socket.recv()
        resp = msgpack.unpackb(msg, raw=False)
        return resp
    
    def exists(self, key):
        cmd = {"cmd": "EXISTS", "key": key}
        self.socket.send(msgpack.packb(cmd, use_bin_type=True))
        msg = self.socket.recv()
        resp = msgpack.unpackb(msg, raw=False)
        return resp

if __name__ == '__main__':
    cache_client = CacheClient("tcp://localhost:7777","tcp://localhost:7779")
    print(cache_client.exists("key1"))
    print(cache_client.exists("key2"))
    print(cache_client.get("key1"))
    print(cache_client.get("key2"))
    print(cache_client.set("key1", "value1"))
    print(cache_client.get("key1"))
    print(cache_client.exists("key1"))
    print(cache_client.delete("key1"))
    print(cache_client.exists("key1"))
    print(cache_client.get("key1"))

    print(cache_client.set("key2", {"name": "John Doe", "age": 30,"codes":[1,2,3],"embeddings":[1.0,2.1111,-0.3323]}))
    print(cache_client.exists("key2"))
    print(cache_client.get("key2"))
    print(cache_client.delete("key2"))
    print(cache_client.exists("key2"))