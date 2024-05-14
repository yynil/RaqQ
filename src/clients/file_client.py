import msgpack

class FileClient:
    def __init__(self,frontend_url) -> None:
        import zmq
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(frontend_url)
        self.socket.setsockopt(zmq.RCVTIMEO, 60000)

    def check_file_exists(self,file_path):
        cmd = {"cmd": "CHECK_FILE_EXISTS", "file_path": file_path}
        self.socket.send(msgpack.packb(cmd, use_bin_type=True))
        msg = self.socket.recv()
        resp = msgpack.unpackb(msg, raw=False)
        return resp["exists"]
    
    def add_file(self,file_path):
        cmd = {"cmd": "ADD_FILE", "file_path": file_path}
        self.socket.send(msgpack.packb(cmd, use_bin_type=True))
        msg = self.socket.recv()
        resp = msgpack.unpackb(msg, raw=False)
        return resp["uuids"]
    

if __name__ == '__main__':
    front_end_url = "tcp://localhost:7785"
    file_client = FileClient(front_end_url)
    file_name = "/home/yueyulin/tmp/1.txt"
    print(file_client.check_file_exists(file_name))
    print(file_client.add_file(file_name))
    print(file_client.check_file_exists(file_name))