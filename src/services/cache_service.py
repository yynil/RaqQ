import sqlitedict
import time
import msgpack
import zmq
from helpers import start_proxy
class CacheService:
    def __init__(self, db_path):
        self.db = sqlitedict.SqliteDict(db_path, autocommit=True)
        base_name_db = db_path.split(".")[0]
        db_path = base_name_db+"_time_stamp.db"
        self.time_stamp_db = sqlitedict.SqliteDict(db_path, autocommit=True)

    def get_cache(self, key):
        if key not in self.db:
            return None
        current_time = time.time()
        self.time_stamp_db[key] = current_time
        return self.db.get(key)
    
    def exists(self, key):
        return key in self.db

    def set_cache(self, key, value):
        current_time = time.time()
        self.time_stamp_db[key] = current_time
        self.db[key] = value
        return True
    
    def delete_cache(self, key):
        if key in self.db:
            del self.db[key]
        if key in self.time_stamp_db:
            del self.time_stamp_db[key]
        return True
    
def reader_worker(db_file,reader_backend_url):
    print(f"Reader worker started with db file {db_file} and reader {reader_backend_url}")
    cache_service = CacheService(db_file)
    import zmq
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.connect(reader_backend_url)
    print(f"Reader worker connected to {reader_backend_url}")
    while True:
        message = socket.recv()
        cmd = msgpack.unpackb(message, raw=False)
        if cmd["cmd"] == "GET":
            key = cmd["key"]
            value = cache_service.get_cache(key)
            value = {"value": value,"code": 200}
            socket.send(msgpack.packb(value, use_bin_type=True))
        elif cmd["cmd"] == "EXISTS":
            key = cmd["key"]
            value = cache_service.exists(key)
            value = {"exists": value,"code": 200}
            socket.send(msgpack.packb(value, use_bin_type=True))
        else:
            socket.send(msgpack.packb({"code": 400}, use_bin_type=True))

def writer_worker(db_file,writer_backend_url):
    print(f"Writer worker started with db file {db_file} and writer {writer_backend_url}")
    cache_service = CacheService(db_file)
    import zmq
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.connect(writer_backend_url)
    print(f"Writer worker connected to {writer_backend_url}")
    while True:
        message = socket.recv()
        cmd = msgpack.unpackb(message, raw=False)
        if cmd['cmd'] == 'SET':
            key = cmd['key']
            value = cmd['value']
            cache_service.set_cache(key, value)
            socket.send(msgpack.packb({"code": 200}, use_bin_type=True))
        elif cmd['cmd'] == 'DELETE':
            key = cmd['key']
            cache_service.delete_cache(key)
            socket.send(msgpack.packb({"code": 200}, use_bin_type=True))
        else:
            socket.send(msgpack.packb({"code": 400}, use_bin_type=True))
def start_cache_service(config):
    db_file = config["db_file"]
    reader_size = config["reader"]["num_workers"]
    print(f"Starting cache service with db file {db_file} and reader size {reader_size} and writer size 1")
    reader_backend_url = config["reader"]["back_end"]["protocol"]+"://"+config["reader"]["host"]+":"+str(config["reader"]["back_end"]["port"])
    writer_backend_url = config["writer"]["back_end"]["protocol"]+"://"+config["writer"]["host"]+":"+str(config["writer"]["back_end"]["port"])
    reader_frontend_url = config["reader"]["front_end"]["protocol"]+"://"+config["reader"]["host"]+":"+str(config["reader"]["front_end"]["port"])
    writer_frontend_url = config["writer"]["front_end"]["protocol"]+"://"+config["writer"]["host"]+":"+str(config["writer"]["front_end"]["port"])
    import multiprocessing
    readers_process = []
    for i in range(reader_size):
        process = multiprocessing.Process(target=reader_worker, args=(db_file,reader_backend_url))
        readers_process.append(process)
        process.start()
    writer_process = multiprocessing.Process(target=writer_worker, args=(db_file,writer_backend_url))
    writer_process.start()
    

    print(f'start reader proxy {reader_frontend_url} {reader_backend_url}')
    multiprocessing.Process(target=start_proxy, args=(reader_frontend_url,reader_backend_url)).start()

    print(f'start writer proxy {writer_frontend_url} {writer_backend_url}')
    multiprocessing.Process(target=start_proxy, args=(writer_frontend_url,writer_backend_url)).start()