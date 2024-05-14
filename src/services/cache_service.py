import sqlitedict
import time
import msgpack
import zmq
from helpers import start_proxy,ServiceWorker

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
    

class CacheReaderService(ServiceWorker):
    def init_with_config(self, config):
        self.cache_service = CacheService(config["db_file"])
    
    def process(self, cmd):
        if cmd["cmd"] == "GET":
            key = cmd["key"]
            value = self.cache_service.get_cache(key)
            return value
        elif cmd["cmd"] == "EXISTS":
            key = cmd["key"]
            value = self.cache_service.exists(key)
            return value
        return ServiceWorker.UNSUPPORTED_COMMAND


class CacheWriterService(ServiceWorker):
    def init_with_config(self, config):
        self.cache_service = CacheService(config["db_file"])
    
    def process(self, cmd):
        if cmd['cmd'] == 'SET':
            key = cmd['key']
            value = cmd['value']
            self.cache_service.set_cache(key, value)
            return True
        elif cmd['cmd'] == 'DELETE':
            key = cmd['key']
            self.cache_service.delete_cache(key)
            return True
        return ServiceWorker.UNSUPPORTED_COMMAND

