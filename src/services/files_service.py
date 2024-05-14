import sqlite3
import uuid
from helpers import ServiceWorker

table_name = 'file_status'
status_table_name = 'file_status'
chunk_table_name = 'chunk_status'
create_status_table_sql = "create table if not exists file_status (file_path text primary key, status text, last_updated text)"
create_chunk_table_sql = "create table if not exists chunk_status (file_path text, chunk_id integer, status text,content text,uuid text, last_updated text, primary key(file_path,chunk_id))"
valid_status = ['waitinglist','processing','processed','failed']
import time

class FileStatusManager:
    def __init__(self,db_path) -> None:
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        #check if the table exists
        self.conn.execute(create_status_table_sql)
        self.conn.execute(create_chunk_table_sql)
        self.conn.commit()
    
    def get_file_status(self,file_path):
        cursor = self.conn.cursor()
        cursor.execute(f"select status from {status_table_name} where file_path = ?",(file_path,))
        result = cursor.fetchone()
        if result is None:
            return None
        return result[0]
    
    def get_chunks_status(self,file_path):
        cursor = self.conn.cursor()
        cursor.execute(f"select chunk_id,status,uuid from {chunk_table_name} where file_path = ?",(file_path,))
        result = cursor.fetchall()
        if result is None:
            return None
        return result
    
    def get_chunks_by_status(self,file_path,status):
        cursor = self.conn.cursor()
        cursor.execute(f"select chunk_id,uuid from {chunk_table_name} where file_path = ? and status = ?",(file_path,status))
        result = cursor.fetchall()
        if result is None:
            return None
        return result
    
    def add_file_chunks(self,file_path,texts):
        cursor = self.conn.cursor()
        uuids = [str(uuid.uuid4()) for _ in range(len(texts))]
        for i,text in enumerate(texts):
            cursor.execute(f"insert into {chunk_table_name} (file_path,chunk_id,status,content,uuid) values (?,?,?,?,?)",(file_path,i,'waitinglist',text,uuids[i]))
        self.conn.commit()
        return uuids

    def set_file_status(self,file_path,status):
        if status not in valid_status:
            raise ValueError(f"Invalid status {status}")
        cursor = self.conn.cursor()
        cursor.execute(f"update {status_table_name} set status = ?,last_updated = datetime('now') where file_path = ?",(status,file_path))
        self.conn.commit()

    def set_chunk_status(self,file_path,chunk_id,status):
        if status not in valid_status:
            raise ValueError(f"Invalid status {status}")
        cursor = self.conn.cursor()
        cursor.execute(f"update {chunk_table_name} set status = ?,last_updated = datetime('now') where file_path = ? and chunk_id = ?",(status,file_path,chunk_id))
        self.conn.commit()

    def delete_file(self,file_path):
        cursor = self.conn.cursor()
        cursor.execute(f"delete from {status_table_name} where file_path = ?",(file_path,))
        cursor.execute(f"delete from {chunk_table_name} where file_path = ?",(file_path,))
        self.conn.commit()

    def close(self):
        self.conn.close()


    def delte_file_chunks(self,file_path):
        cursor = self.conn.cursor()
        cursor.execute(f"delete from {chunk_table_name} where file_path = ?",(file_path,))
        self.conn.commit()

    def get_all_files(self):
        cursor = self.conn.cursor()
        cursor.execute(f"select file_path from {status_table_name}")
        result = cursor.fetchall()
        if result is None:
            return None
        return [r[0] for r in result]
    
    def check_file_exists(self,file_path):
        cursor = self.conn.cursor()
        cursor.execute(f"select count(*) from {status_table_name} where file_path = ?",(file_path,))
        result = cursor.fetchone()
        return result[0] > 0
    
    def check_chunk_exists(self,file_path,chunk_id):
        cursor = self.conn.cursor()
        cursor.execute(f"select count(*) from {chunk_table_name} where file_path = ? and chunk_id = ?",(file_path,chunk_id))
        result = cursor.fetchone()
        return result[0] > 0
    
    def get_chunk_content(self,file_path,chunk_id):
        cursor = self.conn.cursor()
        cursor.execute(f"select content from {chunk_table_name} where file_path = ? and chunk_id = ?",(file_path,chunk_id))
        result = cursor.fetchone()
        return result[0]
    
    def get_chunk_uuid(self,file_path,chunk_id):
        cursor = self.conn.cursor()
        cursor.execute(f"select uuid from {chunk_table_name} where file_path = ? and chunk_id = ?",(file_path,chunk_id))
        result = cursor.fetchone()
        return result[0]
    
class FileStatusMonitorThread:
    def __init__(self,fsm:FileStatusManager) -> None:
        self.fsm = fsm

    def run(self):
        print("Starting FileStatusMonitorThread")
        while(True):
            time.sleep(30)
            print("Checking file status")

def chunk_file_contents(file_path,chunk_size=1024):
    #This is a demo function to chunk a file into smaller pieces
    chunks = []
    with open(file_path) as f:
        str_input = ""
        for line in f:
            line = line.strip()
            if len(str_input) + len(line) > chunk_size:
                chunks.append(str_input)
                str_input = ""
            str_input += line
        if len(str_input) > 0:
            chunks.append(str_input)
    return chunks

class ServiceWorker(ServiceWorker):
    def init_with_config(self,config):
        self.db_path = config['db_path']
        self.fsm = FileStatusManager(self.db_path)

    def process(self,cmd):
        if cmd['cmd'] == 'ADD_FILE':
            file_path = cmd['file_path']
            if self.fsm.check_file_exists(file_path):
                return "File already exists"
            else:
                self.fsm.set_file_status(file_path,'processing')
                texts = chunk_file_contents(file_path)
                uuids = self.fsm.add_file_chunks(file_path,texts)
                return uuids
        elif cmd['cmd'] == 'GET_FILE_STATUS':
            file_path = cmd['file_path']
            status = self.fsm.get_file_status(file_path)
            return status
        elif cmd['cmd'] == 'GET_CHUNK_STATUS':
            file_path = cmd['file_path']
            status = self.fsm.get_chunks_status(file_path)
            return status
        elif cmd['cmd'] == 'GET_CHUNK_CONTENT':
            file_path = cmd['file_path']
            chunk_id = cmd['chunk_id']
            content = self.fsm.get_chunk_content(file_path,chunk_id)
            return content
        elif cmd['cmd'] == 'GET_CHUNK_UUID':
            file_path = cmd['file_path']
            chunk_id = cmd['chunk_id']
            uuid = self.fsm.get_chunk_uuid(file_path,chunk_id)
            return uuid
        elif cmd['cmd'] == 'GET_CHUNKS_BY_STATUS':
            file_path = cmd['file_path']
            status = cmd['status']
            chunks = self.fsm.get_chunks_by_status(file_path,status)
            return chunks
        elif cmd['cmd'] == 'CHECK_FILE_EXISTS':
            file_path = cmd['file_path']
            exists = self.fsm.check_file_exists(file_path)
            return exists 
        else:
            return ServiceWorker.UNSUPPORTED_COMMAND
   
if __name__ == '__main__':
    file_name = 'README.md'
    file_content = chunk_file_contents(file_name)
    db_path = 'file_status.db'
    fsm = FileStatusManager(db_path)
    fsm.set_file_status(file_name,'processing')
    print(fsm.add_file_chunks(file_name,file_content))
    print(fsm.get_file_status(file_name))
    print(fsm.get_chunks_status(file_name))
    fsm.set_chunk_status(file_name,0,'processed')
    print(fsm.get_chunk_content(file_name,0))
    print(fsm.get_chunk_uuid(file_name,0))
    print(fsm.get_chunks_status(file_name))
    fsm.delete_file(file_name)
    print(fsm.get_file_status(file_name))
    fsm.close()

    from threading import Thread
    monitor_thread = FileStatusMonitorThread(fsm)
    t = Thread(target=monitor_thread.run)
    t.start()
    t.join()