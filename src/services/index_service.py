CHROMA_DB_COLLECTION_NAME = 'chroma_db'
from helpers import ServiceWorker
import chromadb

def init_once(config):
    chroma_path = config["chroma_path"]
    chroma_port = config["chroma_port"]
    chroma_host = config["chroma_host"]
    print(f"Start chroma db")
    #spawn a process "chroma run --path chroma_path --port chroma_port --host chroma_host" 
    import subprocess
    command = f"chroma run --path {chroma_path} --port {chroma_port} --host {chroma_host}"
    process = subprocess.Popen(command,shell=True)
    print(f"Started indexing service with command {command}, pid is {process.pid}")
    import time
    time.sleep(5)
class ServiceWorker(ServiceWorker):
    _has_run_init = False
    @staticmethod
    def init_chroma_db(cls,chroma_path,chroma_host,chroma_port):
        if not cls._has_run_init:
            print(f"Start chroma db")
            #spawn a process "chroma run --path chroma_path --port chroma_port --host chroma_host" 
            import subprocess
            command = f"chroma run --path {chroma_path} --port {chroma_port} --host {chroma_host}"
            process = subprocess.Popen(command,shell=True)
            print(f"Started indexing service with command {command}, pid is {process.pid}")
            cls._has_run_init = True
            print(f"Chroma db is started")
        else:
            print(f"Chroma db is already started")
    def init_with_config(self, config):
        
        llm_front_end_url = config["llm_front_end_url"]
        from clients.llm_client import LLMClient
        self.llm_client = LLMClient(llm_front_end_url)
        chroma_port = config["chroma_port"]
        chroma_host = config["chroma_host"]
        
        #Init the chromadb if needed
        chroma_client = chromadb.HttpClient(host=chroma_host,
                                            port=chroma_port)
        if CHROMA_DB_COLLECTION_NAME not in [c.name for c in chroma_client.list_collections()]:
            chroma_client.create_collection(CHROMA_DB_COLLECTION_NAME,
                                            metadata={"hnsw:space": "cosine"})
        print(f"Chroma db collection {CHROMA_DB_COLLECTION_NAME} is created")
        print(f"Chroma db collection {CHROMA_DB_COLLECTION_NAME} is ready")
        print(f'Current collections are {chroma_client.list_collections()}')
        del chroma_client
        self.chroma_host = chroma_host
        self.chroma_port = chroma_port
        
    def process(self, cmd):
        if cmd['cmd'] == 'INDEX_TEXTS':
            keys = cmd["keys"]
            values = cmd["texts"]
            embeddings = self.llm_client.encode(values)["value"]

            chroma_client = chromadb.HttpClient(host=self.chroma_host,
                                                port=self.chroma_port)
            
            collection = chroma_client.get_collection(CHROMA_DB_COLLECTION_NAME)
            
            collection.add(
                ids=keys,
                embeddings=embeddings,
                documents=values
            )

            #index the value
            return True
        elif cmd['cmd'] == 'SEARCH_NEARBY':
            text = cmd["text"]
            embedings = self.llm_client.encode([text])["value"]
            print(f"Searching nearby for {text} with embeddings {embedings}")
            chroma_client = chromadb.HttpClient(host=self.chroma_host,
                                                port=self.chroma_port)
            collection = chroma_client.get_collection(CHROMA_DB_COLLECTION_NAME)
            search_result = collection.query(
                query_embeddings=embedings,
                n_results=3,
                include=['documents','distances'])
            return search_result
        return ServiceWorker.UNSUPPORTED_COMMAND
