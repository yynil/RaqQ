CHROMA_DB_COLLECTION_NAME = 'chroma_db'
from helpers import start_proxy

def indexing_worker(backend_url,llm_front_end_url,chroma_host,chroma_port):
    import zmq
    import os
    import msgpack
    import sys
    import chromadb
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from clients.llm_client import LLMClient
    llm_client = None
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.connect(backend_url)
    print(f"\033[93m Indexing worker connected to {backend_url} at process {os.getpid()}\033[0m")   
    while True:
        message = socket.recv()
        if llm_client is None:
            llm_client = LLMClient(llm_front_end_url)
        print(f"\033[93m Connected to LLMClient \033[0m")
        cmd = msgpack.unpackb(message, raw=False)
        if cmd["cmd"] == "INDEX_TEXTS":
            keys = cmd["keys"]
            values = cmd["texts"]
            embeddings = llm_client.encode(values)["value"]

            chroma_client = chromadb.HttpClient(host=chroma_host,
                                                port=chroma_port)
            
            collection = chroma_client.get_collection(CHROMA_DB_COLLECTION_NAME)
            
            collection.add(
                ids=keys,
                embeddings=embeddings,
                documents=values
            )

            #index the value
            socket.send(msgpack.packb({"code": 200}, use_bin_type=True))
        elif cmd["cmd"] == "SEARCH_NEARBY":
            text = cmd["text"]
            embedings = llm_client.encode([text])["value"]
            print(f"Searching nearby for {text} with embeddings {embedings}")
            chroma_client = chromadb.HttpClient(host=chroma_host,
                                                port=chroma_port)
            collection = chroma_client.get_collection(CHROMA_DB_COLLECTION_NAME)
            search_result = collection.query(
                query_embeddings=embedings,
                n_results=3,
                include=['documents','distances'])
            socket.send(msgpack.packb({"code": 200,"result": search_result}, use_bin_type=True))        
        else:
            socket.send(msgpack.packb({"code": 400}, use_bin_type=True))



def start_indexing_service(config):
    chroma_path = config["chroma_path"]
    chroma_port = config["chroma_port"]
    chroma_host = config["chroma_host"]

    #spawn a process "chroma run --path chroma_path --port chroma_port --host chroma_host" 
    import subprocess
    command = f"chroma run --path {chroma_path} --port {chroma_port} --host {chroma_host}"
    process = subprocess.Popen(command,shell=True)
    print(f"Started indexing service with command {command}, pid is {process.pid}")
    import time
    time.sleep(5)
    #Init the chromadb if needed
    import chromadb
    chroma_client = chromadb.HttpClient(host=chroma_host,
                                        port=chroma_port)
    if CHROMA_DB_COLLECTION_NAME not in [c.name for c in chroma_client.list_collections()]:
        chroma_client.create_collection(CHROMA_DB_COLLECTION_NAME,
                                        metadata={"hnsw:space": "cosine"})
    print(f"Chroma db collection {CHROMA_DB_COLLECTION_NAME} is created")
    print(f"Chroma db collection {CHROMA_DB_COLLECTION_NAME} is ready")
    print(f'Current collections are {chroma_client.list_collections()}')
    del chroma_client
    front_end_url = config["front_end"]["protocol"] + "://" + config["front_end"]["host"] + ":" + str(config["front_end"]["port"])
    print(f"Indexing service front end url is {front_end_url}")
    backend_url = config["back_end"]["protocol"] + "://" + config["back_end"]["host"] + ":" + str(config["back_end"]["port"])
    print(f"Indexing service back end url is {backend_url}")
    num_workers = config.get("num_workers",1)
    print(f"Starting {num_workers} workers")
    
    import multiprocessing
    llm_front_end_url = config["llm_front_end_url"]
    for i in range(num_workers):
        process = multiprocessing.Process(target=indexing_worker, args=(backend_url,llm_front_end_url,chroma_host,chroma_port))
        process.start()

    
    print(f'Starting proxy with front end url {front_end_url} and back end url {backend_url}')
    process = multiprocessing.Process(target=start_proxy, args=(front_end_url,backend_url))
    process.start()
    print(f"\033[92mIndexing service started\033[0m")