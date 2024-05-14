import zmq


def start_proxy(frontend_url, backend_url):
    import zmq
    print(f'\033[91mstart proxy {frontend_url} {backend_url}\033[0m')
    context = zmq.Context()
    frontend = context.socket(zmq.ROUTER)
    frontend.bind(frontend_url)
    backend = context.socket(zmq.DEALER)
    backend.bind(backend_url)
    zmq.proxy(frontend, backend)

from abc import ABC, abstractmethod
import os
import msgpack

 

class ServiceWorker(ABC):
    UNSUPPORTED_COMMAND = 'Unsupported command'
    def __init__(self,backend_url,config):
        self.init_with_config(config)
        self.backend_url = backend_url
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.connect(backend_url)
        print(f"\033[93m Service worker {self.__class__.__name__} connected to {backend_url} at process {os.getpid()}\033[0m")
    
    @abstractmethod
    def init_with_config(self,config):
        pass

    @abstractmethod
    def process(self,cmd):
        pass
    
    def run(self):
        while True:
            message = self.socket.recv()
            cmd = msgpack.unpackb(message, raw=False)
            try:
                resp = self.process(cmd)
                if resp==ServiceWorker.UNSUPPORTED_COMMAND:
                    resp = {"code": 400,"error": "Unsupported command"}
                else:
                    resp = {"code": 200, "value": resp}
                self.socket.send(msgpack.packb(resp, use_bin_type=True))
            except Exception as e:
                print(f"Error processing command {cmd} with error {e}")
                resp = {"code": 400,"error": str(e)}
                self.socket.send(msgpack.packb(resp, use_bin_type=True))

def start_process(module_name,backend_url,config):
    import importlib
    module = importlib.import_module(module_name)
    class_name = config.get("class_name","ServiceWorker")
    service_cls = getattr(module,class_name)
    service_instance = service_cls(backend_url,config)
    print(f"\033[93mStarting service worker {service_cls} with backend url {backend_url} at process {os.getpid()}\033[0m")
    service_instance.run()

def start_service(service_cls :str,config):
    backend_url = config["back_end"]["protocol"] + "://" + config["back_end"]["host"] + ":" + str(config["back_end"]["port"])
    frontend_url = config["front_end"]["protocol"] + "://" + config["front_end"]["host"] + ":" + str(config["front_end"]["port"])
    print(f"\033[91mStarting service {service_cls} with backend url {backend_url} and frontend url {frontend_url}\033[0m")
    num_workers = config.get("num_workers",1)
    spawn_method = config.get("spawn_method","fork")
    import multiprocessing
    multiprocessing.set_start_method(spawn_method, force=True)
    print(f'\033[91mStarting {num_workers} workers\033[0m')
    for i in range(num_workers):
       multiprocessing.Process(target=start_process, args=(service_cls,backend_url,config)).start()
    print(f'\033[91mstart proxy {frontend_url} {backend_url}\033[0m')
    multiprocessing.Process(target=start_proxy, args=(frontend_url,backend_url)).start()
    print(f"\033[91mService {service_cls} started\033[0m")