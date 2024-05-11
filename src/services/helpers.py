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