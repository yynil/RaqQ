import zmq
from configuration import Configuration
from helpers import start_service
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Service start")
    parser.add_argument("--service_config", help="Service configuration file",type=str,default="resources/services_new.yml")
    args = parser.parse_args()
    config = Configuration(args.service_config)
    services = config.config.keys()
    print(f"Starting services {services}")
    for service in services:
        if config.config[service]["enabled"]:
            config_service = config.config[service]
            print(f"Starting service {service}")
            service_module_name = config_service["service_module"]
            is_init_once = config_service.get("is_init_once",False)
            if is_init_once:
                print(f"Init once for {service_module_name}")
                import importlib
                module = importlib.import_module(service_module_name)
                module.init_once(config_service)
            start_service(service_module_name,config_service)