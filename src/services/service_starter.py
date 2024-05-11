import zmq
from configuration import Configuration
from cache_service import start_cache_service
from llm_service import start_llm_service
from index_service import start_indexing_service
starters = {
    "cache": start_cache_service,
    "llm": start_llm_service,
    "indexing": start_indexing_service
}
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Service start")
    parser.add_argument("--service_config", help="Service configuration file",type=str,default="resources/services.yml")
    args = parser.parse_args()
    config = Configuration(args.service_config)
    services = config.config.keys()
    print(f"Starting services {services}")
    for service in services:
        if service in starters :
            if config.config[service]["enabled"]:
                print(f"Starting service {service}")
                starters[service](config.config[service])
            else:
                print(f"Service {service} is disabled")
        else:
            print(f"No starter found for service {service}")