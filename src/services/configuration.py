import yaml

class Configuration:
    def __init__(self, config_file):
        with open(config_file) as f:
            self.config = yaml.safe_load(f)
    
if __name__ == "__main__":
    config = Configuration("resources/services.yml")
    print(config.config)
    print(config.config["response"]["reader"])