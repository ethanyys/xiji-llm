import yaml


class YamlParser:
    @staticmethod
    def read_yaml_to_dict(yaml_path):
        with open(yaml_path, 'r', encoding='utf-8') as file_stream:
            yaml_dict = yaml.load(file_stream, Loader=yaml.Loader)
        return yaml_dict

    @staticmethod
    def save_dicts_to_yaml(dict_value, save_path):
        with open(save_path, 'w') as file:
            file.write(yaml.dump(dict_value, allow_unicode=True))

