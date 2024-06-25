if __name__ == "__main__":
    # For debugging
    import sys, os
    sys.path.append(os.path.dirname(sys.path[0]))

import qga_toolbox
from qga_toolbox.subroutines import cloning, sorting, mutation, mixing
from qga_toolbox.generators import problem_generators, state_generators
from qga_toolbox import feature_trackers

from typing import Callable

import yaml


keyword_type_map = {
    'track-features': list[type],
    'problem-generating-procedure': type,
    'initial-state-generating-procedure': type,
    'cloning': type,
    'sorting': type,
    'mixing': type,
    'mutation': type
}

keyword_module_map = {
    'track-features': feature_trackers,
    'problem-generating-procedure': problem_generators,
    'initial-state-generating-procedure': state_generators,
    'cloning': cloning,
    'sorting': sorting,
    'mutation': mutation,
    'mixing': mixing
}

common_keys = set(keyword_type_map.keys())
if  common_keys != set(keyword_module_map.keys()):
    raise RuntimeError('The keys in keyword_type_map and keyword_module_map should match.')

for key in common_keys:
    keyword_type_map[key.replace('-', '_')] = keyword_type_map[key]
    keyword_module_map[key.replace('-', '_')] = keyword_module_map[key]

global_arguments = ['population-size', 'chromosome-size']


def load_yaml_data(file_path: str):
    with open(file_path) as file:
        yaml_data_list = list(yaml.load_all(file, Loader=yaml.loader.SafeLoader))
    return yaml_data_list


def parse_values_to_pystrings(yaml_data: dict[str, object]) -> dict[str, object]:

    def _to_type_case(value: str | list[str], expected_type: type) -> str | list[str]:
        if expected_type == type:
            return ''.join(word.capitalize() for word in value.split('-'))
        elif expected_type == Callable:
            return value.replace('-', '_')
        elif expected_type == list[type]:
            return [_to_type_case(v, type) for v in value]
        elif expected_type == list[Callable]:
            return [_to_type_case(v, Callable) for v in value]
        else:
            raise ValueError(f"The expeted type {expected_type} is not supported.")
    
    parsed_data = dict()

    for keyword, value in yaml_data.items():
        if type(value) == dict:  # dicts are parsed recursively
            parsed_data[keyword] = parse_values_to_pystrings(value)
        elif keyword in keyword_type_map:  # custom parse for predefined keywords
            parsed_data[keyword] = _to_type_case(value, keyword_type_map[keyword])
        else:  # default parse for other keywords
            parsed_data[keyword] = value

    return parsed_data

def parse_attribute_dicts(input_dict: dict[str, object]) -> dict[str, object]:

    # Recursively scan the input_dict looking for callabes/types
    # and ensuring that their -attributes keyword exists
    def _ensure_attribute_key(adict: dict[str, object]) -> dict[str, object]:
        new_adict = dict()
        for key, value in adict.items():            
            if key in keyword_type_map:
                new_adict[key] = value
                new_adict[f'{key}-attributes'] = _ensure_attribute_key(adict.get(f'{key}-attributes', dict()))
            elif not key.endswith('-attributes'):
                if isinstance(value, dict):
                    new_adict[key] = _ensure_attribute_key(value)
                else:
                    new_adict[key] = value
            elif key.removesuffix('-attributes') not in adict:
                new_adict[key] = _ensure_attribute_key(value)
        return new_adict
                
    new_input_dict = _ensure_attribute_key(input_dict)

    # Including global attributes
    # This is done before replacing '-' with '_' to be compatible with input_dict
    def _include_global_attributes(adict: dict[str, object]) -> dict[str, object]:
        new_adict = dict()
        for key, value in adict.items():
            if isinstance(value, dict):
                new_adict[key] = _include_global_attributes(value)
                if key.endswith('-attributes'):
                    for global_arg in global_arguments:
                        new_adict[key][global_arg] = input_dict[global_arg]
            else:
                new_adict[key] = value
        return new_adict

    new_input_dict = _include_global_attributes(new_input_dict)

    # Replace '-' with '_' in EVERY key of the dict
    def _args_to_pyformat(adict: dict[str, object]) -> dict[str, object]:
        new_adict = dict()
        for key, value in adict.items():
            if isinstance(value, dict):
                new_adict[key.replace('-','_')] = _args_to_pyformat(value)
            else:
                new_adict[key.replace('-','_')] = value
        return new_adict
    
    return _args_to_pyformat(new_input_dict)


def parse_to_python_objects(input_dict: dict[str, object]):
    def _search_in_module(module, keyword: str | list[str], attributes: dict[str, object] = None):
        if isinstance(keyword, list):
            return [_search_in_module(module, kw, attributes) for kw in keyword]
        try:
            if attributes is None:
                return eval(f"{module.__name__}.{keyword}")
            else:
                return eval(f"{module.__name__}.{keyword}")(**attributes)
        except AttributeError as aerr:
            raise aerr

    # Using the search in module function and providing the arguments
    def _args_to_objects(adict: dict[str, object]) -> dict[str, object]:
        new_adict = dict()
        for key, value in adict.items():
            
            if key in keyword_module_map:
                # parse its attributes first
                new_adict[f'{key}_attributes'] = _args_to_objects(adict[f'{key}_attributes'])

                new_adict[key] = _search_in_module(keyword_module_map[key], value, new_adict[f'{key}_attributes'])
            elif not key.endswith('_attributes'):
                if isinstance(value, dict):
                    new_adict[key] = _args_to_objects(value)
                else:
                    new_adict[key] = value
            elif key.removesuffix('_attributes') not in adict:
                new_adict[key] = _args_to_objects(value)
        return new_adict

    return _args_to_objects(input_dict)


def parse_yaml(file_path: str) -> list[dict]:
    yaml_data_list = load_yaml_data(file_path)
    input_data_list = []

    for yaml_data in yaml_data_list:

        # Parse values to Python style
        input_data = parse_values_to_pystrings(yaml_data)
        
        # Parse attribute dictionaries
        input_data = parse_attribute_dicts(input_data)
        
        # Parse to python objects
        input_data = parse_to_python_objects(input_data)

        input_data_list.append(input_data)
                
    return input_data_list


if __name__ == "__main__":
    file_path = 'inputs/input_model.yaml'
    input_dt = parse_yaml(file_path)
    for input_dt_i in input_dt:
        for k, v in input_dt_i.items():
            print(k, v)
        print()