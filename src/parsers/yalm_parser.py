import subroutines
import feature_trackers
import generators
import subroutines.cloning
import subroutines.mutation
import subroutines.sorting

from typing import Literal

import yaml

def _search_in_module(module, keyword: str, typestr: Literal['class', 'function']):
    # may raise AttributeError
    if typestr == 'class':
        keyword = ''.join(word.capitalize() for word in keyword.split('-'))
    elif typestr == 'function':
        keyword = keyword.replace('-', '_')
    
    try:
        return eval(f"{module.__name__}.{keyword}")
    except AttributeError as aerr:
        if typestr == 'class':
            keyword = keyword.upper()
            return eval(f"{module.__name__}.{keyword}")
        else:
            raise aerr


def _parse_attribute_dict(attr_dict: dict[str, object], input_data: dict[str, object]):
    new_attr_dict = dict()
    new_attr_dict['population_size'] = input_data['population-size']
    new_attr_dict['chromosome_size'] = input_data['chromosome-size']
    for key, value in attr_dict.items():
        new_attr_dict[key.replace('-', '_')] = value
    return new_attr_dict


def parse_yalm(file_path: str) -> list[dict]:
    with open(file_path) as file:
        yaml_data_list = list(yaml.load_all(file, Loader=yaml.loader.SafeLoader))
    input_data_list = []

    for yaml_data in yaml_data_list:
        input_data = yaml_data.copy()

        # Initialize these!!
        # Any attribute dict should also get the population size and individuals
        input_data['track-features'] = {feat_name: _search_in_module(feature_trackers, feat_name, 'class')() for feat_name in yaml_data['track-features']}
        input_data['problem-generating-procedure'] = _search_in_module(generators.problem_generators, yaml_data['problem-generating-procedure'], 'function')
        input_data['initial-state-generating-procedure'] = _search_in_module(generators.state_generators, yaml_data['initial-state-generating-procedure'], 'function')

        for key_attr, attr_dict in yaml_data.items() :
            if key_attr.split('-')[-1] == 'attributes':
                input_data[key_attr] = _parse_attribute_dict(attr_dict, yaml_data)

        qga_attributes = input_data['qga-attributes']
        
        for initializing_subroutine in ['cloning', 'sorting', 'mutation']:
            key_attr = f'{initializing_subroutine}-attributes'
            if key_attr in qga_attributes:
                qga_attributes[key_attr] = _parse_attribute_dict(attr_dict, yaml_data)
            else:
                qga_attributes[key_attr] = _parse_attribute_dict(dict(), yaml_data)
            
            qga_attributes[initializing_subroutine] = _search_in_module(eval(f"subroutines.{initializing_subroutine}"), 
                                                                        qga_attributes[initializing_subroutine], 
                                                                        'class')(**qga_attributes[key_attr])
            
        input_data_list.append(input_data)
    return input_data_list


if __name__ == "__main__":
    file_path = 'inputs/input_model.yaml'
    input_data = parse_yalm(file_path)
    for input_data in input_data:
        for k, v in input_data.items():
            print(k, v)
        print()