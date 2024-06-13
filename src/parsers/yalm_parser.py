if __name__ == "__main__":
    # For debugging
    import sys, os
    sys.path.append(os.path.dirname(sys.path[0]))
    
import subroutines
import feature_trackers
import generators.problem_generators
import generators.state_generators
import subroutines.cloning
import subroutines.mutation
import subroutines.sorting
import subroutines.mixing

from typing import Literal, Callable

import yaml


keyword_type_map = {
    'track-features': list[type],
    'problem-generating-procedure': type,
    'initial-state-generating-procedure': type,
    'cloning': type,
    'sorting': type,
    'mutation': type
}

keyword_module_map = {
    'track-features': feature_trackers,
    'problem-generating-procedure': generators.problem_generators,
    'initial-state-generating-procedure': generators.state_generators,
    'cloning': subroutines.cloning,
    'sorting': subroutines.sorting,
    'mutation': subroutines.mutation,
    'mixing': subroutines.mixing
}

global_arguments = ['population-size', 'chromosome-size']


def _parse_values_to_pystrings(yaml_data: dict[str, object]) -> dict[str, object]:

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
            parsed_data[keyword] = _parse_values_to_pystrings(value)
        elif keyword in keyword_type_map:  # custom parse for predefined keywords
            parsed_data[keyword] = _to_type_case(value, keyword_type_map[keyword])
        else:  # default parse for other keywords
            parsed_data[keyword] = value

    return parsed_data


def _parse_attribute_dicts(input_dict: dict[str, object]) -> dict[str, object]:
    output_dict = input_dict.copy()

    def _parse_single_attribue_dict(attr_dict: dict[str, object]):
        new_attr_dict = attr_dict.copy()
        
        # Including global arguments
        for gaargs in global_arguments:
            new_attr_dict[gaargs] = output_dict[gaargs]

        new2_attr_dict = dict()

        # Using correct formating
        for key, value in new_attr_dict.items():
            if key.split('-')[-1] == 'attributes':
                new2_attr_dict[key] = _parse_single_attribue_dict(value)
            else:
                new2_attr_dict[key.replace('-', '_')] = value

        return new2_attr_dict
    
    def _parse_qga_attribue_dict(qga_dict: dict[str, object]):
        new2_qga_dict = dict()

        # Using correct formating
        for key, value in qga_dict.items():
            if keyword in keyword_type_map:
                if (keyword_type_map[key] in [type, Callable]):
                    key_attrs = f"{key}-attributes"
                    attr_dict = qga_dict.get(key_attrs, dict())
                    new2_qga_dict[key_attrs] = _parse_single_attribue_dict(attr_dict)
                elif (keyword_type_map[keyword] in [list[type], list[Callable]]):
                    # Decide how to parse these arguments and what input formats will be supported
                    pass
            else:
                new2_qga_dict[key.replace('-', '_')] = value

        return new2_qga_dict
    
    for keyword, value in input_dict.items():
         # For type class or function find or initiate its attribute dict
        if keyword in keyword_type_map:
            if (keyword_type_map[keyword] in [type, Callable]):
                key_attrs = f"{keyword}-attributes"
                attr_dict = output_dict.get(key_attrs, dict())
                output_dict[key_attrs] = _parse_single_attribue_dict(attr_dict)

            elif (keyword_type_map[keyword] in [list[type], list[Callable]]):
               # Decide how to parse these arguments and what input formats will be supported
               pass
        elif keyword == 'qga-attributes':
            output_dict[keyword] = _parse_qga_attribue_dict(value)

                            
    return output_dict

"""
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


def _parse_attribute_dict(attr_dict: dict[str, object], input_dict: dict[str, object]):
    new_attr_dict = dict()
    new_attr_dict['population_size'] = input_dict['population-size']
    new_attr_dict['chromosome_size'] = input_dict['chromosome-size']
    for key, value in attr_dict.items():
        if key.split('-')[-1] != 'attributes':
            new_attr_dict[key.replace('-', '_')] = value
        else:
            new_attr_dict[key] = value
    return new_attr_dict
"""

def _parse_to_python_objects(input_dict: dict[str, object]):
    # Using the search in module function and providing the arguments
    return input_dict

def parse_yaml(file_path: str) -> list[dict]:
    with open(file_path) as file:
        yaml_data_list = list(yaml.load_all(file, Loader=yaml.loader.SafeLoader))
    input_data_list = []

    for yaml_data in yaml_data_list:

        # Parse values to Python style
        input_data = _parse_values_to_pystrings(yaml_data)

        # Parse attribute dictionaries
        input_data = _parse_attribute_dicts(input_data)

        # Parse to python objects
        input_data = _parse_to_python_objects(input_data)

        input_data_list.append(input_data)
        continue

        for key_attr, attr_dict in yaml_data.items() :
            if key_attr.split('-')[-1] == 'attributes':
                input_data[key_attr] = _parse_attribute_dict(attr_dict, yaml_data)

        input_data['track-features'] = {feat_name: _search_in_module(feature_trackers, feat_name, 'class')() for feat_name in yaml_data['track-features']}
        input_data['problem-generating-procedure'] = _search_in_module(generators.problem_generators, yaml_data['problem-generating-procedure'], 'class')(**input_data["problem-generating-procedure-attributes"])
        input_data['initial-state-generating-procedure'] = _search_in_module(generators.state_generators, yaml_data['initial-state-generating-procedure'], 'class')(**input_data["initial-state-generating-procedure-attributes"])

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
        
        
    return input_data_list


if __name__ == "__main__":
    file_path = 'inputs/input_model.yaml'
    input_dt = parse_yaml(file_path)
    for input_dt_i in input_dt:
        for k, v in input_dt_i.items():
            print(k, v)
        print()