import argparse
import os
import json

def parse_json(json_file):
    with open(json_file) as f:
      attrs = json.load(f)
      params = {}
    [params[k] = v for k, v in attrs.iteritems()]

    return params


#Get input argument for config python file
parser = argparse.argument_parser()

parser.add_argument('config_file')
parser.add_argument('json_file')

args = parser.parse_args()

config_file = os.path.abspath(args.config_file)
json_file   = os.path.abspath(args.json_file)

if ".py" not in config_file:
    raise RuntimeError('{} not a .py file, config file must be a python file'.format(config_file))

if ".json" not in json_file:
    raise RuntimeError('{} not a .json file, json file must be a json file'.format(json_file))

config_name = config_file.split('/')[-1]
config_dir  = os.path.abspath(config_file.replace(config_name,''))

sys.path.append(config_dir)

#Import config file and get relevant objects
config = __import__(config_file.replace('.py',''))
params = parse_json(json_file)

experiment = config.configure(params)

experiment.initialize()

experiment.train()

experiment.finalize()
