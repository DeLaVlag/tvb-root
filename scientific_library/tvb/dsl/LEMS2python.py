# -*- coding: utf-8 -*-

"""
LEMS2python module implements a DSL code generation using a TVB-specific LEMS-based DSL.

.. moduleauthor:: Michiel. A. van der Vlag <m.van.der.vlag@fz-juelich.de>   
.. moduleauthor:: Marmaduke Woodman <marmaduke.woodman@univ-amu.fr>

"""

from mako.template import Template
import tvb
import os
from tvb.basic.logger.builder import get_logger
from tvb.dsl.NeuroML.lems.model.model import Model

import csv
import numpy as np

from tvbtest_dataset.tvbtest_dataset import *

from kgquery.queryApi import KGClient

# from jupyter_collab_storage import oauth_token_handler as oauth

logger = get_logger(__name__)

def default_lems_folder():
    here = os.path.dirname(os.path.abspath(__file__))
    xmlpath = os.path.join(here, 'NeuroML', 'XMLmodels')
    return xmlpath


def lems_file(model_name, folder=None):
    folder = folder or default_lems_folder()
    return os.path.join(folder, model_name.lower() + '.xml')


def load_model(model_filename, folder=None):
    "Load model from filename"

    fp_xml = lems_file(model_filename, folder)

    # instantiate LEMS lib
    model = Model()
    model.import_from_file(fp_xml)

    # do some inventory. check if boundaries are set for any sv to print the boundaries section in template
    svboundaries = 0
    for i, sv in enumerate(model.component_types[model_filename].dynamics.state_variables):
        if sv.boundaries != 'None' and sv.boundaries != '' and sv.boundaries:
            svboundaries = 1
            continue

    return model, svboundaries


def default_template():
    here = os.path.dirname(os.path.abspath(__file__))
    tmp_filename = os.path.join(here, 'tmpl8_regTVB.py')
    template = Template(filename=tmp_filename)
    return template

def load_atlas_data():
    """
    search for mha annotated files and add their data to 'atlas_data' 2 dim array
    """

    atlas_data = []
    atlas_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'NeuroML', 'atlas_data')
    for findex, filename in enumerate(os.listdir(atlas_dir)):
        if filename.__contains__("mha"):
            file = (os.path.join(atlas_dir, filename))
            atlas_data.insert(findex, np.genfromtxt(file, delimiter=','))

    return atlas_data

def process_atlas_data(atlas_data):
    """
    some processing of the data, in this case normalize the data
    """

    for regions, data in enumerate(atlas_data):
        datalength=(len(data))
        for regs, dat in enumerate(data):
            atlas_data[regions][regs]=atlas_data[regions][regs]/datalength

    return atlas_data

def annotate_template_data(atlas_data, model_name, model):
    """
    search for mha annotated constant names and add region specific 'atlas_data' to constants
    such to be processed in regular model generation
    """
    cnstcntr=0
    for cnstnr, cnst in enumerate(model.component_types[model_name].constants):
        if 'mha' in cnst.name:
            cnst.default=atlas_data[cnstcntr]
            cnstcntr += 1


def render_model(model_name, template=None, folder=None):
    model, svboundaries = load_model(model_name, folder)
    annotate_template_data(process_atlas_data(load_atlas_data()), model_name, model)
    template = template or default_template()
    model_str = template.render(
                            dfunname=model_name,
                            const=model.component_types[model_name].constants,
                            dynamics=model.component_types[model_name].dynamics,
                            svboundaries=svboundaries,
                            exposures=model.component_types[model_name].exposures
                            )
    return model_str


def regTVB_templating(model_filename, folder=None):
    """
    modelfile.py is placed results into tvb/simulator/models
    for new models models/__init.py__ is auto_updated if model is unfamiliar to tvb
    file_class_name is the name of the produced file and also the model's class name
    the path to XML model files folder can be added with the 2nd argument.
    example model files:
        epileptort.xml
        generic2doscillatort.xml
        kuramotot.xml
        montbriot.xml
        reducedwongwangt.xml
    """

    # file locations
    modelfile = os.path.join(os.path.dirname(tvb.__file__), 'simulator', 'models', model_filename.lower() + '.py')

    # start templating
    model_str = render_model(model_filename, template=default_template(), folder=folder)

    # write templated model to file
    with open(modelfile, "w") as f:
        f.writelines(model_str)

    # write new model to __init__.py such it is familiar to TVB if not already present
    try:
        doprint=True
        modelenumnum=0
        modulemodnum=0
        with open("{}{}".format(os.path.dirname(tvb.__file__),'/simulator/models/__init__.py'), "r+") as f:
            lines = f.readlines()
            for num, line in enumerate(lines):
                if (model_filename.upper() + ' = ' + "\"" + model_filename + "\"") in line:
                    doprint=False
                elif ("class ModelsEnum(Enum):") in line:
                    modelenumnum = num
                elif ("_module_models = {") in line:
                    modulemodnum = num
            if doprint:
                lines.insert(modelenumnum + 1, "    " + model_filename.upper() + ' = ' + "\"" + model_filename + "\"\n")
                lines.insert(modulemodnum + 2, "    " + "'" + model_filename.lower() + "'" + ': '
                             + "[ModelsEnum." + model_filename.upper() + "],\n")
                f.truncate(0)
                f.seek(0)
                f.writelines(lines)
    except IOError as e:
        logger.error('ioerror: %s', e)

if __name__ == "__main__":

    # example run for ReducedWongWang model
    regTVB_templating('MontbrioT', './NeuroML/XMLmodels/')

    # print(os.environ)
    # token = os.environ["HBP_AUTH_TOKEN"]=""
    # client = KGClient(token, "https://kg.humanbrainproject.eu/query")
    # # example = TVBtestDataset(client, '10.25493/1ECN-6SM')
    # example = TVBtestDataset(client)
    # # print(example.create_filter_params())
    # dataatje = example.fetch()

    # datadict={}
    # returneddatadict={}
    # returneddatadict = example.create_result(datadict)
    # print(vars(returneddatadict))