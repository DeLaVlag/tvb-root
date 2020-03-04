from mako.template import Template
import tvb
import sys
import os
sys.path.append("{}{}".format(os.path.dirname(tvb.__file__),"/dsl/NeuroML/lems"))
from model.model import Model

def regTVB_templating(model_filename):
    """
    modelfile.py is placed results into tvb/simulator/models
    for new models models/__init.py__ is auto_updated if model is unfamiliar to tvb
    file_class_name is the name of the producedfile and also the class name

    .. moduleauthor:: Michiel. A. van der Vlag <m.van.der.vlag@fz-juelich.de>
    """

    # file locations
    fp_xml = "{}{}{}{}".format(os.path.dirname(tvb.__file__),'/dsl/NeuroML/XMLmodels/',model_filename.lower(),'.xml')
    print('fp_xml:', fp_xml)
    modelfile = "{}{}{}{}".format(os.path.dirname(tvb.__file__),'/simulator/models/',model_filename.lower(),'.py')

    # instantiate LEMS lib
    model = Model()
    model.import_from_file(fp_xml)

    # do some inventory. check if boundaries are set for any sv to print the boundaries section in template
    svboundaries = 0
    for i, sv in enumerate(model.component_types[model_filename].dynamics.state_variables):
        if sv.boundaries != 'None' and sv.boundaries != '' and sv.boundaries:
            svboundaries = 1
            continue

    # start templating
    print("{}{}".format(os.path.dirname(tvb.__file__),'/dsl/tmpl8_regTVB.py'))
    template = Template(filename="{}{}".format(os.path.dirname(tvb.__file__),'/dsl/tmpl8_regTVB.py'))
    model_str = template.render(
                            dfunname=model_filename,
                            const=model.component_types[model_filename].constants,
                            dynamics=model.component_types[model_filename].dynamics,
                            svboundaries=svboundaries,
                            exposures=model.component_types[model_filename].exposures
                            )
    # write templated model to file
    with open(modelfile, "w") as f:
        f.writelines(model_str)

    # write new model to init.py such it is familiar to TVB if not already present
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
    except:
        print('unable to add new model to __init__.py')