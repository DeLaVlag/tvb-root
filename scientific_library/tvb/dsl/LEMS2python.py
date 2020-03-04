from mako.template import Template
from model.model import Model
import os
import tvb

def regTVB_templating(model_filename):
    """
    function will start generation of regular TVB models according to fp_xml
    modelfile.py is placed results into tvb/simulator/models
    for new models models/__init.py__ is auto_updated if model is unfamiliar to tvb
    file_class_name is the name of the producedfile and also the class name
    """

    fp_xml = "{}{}{}{}".format(os.path.dirname(tvb.__file__),'/dsl/NeuroML/XMLmodels/',model_filename.lower(),'.xml')
    modelfile = "{}{}{}{}".format(os.path.dirname(tvb.__file__),'/simulator/models/',model_filename.lower(),'.py')

    fp_xml = 'NeuroML/XMLmodels/' + model_filename.lower() + '.xml'
    modelfile = "../simulator/models/" + model_filename.lower() + "T.py"

    model = Model()
    model.import_from_file(fp_xml)

    modelist = list()
    modelist.append(model.component_types[model_filename])

    # do some inventory. check if boundaries are set for any sv to print the boundaries section in template
    svboundaries = 0
    for i, sv in enumerate(modelist[0].dynamics.state_variables):
        if sv.boundaries != 'None' and sv.boundaries != '' and sv.boundaries:
            svboundaries = 1
            continue

    # add a T to the class name to not overwrite existing models
    # start templating
    modelname=model_filename+'T'
    template = Template(filename='tmpl8_regTVB.py')
    model_str = template.render(
                            dfunname=modelname,
                            const=modelist[0].constants,
                            dynamics=modelist[0].dynamics,
                            svboundaries=svboundaries,
                            exposures=modelist[0].exposures
                            )
    # write template to file
    with open(modelfile, "w") as f:
        f.writelines(model_str)

    # write new model to init.py such it is familiar to TVB
    # doprint=1
    # with open("../simulator/models/__init__.py", "r+") as f:
    #     for line in f.readlines():
    #         if ("from ." + filename.lower() + "T import " + modelname) in line:
    #             doprint=0
    #     if doprint:
    #         f.writelines("\nfrom ." + filename.lower() + "T import " + modelname)

    # write new model to init.py such it is familiar to TVB if not already present
    # try:
    doprint=True
    modelenumnum=0
    modulemodnum=0
    with open("{}{}".format(os.path.dirname(tvb.__file__),'/simulator/models/__init__.py'), "r+") as f:
        lines = f.readlines()
        for num, line in enumerate(lines):
            if (model_filename.upper() + 'T = ' + "\"" + model_filename + "T\"") in line:
                doprint=False
            elif ("class ModelsEnum(Enum):") in line:
                modelenumnum = num
            elif ("_module_models = {") in line:
                modulemodnum = num
        if doprint:
            lines.insert(modelenumnum + 1, "    " + model_filename.upper() + 'T = ' + "\"" + model_filename + "T\"\n")
            lines.insert(modulemodnum + 2, "    " + "'" + model_filename.lower() + "T'" + ': '
                         + "[ModelsEnum." + model_filename.upper() + "T],\n")
            f.truncate(0)
            f.seek(0)
            f.writelines(lines)
    # except:
    #     print('unable to add new model to __init__.py')

if __name__ == '__main__':
    drift_templating('Generic2dOscillator')

