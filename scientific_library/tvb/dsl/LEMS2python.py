from mako.template import Template
from model.model import Model

def regTVB_templating(model_filename):
    """
    function will start generation of regular TVB models according to fp_xml
    modelfile.py is placed results into tvb/simulator/models
    for new models models/__init.py__ is auto_updated if model is unfamiliar to tvb
    file_class_name is the name of the producedfile and also the class name

    .. moduleauthor:: Michiel. A. van der Vlag <m.van.der.vlag@fz-juelich.de>
    """

    # file locations
    fp_xml = 'NeuroML/XMLmodels/' + model_filename.lower() + '.xml'
    modelfile = "../simulator/models/" + model_filename.lower() + ".py"

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
    template = Template(filename='tmpl8_regTVB.py')
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
    doprint=True
    with open("../simulator/models/__init__.py", "r+") as f:
        for line in f.readlines():
            if ("from ." + model_filename.lower() + " import " + model_filename) in line:
                doprint=False
        if doprint:
            f.writelines("\nfrom ." + model_filename.lower() + " import " + model_filename)