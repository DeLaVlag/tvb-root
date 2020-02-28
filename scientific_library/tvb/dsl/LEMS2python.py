from mako.template import Template
from model.model import Model

def regTVB_templating(target):
    """
    function will start generation of regular TVB models according to inputfile.xml
    modelfile.py is placed results into tvb model directory
    for new models models/__init.py__ is not auto_updated
    modelname is also the class name
    filename is TVB output filename

    """

    def montbrio():
        modelname = 'Theta2D'
        filename = 'montbrio'
        return modelname, filename

    def epileptor():
        modelname = 'Epileptor'
        filename = 'epileptor'
        return modelname, filename

    def oscillator():
        modelname = 'Generic2dOscillator' # is also the class name
        filename = 'oscillator' # TVB output file name
        return modelname, filename

    def wong_wang():
        modelname = 'ReducedWongWang' # is also the class name
        filename = 'wong_wang' # TVB output file name
        return modelname, filename

    def kuramoto():
        modelname = 'Kuramoto'  # is also the class name
        filename = 'kuramoto'  # TVB output file name
        return modelname, filename

    switcher = {
        'Kuramoto': kuramoto,
        'ReducedWongWang': wong_wang,
        'Generic2dOscillator': oscillator,
        'Epileptor': epileptor,
        'Montbrio': montbrio
    }

    func = switcher.get(target, 'invalid model choice')
    modelname, filename = func()
    print('\n Building and running model:', target)

    fp_xml = 'NeuroML/' + filename.lower() + '.xml'
    modelfile = "../simulator/models/" + filename.lower() + "T.py"

    model = Model()
    model.import_from_file(fp_xml)
    #modelextended = model.resolve()

    modelist = list()
    modelist.append(model.component_types[modelname])
    # print((modelist[0].dynamics.conditional_derived_variables['ctmp0'].cases[1]))

    # do some inventory. check if boundaries are set for any sv to print the boundaries section in template
    svboundaries = 0
    for i, sv in enumerate(modelist[0].dynamics.state_variables):
        if sv.boundaries != 'None' and sv.boundaries != '' and sv.boundaries:
            svboundaries = 1
            continue

    # add a T to the class name for comparison to previous models
    modelname = modelname + 'T'
    # start templating
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
    newfile=0
    if (newfile):
        with open("../simulator/models/__init__.py", "a+") as f:
            f.writelines("from ." + filename + "T import " + modelname)

if __name__ == '__main__':
    drift_templating('Generic2dOscillator')

