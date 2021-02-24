import pytest, os, glob, itertools, numpy as np
from tvb.tests.library.base_testcase import BaseTestCase
from tvb.rateML import XML2model
from tvb.rateML.XML2model import RateML, Utils

import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule


framework_path, _ = os.path.split(XML2model.__file__)
XMLModel_path = os.path.join(framework_path, "XMLmodels")
generatedModels_path = os.path.join(framework_path, "generatedModels")

def compiler_opts():
    opts = ['--ptxas-options=-v', '-maxrregcount=32', '-lineinfo']
    opts.append('-lineinfo')
    opts.append('-DWARP_SIZE=%d' % (32,))
    opts.append('-DBLOCK_DIM_X=%d' % (32,))
    opts.append('-DNH=%s' % ('nh',))
#######

class TestRateML():
    models=["epileptor", "kuramoto", "montbrio", "oscillator", "rwongwang"]
    languages = ["python", "cuda"]

    @pytest.mark.slow
    @pytest.mark.parametrize('model,language', itertools.product(models, languages))
    def test_XMLmodel_validation(self, model, language):
        rateml = RateML(model_filename=model,
                        language=language,
                        XMLfolder=XMLModel_path,
                        GENfolder=generatedModels_path)
        validated, error = rateml.XSD_validate_XML()
        assert validated == True and len(error) == 0

    @pytest.mark.slow
    @pytest.mark.parametrize('model,language', itertools.product(models, languages))
    def test_load_and_preprocess_model(self, model, language):
        rateml = RateML(model_filename=model,
                        language=language,
                        XMLfolder=XMLModel_path,
                        GENfolder=generatedModels_path)
        model, _, _, _, _, error = rateml.load_model()
        if len(error) > 0:
            print("\n", error)
        assert len(error) == 0 and model != None

    @pytest.mark.slow
    @pytest.mark.parametrize('model,language', itertools.product(models, languages))
    def test_convert_model(self, model, language):
        rateml = RateML(model_filename=model,
                        language=language,
                        XMLfolder=XMLModel_path,
                        GENfolder=generatedModels_path)
        finished, validation = rateml.transform()
        if len(validation) > 0:
            print("\n", validation)
        assert finished == True and len(validation) == 0

    @pytest.mark.slow
    @pytest.mark.parametrize('model', models)
    def test_compile_cuda_models(self, model):
        source_file = os.path.join(generatedModels_path, model + ".c")
        compiled = False
        with open(source_file, 'r') as f:
            mod_content = f.read().replace('\n', ' ')
            mod_content = mod_content.replace('M_PI_F', '%ff' % (np.pi,))

            idirs = [os.path.dirname(os.path.abspath(__file__))]
            mod = SourceModule(mod_content, options=compiler_opts(), include_dirs=idirs, no_extern_c=True, keep=False)
            assert mod is not None
            compiled = True

        assert compiled

