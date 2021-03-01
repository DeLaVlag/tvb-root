import pytest, os, glob, itertools, numpy as np, re
from tvb.rateML.run.__main__ import TVB_test
from tvb.rateML.run.cuda_run import CudaRun
from tvb.tests.library.base_testcase import BaseTestCase
from tvb.rateML import XML2model
from tvb.rateML.XML2model import RateML, Utils
from pathlib import Path

import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

from tvb.rateML.run import __main__
from cuda_run import CudaRun


framework_path, _ = os.path.split(XML2model.__file__)
XMLModel_path = os.path.join(framework_path, "XMLmodels")
generatedModels_path = os.path.join(framework_path, "generatedModels")
#dic_regex_mincount = {r'^__global':1, r'^__device':1}
dic_regex_mincount = {r'^__global':1,
                      r'^__device':3,
                      r'^__device__ float wrap_it_':2,
                      r'state\(\(\(':1,
                      r'state\(\(t':2,
                      r'tavg\(':1,
                      r'= params\(\d\)\;$':2}


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
            mod_content = mod_content.replace('M_PI_F', '%ff' % (np.pi,))

            # Compile model
            mod = SourceModule(mod_content, options=compiler_opts(), include_dirs=[], no_extern_c=True, keep=False)
            assert mod is not None
            compiled = True

        assert compiled

    @pytest.mark.slow
    @pytest.mark.parametrize('model', models)
    def test_simulation_cuda_models(self, model):
        source_file = os.path.join(generatedModels_path, model + ".c")
        with open(source_file, 'r') as f:
            #mod_content = mod_content.replace('M_PI_F', '%ff' % (np.pi,))
            #Compile model
            #mod = SourceModule(mod_content, options=compiler_opts(), include_dirs=[], no_extern_c=True, keep=False)
            #mod_func = "{}{}{}{}".format('_Z', len(model), model, 'jjjjjffPfS_S_S_S_')

            #execute the function into the GPU
            #func = mod.get_function(mod_func)
            ###########

            logging.basicConfig(level=logging.DEBUG if self.args.verbose else logging.INFO)
            logger = logging.getLogger('[TVB_CUDA]')

            cudarun = CudaRun()

            tester = TVB_test()

            tavg_data = cudarun.run_simulation(tester.weights, tester.lengths, tester.params, tester.speeds, logger,
                                               tester.args, tester.n_nodes, tester.n_work_items, tester.n_params, tester.nstep,
                                               tester.n_inner_steps, tester.buf_len, tester.states, tester.dt, tester.min_speed)
            print(vars(tavg_data))

    @pytest.mark.slow
    @pytest.mark.parametrize('model', models)
    def test_contentcheck_cuda_models(self, model):

        source_file = os.path.join(generatedModels_path, model + ".c")
        with open(source_file, "r") as f:
            lines = f.read()

            # trucate the file to avoid processing bold_update
            lines = lines.split("__global__ void bold_update")[0]

            pattern_model = r'^__global__ void ' + model + '\('
            if len(re.findall(pattern=pattern_model, string=lines, flags=re.IGNORECASE + re.MULTILINE + re.DOTALL)) <= 0:
                print("Error", pattern_model, "did not found", model)
                assert False
            else:
                assert True

            for regex, mincount in dic_regex_mincount.items():
                matches = re.findall(pattern=regex, string=lines, flags=re.IGNORECASE + re.MULTILINE + re.DOTALL)

                if len(matches) < dic_regex_mincount[regex]:
                    print("Error", regex, "found", len(matches), "of", mincount)
                    assert False
                else:
                    assert True