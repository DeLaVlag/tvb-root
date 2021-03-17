import pytest, os, glob, itertools, numpy as np, re, argparse, subprocess, pickle
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
run_path = os.path.join(framework_path, "run")
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

def simulation_args(model):
    parser = argparse.ArgumentParser(description='Run parameter sweep.')
    parser.add_argument('-c', '--n_coupling', help='num grid points for coupling parameter', default=32, type=int)
    parser.add_argument('-s', '--n_speed', help='num grid points for speed parameter', default=32, type=int)
    parser.add_argument('-n', '--n_time', help='number of time steps to do (default 400)', type=int, default=4)
    parser.add_argument('-v', '--verbose', help='increase logging verbosity', action='store_true', default='-v')
    parser.add_argument('--node_threads', default=1, type=int)
    parser.add_argument('--model',
                        help="neural mass model to be used during the simulation",
                        default=model)
    parser.add_argument('--filename', default=model+".c", type=str,
                        help="Filename to use as GPU kernel definition")
    parser.add_argument('--stts', default="1", type=int, help="Number of states of model")
    parser.add_argument('--tvbn', default="68", type=str, help="Number of tvb nodes")

    return parser.parse_args()
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
            mod_content = f.read().replace('M_PI_F', '%ff' % (np.pi,))

            # Compile model
            mod = SourceModule(mod_content, options=compiler_opts(), include_dirs=[], no_extern_c=True, keep=False)
            assert mod is not None
            compiled = True

        assert compiled

    @pytest.mark.slow
    def test_simulation_cuda_models(self):

        model = "kuramoto"
        n_coupling = 32
        n_speed = 32
        n_steps = 4
        total_data = n_coupling * n_speed
        path = os.path.join(run_path, "__main__.py")
        cmd = "python " + path + " --model " + model + " -c " + str(n_coupling) + " -s " + str(n_speed) + " -n " + str(
            n_steps) + " --tvbn 68 --stts 1"
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   universal_newlines=True)
        out, err = process.communicate()

        # Warnings are in sterr
        print(err, '\n\n', out)

        # Reading the simulation data
        tavg_file = open(os.path.join(run_path, 'tavg_data'), 'rb')
        tavg_data = pickle.load(tavg_file)
        tavg_file.close()
        b, c, d = tavg_data.shape
        print(tavg_data.shape)
        assert d == total_data

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