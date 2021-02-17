import pytest, os, glob, itertools
from tvb.tests.library.base_testcase import BaseTestCase
from tvb.rateML import XML2model
from tvb.rateML.XML2model import RateML, Utils


framework_path, _ = os.path.split(XML2model.__file__)
XMLModel_path = os.path.join(framework_path, "XMLmodels")
generatedModels_path = os.path.join(framework_path, "generatedModels")

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