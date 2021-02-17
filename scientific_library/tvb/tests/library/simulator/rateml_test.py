# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and 
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2020, Baycrest Centre for Geriatric Care ("Baycrest") and others
#
# This program is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE.  See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this
# program.  If not, see <http://www.gnu.org/licenses/>.
#
#
#   CITATION:
# When using The Virtual Brain for scientific publications, please cite it as follows:
#
#   Paula Sanz Leon, Stuart A. Knock, M. Marmaduke Woodman, Lia Domide,
#   Jochen Mersmann, Anthony R. McIntosh, Viktor Jirsa (2013)
#       The Virtual Brain: a simulator of primate brain network dynamics.
#   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)
#
#

"""
Test for tvb.simulator.noise module

.. moduleauthor:: Aarón Pérez Martín <a.perez.martin@fz-juelich.de>

"""
from tvb.tests.library.base_testcase import BaseTestCase
import os, glob, pytest
from tvb.rateML import XML2model

from tvb.rateML.XML2model import RateML, Utils


framework_path, _ = os.path.split(XML2model.__file__)
XMLModel_path = os.path.join(framework_path, "XMLmodels")
generatedModels_path = "/home/aaron/tvb_temporal_folder" #os.path.join(framework_path, "generatedModels")


def load_one_model(keyword, language):
    if Utils.model_exist(keyword, filenames):
        rateml = RateML(model_filename=keyword,
                                  language=language,
                                  XMLfolder=XMLModel_path,
                                  GENfolder=generatedModels_path)
        model, _, _, _, _, error = rateml.load_model()
        if len(error) > 0:
            print("\n", error)
        return model, error

def convert_one_model(keyword, language):
    rateml = RateML(model_filename=keyword,
                              language=language,
                              XMLfolder=XMLModel_path,
                              GENfolder=generatedModels_path)
    finished, validation = rateml.transform()
    if len(validation)>0:
        print("\n",validation)
    return finished, validation

#######

list, filenames = Utils.load_models(location=XMLModel_path, extension=".xml")

def test_locate_models():
    assert len(list) > 0

def test_XMLmodel_validation():
    keyword = "oscillator"
    if Utils.model_exist(keyword, filenames):
        rateml = RateML(model_filename=keyword,
                        language="cuda",
                        XMLfolder=XMLModel_path,
                        GENfolder=generatedModels_path)
        validated, error = rateml.XSD_validate_XML()
        assert validated == True and len(error) == 0

#Preprocesing step
def test_load_and_preprocess_model():
    model, error = load_one_model(keyword="oscillator", language="cuda")
    assert len(error) == 0
    assert model != None

def test_convert_model():
    finished, validation = convert_one_model("oscillator", "python")
    assert finished == True and len(validation) == 0

    finished, validation = convert_one_model("oscillator", "cuda")
    assert finished == True and len(validation) == 0


class TestProcessModels:
    models=["epileptor"]


