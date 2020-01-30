# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
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

import requests
from tvb.core.neotraits.h5 import ViewModelH5
from tvb.interfaces.rest.client.client_decorators import handle_response
from tvb.interfaces.rest.client.main_api import MainApi
from tvb.interfaces.rest.commons import RestLink, LinkPlaceholder
from tvb.interfaces.rest.commons.dtos import DataTypeDto


class OperationApi(MainApi):
    @handle_response
    def get_operation_status(self, operation_gid, token):
        return requests.get(self.build_request_url(RestLink.OPERATION_STATUS.compute_url(True, {
            LinkPlaceholder.OPERATION_GID.value: operation_gid
        })), headers=self.get_headers(token))

    @handle_response
    def get_operations_results(self, operation_gid, token):
        response = requests.get(
            self.build_request_url(RestLink.OPERATION_RESULTS.compute_url(True, {
                LinkPlaceholder.OPERATION_GID.value: operation_gid
            })), headers=self.get_headers(token))
        return response, DataTypeDto

    @handle_response
    def launch_operation(self, project_gid, algorithm_module, algorithm_classname, view_model, temp_folder, token):
        h5_file_path = temp_folder + '/ViewModel.h5'

        h5_file = ViewModelH5(h5_file_path, view_model)
        h5_file.store(view_model)
        h5_file.close()

        file_obj = open(h5_file_path, 'rb')
        return requests.post(self.build_request_url(RestLink.LAUNCH_OPERATION.compute_url(True, {
            LinkPlaceholder.PROJECT_GID.value: project_gid,
            LinkPlaceholder.ALG_MODULE.value: algorithm_module,
            LinkPlaceholder.ALG_CLASSNAME.value: algorithm_classname
        })), files={"file": ("ViewModel.h5", file_obj)}, headers=self.get_headers(token))
