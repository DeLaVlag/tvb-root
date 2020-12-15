#!/usr/bin/env python3

from __future__ import print_function
import sys
import numpy as np
import os.path
import numpy as np
import itertools
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import pytools
import time
import argparse
import logging

np.set_printoptions(threshold=sys.maxsize)

here = os.path.dirname(os.path.abspath(__file__))

class Parsweep:

    # def load_connectome(dataset):#{{{
    #     # load connectome & normalize
    #     if dataset == 'hcp':
    #         npz = np.load('hcp-100.npz')
    #         weights = npz['weights'][0].astype(np.float32)
    #         lengths = npz['lengths'][0].astype(np.float32)
    #     elif dataset == 'sep':
    #         npz = np.load('sep.npz')
    #         weights = npz['weights'].astype(np.float32)
    #         lengths = npz['lengths'].astype(np.float32)
    #     else:
    #         raise ValueError('unknown dataset name %r' % (dataset, ))
    #     # weights /= {'N':2e3, 'Nfa': 1e3, 'FA': 1.0}[mattype]
    #     weights /= weights.max()
    #     assert (weights <= 1.0).all()
    #     return weights, lengths#}}}

    # def expand_params(couplings, speeds):#{{{
    #     ns = speeds.size
    #     nc = couplings.size
    #     params = itertools.product(speeds, couplings)
    #     params_matrix = np.array([vals for vals in params])
    #     return params_matrix#}}}
    #
    # def setup_params(nc, ns):#{{{
    #     # the correctness checks at the end of the simulation
    #     # are matched to these parameter values, for the moment
    #     couplings = np.logspace(1.6, 3.0, nc)
    #     speeds = np.logspace(0.0, 2.0, ns)
    #     return couplings, speeds#}}}

    def make_kernel(self, source_file, warp_size, block_dim_x, args, ext_options='', #{{{
            caching='none', lineinfo=False, nh='nh', model='kuromoto'):
        with open(source_file, 'r') as fd:
            source = fd.read()
            source = source.replace('M_PI_F', '%ff' % (np.pi, ))
            opts = ['--ptxas-options=-v', ]# '-maxrregcount=32']# '-lineinfo']
            if lineinfo:
                opts.append('-lineinfo')
            opts.append('-DWARP_SIZE=%d' % (warp_size, ))
            opts.append('-DBLOCK_DIM_X=%d' % (block_dim_x, ))
            opts.append('-DNH=%s' % (nh, ))
            if ext_options:
                opts.append(ext_options)
            cache_opt = {
                'none': None,
                'shuffle': '-DCACHING_SHUFFLE',
                'shared': '-DCACHING_SHARED',
                'shared_sync': '-DCACHING_SHARED_SYNC',
            }[caching]
            if cache_opt:
                opts.append(cache_opt)
            idirs = [here]
            # logger.info('nvcc options %r', opts)
            network_module = SourceModule(
                    source, options=opts, include_dirs=idirs,
                    no_extern_c=True,
                    keep=False,
            )
            # mod_func = '_Z9EpileptorjjjjjffPfS_S_S_S_'
            # mod_func = '_Z8KuramotojjjjjffPfS_S_S_S_'
            # mod_func = '_Z9RwongwangjjjjjffPfS_S_S_S_'
            # mod_func = '_Z12KuratmotorefjjjjjffPfS_S_S_S_'
            mod_func = "{}{}{}{}".format('_Z', len(args.model), args.model, 'jjjjjffPfS_S_S_S_')
            step_fn = network_module.get_function(mod_func)

        # nvcc the bold model kernel
        with open('balloon.c', 'r') as fd:
            source = fd.read()
            opts = []
            opts.append('-DWARP_SIZE=%d' % (warp_size, ))
            opts.append('-DBLOCK_DIM_X=%d' % (block_dim_x, ))
            bold_module = SourceModule(source, options=opts)
            bold_fn = bold_module.get_function('bold_update')
        with open('covar.c', 'r') as fd:
            source = fd.read()
            opts = ['-ftz=true']  # for faster rsqrtf in corr
            opts.append('-DWARP_SIZE=%d' % (warp_size, ))
            opts.append('-DBLOCK_DIM_X=%d' % (block_dim_x, ))
            covar_module = SourceModule(source, options=opts)
            covar_fn = covar_module.get_function('update_cov')
            cov_corr_fn = covar_module.get_function('cov_to_corr')
        return step_fn, bold_fn, covar_fn, cov_corr_fn #}}}

    def cf(self, array):#{{{
        # coerce possibly mixed-stride, double precision array to C-order single precision
        return array.astype(dtype='f', order='C', copy=True)#}}}

    def nbytes(self, data):#{{{
        # count total bytes used in all data arrays
        nbytes = 0
        for name, array in data.items():
            nbytes += array.nbytes
        return nbytes#}}}

    def make_gpu_data(self, data):#{{{
        # put data onto gpu
        gpu_data = {}
        for name, array in data.items():
            gpu_data[name] = gpuarray.to_gpu(self.cf(array))
        return gpu_data#}}}

    def gpu_info(self):
        cmd = "nvidia-smi -q -d MEMORY,UTILIZATION"
        returned_value = os.system(cmd)  # returns the exit code in unix
        print('returned value:', returned_value)

    def run_simulation(self, weights, lengths, params_matrix, speeds, logger, args, n_nodes, n_work_items, n_params, nstep,
                       n_inner_steps,
                       buf_len, states, dt, min_speed):

        logger.info('caching strategy %r', args.caching)
        if args.test and args.n_time % 200:
            logger.warning('rerun w/ a multiple of 200 time steps (-n 200, -n 400, etc) for testing') #}}}

        # setup data#{{{
        data = { 'weights': weights, 'lengths': lengths, 'params': params_matrix.T }
        base_shape = n_work_items,
        for name, shape in dict(
                tavg0=(n_nodes,),
                tavg1=(n_nodes,),
                state=(buf_len, states * n_nodes),
                bold_state=(4, n_nodes),
                bold=(n_nodes, ),
                covar_means=(2 * n_nodes, ),
                covar_cov=(n_nodes, n_nodes, ),
                corr=(n_nodes, n_nodes, ),
                ).items():
            data[name] = np.zeros(shape + base_shape, 'f')
        data['bold_state'][1:] = 1.0#}}}

        # logger.info(data['bold_state'])
        logger.info('bold_state.shape %r', data['bold_state'].shape)

        gpu_data = self.make_gpu_data(data)#{{{
        logger.info('history shape %r', data['state'].shape)
        logger.info('on device mem: %.3f MiB' % (self.nbytes(data) / 1024 / 1024, ))#}}}

        # setup CUDA stuff#{{{
        step_fn, bold_fn, covar_fn, cov_corr_fn = self.make_kernel(
                source_file=args.filename,
                warp_size=32,
                block_dim_x=args.n_coupling,
                args=args,
                ext_options='-DRAND123',
                caching=args.caching,
                lineinfo=args.lineinfo,
                nh=buf_len,
                model=args.model,
                )#}}}

        # setup simulation#{{{
        tic = time.time()
        streams = [drv.Stream() for i in range(32)]
        events = [drv.Event() for i in range(32)]
        tavg_unpinned = []
        bold_unpinned = []
        tavg = drv.pagelocked_zeros((32, ) + data['tavg0'].shape, dtype=np.float32)
        bold = drv.pagelocked_zeros((32, ) + data['bold'].shape, dtype=np.float32)
        #}}}

        gridx = args.n_coupling // args.blockszx
        if (gridx == 0):
            gridx = 1;
        gridy = args.n_speed // args.blockszy
        if (gridy == 0):
            gridy = 1;
        final_block_dim = args.blockszx, args.blockszy, 1
        final_grid_dim = gridx, gridy

        # logger.info('final block dim %r', final_block_dim)
        logger.info('final grid dim %r', final_grid_dim)

        # run simulation#{{{
        logger.info('submitting work')
        import tqdm
        for i in tqdm.trange(nstep):

            event = events[i % 32]
            stream = streams[i % 32]

            stream.wait_for_event(events[(i - 1) % 32])

            step_fn(np.uintc(i * n_inner_steps), np.uintc(n_nodes), np.uintc(buf_len), np.uintc(n_inner_steps),
                    np.uintc(n_params), np.float32(dt), np.float32(min_speed),
                    gpu_data['weights'], gpu_data['lengths'], gpu_data['params'], gpu_data['state'],
                    gpu_data['tavg%d' % (i%2,)],
                    block=final_block_dim,
                    grid=final_grid_dim,
                    stream=stream)

            event.record(streams[i % 32])

            # TODO check next integrate not zeroing current tavg?
            tavgk = 'tavg%d' % ((i + 1) % 2, )
            bold_fn(np.uintc(n_nodes),
                    # BOLD model dt is in s, requires 1e-3
                    np.float32(dt * n_inner_steps * 1e-3),
                    gpu_data['bold_state'], gpu_data[tavgk], gpu_data['bold'],
                    # block=(couplings.size, 1, 1), grid=(speeds.size, 1), stream=stream)
                    # block=(args.n_coupling, 1, 1), grid=(speeds.size, 1), stream=stream)
                    block = final_block_dim, grid = final_grid_dim, stream = stream)

            if i >= (nstep // 2):
                i_time = i - nstep // 2
                covar_fn(np.uintc(i_time), np.uintc(n_nodes),
                    gpu_data['covar_cov'], gpu_data['covar_means'], gpu_data[tavgk],
                    block=final_block_dim, grid=final_grid_dim, stream=stream)

            # async wrt. other streams & host, but not this stream.
            if i >= 32:
                stream.synchronize()
                tavg_unpinned.append(tavg[i % 32].copy())
                bold_unpinned.append(bold[i % 32].copy())

            drv.memcpy_dtoh_async(tavg[i % 32], gpu_data[tavgk].ptr, stream=stream)
            drv.memcpy_dtoh_async(bold[i % 32], gpu_data['bold'].ptr, stream=stream)

            if i == (nstep - 1):
                cov_corr_fn(np.uintc(nstep // 2), np.uintc(n_nodes),
                        gpu_data['covar_cov'], gpu_data['corr'],
                        # block=(couplings.size, 1, 1), grid=(speeds.size, 1), stream=stream)
                        block = final_block_dim, grid = final_grid_dim, stream = stream)

        logger.info('waiting for work to finish..')

        # recover uncopied data from pinned buffer
        if nstep > 32:
            for i in range(nstep % 32, 32):
                stream.synchronize()
                tavg_unpinned.append(tavg[i].copy())
                bold_unpinned.append(bold[i].copy())

        for i in range(nstep % 32):
            stream.synchronize()
            tavg_unpinned.append(tavg[i].copy())
            bold_unpinned.append(bold[i].copy())

        corr = gpu_data['corr'].get()
        # logger.info('corr', corr)
        print('corr', corr)

        elapsed = time.time() - tic
        # release pinned memory
        tavg = np.array(tavg_unpinned)
        bold = np.array(bold_unpinned)
        # inform about time
        logger.info('elapsed time %0.3f', elapsed)
        logger.info('%0.3f M step/s', 1e-6 * nstep * n_inner_steps * n_work_items / elapsed)#}}}

        # check results (for smaller sizes)#{{{
        if args.test:
            r, c = np.triu_indices(n_nodes, 1)
            win_size = 200 # 2s
            win_tavg = tavg.reshape((-1, win_size) + tavg.shape[1:])
            err = np.zeros((len(win_tavg), n_work_items))
            # TODO do cov/corr in kernel
            for i, tavg_ in enumerate(win_tavg):
                for j in range(n_work_items):
                    fc = np.corrcoef(tavg_[:, :, j].T)
                    err[i, j] = ((fc[r, c] - weights[r, c])**2).sum()
            # look at 2nd 2s window (converges quickly)
            err_ = err[-1].reshape((speeds.size, couplings.size))
            # change on fc-sc metric wrt. speed & coupling strength
            derr_speed = np.diff(err_.mean(axis=1)).sum()
            derr_coupl = np.diff(err_.mean(axis=0)).sum()
            logger.info('derr_speed=%f, derr_coupl=%f', derr_speed, derr_coupl)
            logger.info('Finished OK')
            #}}}

    # vim: sw=4 sts=4 ts=4 et ai
