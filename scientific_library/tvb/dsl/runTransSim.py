
import sys, os
# sys.path.append('/home/michiel/Documents/TVB/tvb-root/scientific_library/')
# print(sys.path)
import LEMS2python as templating
import TVB_testsuite.tvbRegCudaNumba as TemplSim
import matplotlib.pyplot as plt
import time

# # options for target:
# Kuramoto
# ReducedWongWang
# Generic2dOscillator
# Epileptor
# Montbrio

target="Kuramoto"
# make a model template
# templating.drift_templating(target)
# run tvb with model template
testTemplSim = TemplSim.TVB_test()
testTemplSim.startsim(target)

# target="KuramotoT"
# # make a model template
# # templating.drift_templating(target)
# # run tvb with model template
# testTemplSim = TemplSim.TVB_test()
# testTemplSim.startsim(target)

# plt.show()

# # templating
# model_target_tmpl8 = ["Kuramoto", "ReducedWongWang", "Generic2dOscillator"]
# # model_target_tmpl8 = ["Montbrio", "MontbrioT"]
# for i, trgt in enumerate(model_target_tmpl8):
#     # make a model template
#     # templating.drift_templating(trgt)
# #     time.sleep(2)
#     # run tvb with model template
#     print('Simming:', trgt)
#     testTemplSim = TemplSim.TVB_test()
#     testTemplSim.startsim(trgt)

# comparing templated version against regular version via linear correlation between tavgdata after TVB simulation
# model_target = [["Montbrio", "MontbrioT"], ["Epileptor", "EpileptorT"], ["Kuramoto", "KuramotoT"],
# 				["ReducedWongWang", "ReducedWongWangT"], ["Generic2dOscillator", "Generic2dOscillatorT"]]
# model_target = [["Montbrio", "MontbrioT"]]
# for i, trgtduo in enumerate(model_target):
# 	for j, trgt in enumerate(trgtduo):
# 		# run tvb with model template
# 		testTemplSim = TemplSim.TVB_test()
# 		testTemplSim.startsim(trgt)
