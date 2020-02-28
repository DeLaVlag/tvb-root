import LEMS2python as templating
import TVB_testsuite.tvbRegCudaNumba as TemplSim
import matplotlib.pyplot as plt
import time
from numpy import corrcoef

# model_target = ["Generic2dOscillator"]
model_target = ["Epileptor", "Generic2dOscillator", "Kuramoto", "Montbrio", "ReducedWongWang"]
for i, trgt in enumerate(model_target):

	# make a model template
	templating.regTVB_templating(trgt)

	# run tvb without model template
	testTemplSim = TemplSim.TVB_test()
	tavg0=testTemplSim.startsim(trgt, tmpld=0)

	# run tvb without model template
	testTemplSim = TemplSim.TVB_test()
	tavg1=testTemplSim.startsim(trgt, tmpld=1)

	#compare output to check if same as template
	comparison = tavg0.ravel()==tavg1.ravel()
	print('Templated version is similar to original:', comparison.all())
	print('Correlation coefficient:', corrcoef(tavg0.ravel(), tavg1.ravel())[0, 1])


# plt.show()
