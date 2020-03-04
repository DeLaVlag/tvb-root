import LEMS2python as templating
import matplotlib.pyplot as plt
import time
from numpy import corrcoef

model_target = ["Epileptor", "Generic2dOscillator", "Kuramoto", "ReducedWongWang", "Montbrio"]
for i, trgt in enumerate(model_target):

	def montbrio():
		modelname = 'Theta2D'
		filename = 'montbrio'
		return modelname, filename


	def epileptor():
		modelname = 'Epileptor'
		filename = 'epileptor'
		return modelname, filename


	def oscillator():
		modelname = 'Generic2dOscillator'  # is also the class name
		filename = 'oscillator'  # TVB output file name
		return modelname, filename


	def wong_wang():
		modelname = 'ReducedWongWang'  # is also the class name
		filename = 'wong_wang'  # TVB output file name
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

	func = switcher.get(trgt, 'invalid model choice')
	modelname, filename = func()
	print('\n Building and running model:', trgt)

	# make a model template
	templating.regTVB_templating(trgt)

	import TVB_testsuite.tvbRegCudaNumba as TemplSim
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
