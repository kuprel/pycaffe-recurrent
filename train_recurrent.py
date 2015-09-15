import caffe, numpy, string, os, shutil, cPickle as pickle

save_pkl = lambda f, obj: pickle.dump(obj, open(f, 'wb'), protocol=-1)
load_pkl = lambda f: pickle.load(open(f, 'rb'))

sf = lambda *x: string.join([str(i) for i in x], '_')

solver = caffe.get_solver('solver.prototxt')

X = numpy.load('X.npy')
Y = numpy.load('Y.npy')
m, T, b = X.shape
L = max([int(i.split('_')[-1]) 
		 for i in solver.net.blobs.keys()
		 if 'h_{}_'.format(T) in i]) + 1

param_corresp = [(sf('fc',l), sf('fc',0,l)) for l in range(L+1)]

if os.path.isdir('params'): shutil.rmtree('params')
os.makedirs('params')

step_num = 10
epoch = 1

while True:

	print 'epoch {}'.format(epoch)

	for i in range(m//2):

		# Copy previous final state to current initial state
		for l in range(L):
			state_i = solver.net.blobs[sf('h',0,l)].data
			state_f = solver.net.blobs[sf('h',1,l)].data
			state_i[...] = state_f

		# Insert data
		for t in range(T):
			xt = solver.net.blobs[sf('x',t)].data
			yt = solver.net.blobs[sf('y',t)].data
			if i>0: xt[range(b), X[i-1,t]] = 0
			xt[range(b), X[i,t]] = 1
			yt[...] = Y[i,t]

		solver.step(step_num)

		# Test Net
		if solver.iter%10 == 0:

			# Save params
			params = {ki: [pr.data for pr in solver.net.params[kj]] 
					  for ki, kj in param_corresp}
			save_pkl('params/iter%08d.pkl'%solver.iter, params)

			# Insert data
			for t in range(T):
				xt = solver.test_nets[0].blobs[sf('x',t)].data
				yt = solver.test_nets[0].blobs[sf('y',t)].data
				if i>0: xt[range(b), X[i-1+m//2,t]] = 0
				xt[range(b), X[i+m//2,t]] = 1
				yt[...] = Y[i+m//2,t]
			
			solver.test_nets[0].forward()

			# Compute loss
			loss = lambda t: solver.test_nets[0].blobs[sf('loss',t)].data
			loss = numpy.mean([loss(t) for t in range(T)])	
			print 'test loss:  {}, iter {}'.format(loss, solver.iter)

	# Reset inputs
	for t in range(T):
		solver.test_nets[0].blobs[sf('x',t)].data[...] = 0
		solver.net.blobs[sf('x',t)].data[...] = 0

	# Reset state
	for l in range(L):
		solver.net.blobs[sf('h',0,l)].data[...] = 0

	epoch += 1
	step_num = max(1, step_num/2)