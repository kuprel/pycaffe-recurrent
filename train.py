import caffe, numpy, string, os, shutil, h5py, json
sf = lambda *x: string.join([str(i) for i in x], '_')

# Load hyperparameters
hypes = json.load(open('hypes.json'))

T = hypes['sequence_length']
L = hypes['layers_num']
d = hypes['state_dim']
b = hypes['batch_size']

# Copy data to memory from disk
data_disk = h5py.File('data.h5', 'r')
data = {tt: {xy: data_disk[tt][xy].value for xy in ['X', 'Y']}
		for tt in ['train', 'test']}
data_disk.close()

# Initialize solver
solver = caffe.get_solver('solver.prototxt')

# Create params directory
if os.path.isdir('params'): shutil.rmtree('params')
os.makedirs('params')

def copy_state(net):
	for l in range(L):
		state_i = net.blobs[sf('h',0,l)].data
		state_f = net.blobs[sf('h',T,l)].data
		state_i[...] = state_f

def insert_data(net, X, Y):
	for t in range(T):
		net.blobs[sf('x',t)].data[...] = 0
		net.blobs[sf('x',t)].data[range(b), X[t]] = 1
		net.blobs[sf('y',t)].data[...] = Y[t]

def save_params(net, params_file):
	param_corresp = [(sf('fc',l), sf('fc',0,l)) 
						 for l in range(L+1)]
	params = h5py.File(params_file, 'w')
	for ki, kj in param_corresp:
		pr = solver.net.params[kj]
		params.create_group(ki)
		params[ki]['W'] = pr[0].data
		params[ki]['b'] = pr[1].data

def compute_loss(net):
	loss = lambda t: net.blobs[sf('loss',t)].data
	loss = numpy.mean([loss(t) for t in range(T)])
	return loss

step_num = 5
test_iter = 5
epoch, epoch_test = 1, 1

# Test and train iters
i, j = 0, 0

while True:

	net = solver.net
	copy_state(net)
	X = data['train']['X'][i]
	Y = data['train']['Y'][i]
	insert_data(net, X, Y)
	solver.step(step_num)

	if solver.iter%test_iter == 0:

		net = solver.test_nets[0]
		copy_state(net)
		X = data['test']['X'][j]
		Y = data['test']['Y'][j]
		insert_data(net, X, Y)
		net.forward()

		loss = compute_loss(net)
		print 'test loss: {}, iter {}'.format(loss, solver.iter)

		params_file = 'params/iter%08d.h5'%solver.iter
		save_params(net, params_file)

		j += 1
		if j == len(data['test']['X']):
			epoch_test += 1
			print 'Test Epoch {}'.format(epoch_test)
			j = 0

	i += 1
	if i == len(data['train']['X']):
		epoch += 1
		print 'Epoch {}'.format(epoch)
		i = 0
		step_num = max(1, step_num/2)