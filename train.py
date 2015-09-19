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
nets = {
	'train': solver.net,
	'test': solver.test_nets[0]
}

# Create params directory
if os.path.isdir('params'): shutil.rmtree('params')
os.makedirs('params')

def copy_state(net):
	"""
	Copies previous final state to current initial state
	"""
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

def update_iter(itr, epoch, tt):
	"""
	Increments iter, checks for new epoch, 
	resets state to zero if new epoch
	"""
	itr += 1
	new_epoch = False
	if itr == len(data[tt]['X']):
		new_epoch = True
		epoch += 1
		itr = 0
		for l in range(L):
			nets[tt].blobs[sf('h',0,l)].data[...] = 0
		print '{} epoch {}'.format(tt, epoch)
	return itr, epoch, new_epoch

def initialize(tt):
	copy_state(nets[tt])
	X = data[tt]['X'][i]
	Y = data[tt]['Y'][i]
	insert_data(nets[tt], X, Y)

step_num = 5
test_iter = 5
epoch_train, epoch_test = 1, 1

# Test and train iters
i, j = 0, 0

while True:

	initialize('train')
	solver.step(step_num)
	i, epoch_train, new_epoch = update_iter(i, epoch_train, 'train')
	if new_epoch: step_num = max(1, step_num/2)

	if solver.iter%test_iter == 0:

		initialize('test')
		nets['test'].forward()

		loss = compute_loss(nets['test'])
		print 'test loss: {}, iter {}'.format(loss, solver.iter)

		params_file = 'params/iter%08d.h5'%solver.iter
		save_params(nets['test'], params_file)

		j, epoch_test, new_epoch = update_iter(j, epoch_test, 'test')