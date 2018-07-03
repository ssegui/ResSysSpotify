from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.training.adam import AdamOptimizer
import matplotlib
matplotlib.use('Agg')

from data import *
from models import *
from args import *


logger = get_logger()

args, dir_name = parse_args()
os.makedirs(dir_name)
sess = ed.get_session()


# DATA
d = bayessian_bern_emb_data(args.in_file, args.target_emb_file, args.context_emb_file, args.ns, args.K, args.cs, dir_name, logger)
pickle.dump(d, open(dir_name + "/data.dat", "wb+"))

# MODEL
d = pickle.load(open(dir_name + "/data.dat", "rb+"))
d.batch = d.batch_generator(args.mb)
m = bayesian_emb_model(d, args.mb, sess, dir_name)


def get_n_iters():
    n_batches = len(d.playlists) / args.mb
    if len(d.playlists) % args.mb > 0:
        n_batches += 1
    return int(n_batches) * args.n_epochs, int(n_batches)


# TRAINING
sigmas_list = list()
n_iters, n_batches = get_n_iters()
logger.debug('init training number of iters '+str(n_iters)+' and batches '+str(n_batches))

m.inference.initialize(n_samples=1, n_iter=n_iters, logdir=m.logdir,
                       scale={m.y_pos: n_batches, m.y_neg: n_batches / args.ns},
                       kl_scaling={m.y_pos: n_batches, m.y_neg: n_batches / args.ns},
                       optimizer=AdamOptimizer(learning_rate=0.001)
                       )
init = tf.global_variables_initializer()
sess.run(init)
logger.debug('....starting training')
for i in range(m.inference.n_iter):
    info_dict = m.inference.update(feed_dict=d.feed(args.mb, m.target_placeholder,
                                                    m.context_placeholder,
                                                    m.labels_placeholder,
                                                    m.ones_placeholder,
                                                    m.zeros_placeholder,
                                                    True))
    m.inference.print_progress(info_dict)
    if i % 10000 == 0:
        m.saver.save(sess, os.path.join(m.logdir, "model.ckpt"), i)
        sigmas = m.sigU.eval()[:, 0]
        sigmas_list.append(sigmas)
        pickle.dump(sigmas_list, open(dir_name + "/sigmas.dat", "wb+"))
        if is_goog_embedding(sigmas):
            break
m.saver.save(sess, os.path.join(m.logdir, "model.ckpt"), i)
logger.debug('training finished. Results are saved in ' + dir_name)
m.dump(dir_name + "/variational.dat", d)

logger.debug('Done')