# Inverting deterministic models
# Using bidder information
python invert_model.py load.model_path=<path to checkpoint> hyp.use_bidder_info=True hyp.privacy_lr=0.002 hyp.sigma=0.0 hyp.privacy_iters=50000 hyp.test_batch_size=16 hyp.test_misreport_inits=50 hyp.test_num_examples=3000 hyp.util_num_samples=50

# Without using bidder information
python invert_model.py load.model_path=<path to checkpoint> hyp.use_bidder_info=False hyp.privacy_lr=0.002 hyp.sigma=0.0 hyp.privacy_iters=50000 hyp.test_batch_size=16 hyp.test_misreport_inits=50 hyp.test_num_examples=3000 hyp.util_num_samples=50

# Inverting noisy models
# Using bidder information
python invert_model.py load.model_path=<path to checkpoint> hyp.use_bidder_info=True hyp.privacy_lr=0.002 hyp.sigma=0.2 hyp.privacy_iters=50000 hyp.test_batch_size=16 hyp.test_misreport_inits=50 hyp.test_num_examples=3000 hyp.util_num_samples=50

# Without using bidder information
python invert_model.py load.model_path=<path to checkpoint> hyp.use_bidder_info=False hyp.privacy_lr=0.002 hyp.sigma=0.2 hyp.privacy_iters=50000 hyp.test_batch_size=16 hyp.test_misreport_inits=50 hyp.test_num_examples=3000 hyp.util_num_samples=50