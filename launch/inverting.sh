## With initializations sampled from P
# 1x2 auction
python invert_model.py load.model_checkpoint=model_final.pth hyp.use_bidder_info=False hyp.privacy_lr=0.002 hyp.sigma=0.75 experiment_name=invert_1x2 load.train_experiment_name=train_1x2 load.train_run_id=null hyp.privacy_iters=50000 hyp.test_batch_size=16 hyp.test_misreport_inits=50 hyp.test_num_examples=3000 hyp.util_num_samples=50
python invert_model.py load.model_checkpoint=model_final.pth hyp.use_bidder_info=True hyp.privacy_lr=0.002 hyp.sigma=0.75 experiment_name=invert_1x2 load.train_experiment_name=train_1x2 load.train_run_id=null hyp.privacy_iters=50000 hyp.test_batch_size=16 hyp.test_misreport_inits=50 hyp.test_num_examples=3000 hyp.util_num_samples=50

# 2x2 auction
python invert_model.py load.model_checkpoint=model_final.pth hyp.use_bidder_info=False hyp.privacy_lr=0.002 hyp.sigma=0.75 experiment_name=invert_2x2 load.train_experiment_name=train_2x2 load.train_run_id=null hyp.privacy_iters=50000 hyp.test_batch_size=16 hyp.test_misreport_inits=50 hyp.test_num_examples=3000 hyp.util_num_samples=50
python invert_model.py load.model_checkpoint=model_final.pth hyp.use_bidder_info=True hyp.privacy_lr=0.002 hyp.sigma=0.75 experiment_name=invert_2x2 load.train_experiment_name=train_2x2 load.train_run_id=null hyp.privacy_iters=50000 hyp.test_batch_size=16 hyp.test_misreport_inits=50 hyp.test_num_examples=3000 hyp.util_num_samples=50

# 2x3 auction
python invert_model.py load.model_checkpoint=model_final.pth hyp.use_bidder_info=False hyp.privacy_lr=0.002 hyp.sigma=0.75 experiment_name=invert_3x2 load.train_experiment_name=train_3x2 load.train_run_id=null hyp.privacy_iters=50000 hyp.test_batch_size=16 hyp.test_misreport_inits=50 hyp.test_num_examples=3000 hyp.util_num_samples=50
python invert_model.py load.model_checkpoint=model_final.pth hyp.use_bidder_info=True hyp.privacy_lr=0.002 hyp.sigma=0.75 experiment_name=invert_3x2 load.train_experiment_name=train_3x2 load.train_run_id=null hyp.privacy_iters=50000 hyp.test_batch_size=16 hyp.test_misreport_inits=50 hyp.test_num_examples=3000 hyp.util_num_samples=50

# 3x2 auction
python invert_model.py load.model_checkpoint=model_final.pth hyp.use_bidder_info=False hyp.privacy_lr=0.002 hyp.sigma=0.75 experiment_name=invert_2x3 load.train_experiment_name=train_2x3 load.train_run_id=null hyp.privacy_iters=50000 hyp.test_batch_size=16 hyp.test_misreport_inits=50 hyp.test_num_examples=3000 hyp.util_num_samples=50
python invert_model.py load.model_checkpoint=model_final.pth hyp.use_bidder_info=True hyp.privacy_lr=0.002 hyp.sigma=0.75 experiment_name=invert_2x3 load.train_experiment_name=train_2x3 load.train_run_id=null hyp.privacy_iters=50000 hyp.test_batch_size=16 hyp.test_misreport_inits=50 hyp.test_num_examples=3000 hyp.util_num_samples=50

# 3x3 auction
python invert_model.py load.model_checkpoint=model_final.pth hyp.use_bidder_info=False hyp.privacy_lr=0.002 hyp.sigma=0.75 experiment_name=invert_3x3 load.train_experiment_name=train_3x3 load.train_run_id=null hyp.privacy_iters=50000 hyp.test_batch_size=16 hyp.test_misreport_inits=50 hyp.test_num_examples=3000 hyp.util_num_samples=50
python invert_model.py load.model_checkpoint=model_final.pth hyp.use_bidder_info=True hyp.privacy_lr=0.002 hyp.sigma=0.75 experiment_name=invert_3x3 load.train_experiment_name=train_3x3 load.train_run_id=null hyp.privacy_iters=50000 hyp.test_batch_size=16 hyp.test_misreport_inits=50 hyp.test_num_examples=3000 hyp.util_num_samples=50


## With zero initializations
# 1x2 auction
python invert_model.py load.model_checkpoint=model_final.pth hyp.use_bidder_info=False hyp.privacy_lr=0.002 hyp.sigma=0.75 experiment_name=invert_1x2_zeros load.train_experiment_name=train_1x2 load.train_run_id=null hyp.privacy_iters=50000 hyp.test_batch_size=16 hyp.test_misreport_inits=50 hyp.test_num_examples=3000 hyp.util_num_samples=50 hyp.privacy_data_generating_distribution=zeros

# 2x2 auction
python invert_model.py load.model_checkpoint=model_final.pth hyp.use_bidder_info=False hyp.privacy_lr=0.002 hyp.sigma=0.75 experiment_name=invert_2x2_zeros load.train_experiment_name=train_2x2 load.train_run_id=null hyp.privacy_iters=50000 hyp.test_batch_size=16 hyp.test_misreport_inits=50 hyp.test_num_examples=3000 hyp.util_num_samples=50 hyp.privacy_data_generating_distribution=zeros

# 2x3 auction
python invert_model.py load.model_checkpoint=model_final.pth hyp.use_bidder_info=False hyp.privacy_lr=0.002 hyp.sigma=0.75 experiment_name=invert_3x2_zeros load.train_experiment_name=train_3x2 load.train_run_id=null hyp.privacy_iters=50000 hyp.test_batch_size=16 hyp.test_misreport_inits=50 hyp.test_num_examples=3000 hyp.util_num_samples=50 hyp.privacy_data_generating_distribution=zeros

# 3x2 auction
python invert_model.py load.model_checkpoint=model_final.pth hyp.use_bidder_info=False hyp.privacy_lr=0.002 hyp.sigma=0.75 experiment_name=invert_2x3_zeros load.train_experiment_name=train_2x3 load.train_run_id=null hyp.privacy_iters=50000 hyp.test_batch_size=16 hyp.test_misreport_inits=50 hyp.test_num_examples=3000 hyp.util_num_samples=50 hyp.privacy_data_generating_distribution=zeros

# 3x3 auction
python invert_model.py load.model_checkpoint=model_final.pth hyp.use_bidder_info=False hyp.privacy_lr=0.002 hyp.sigma=0.75 experiment_name=invert_3x3_zeros load.train_experiment_name=train_3x3 load.train_run_id=null hyp.privacy_iters=50000 hyp.test_batch_size=16 hyp.test_misreport_inits=50 hyp.test_num_examples=3000 hyp.util_num_samples=50 hyp.privacy_data_generating_distribution=zeros


