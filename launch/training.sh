# 1x2 auction
python train_model.py experiment_name=train_1x2 load.train_experiment_name=null hyp.train_num_examples=640000 hyp.epochs=10 hyp.train_batch_size=128 hyp.lr=0.001 hyp.test_val_period=1 hyp.save_period=1 hyp.test_misreport_lr=0.1 hyp.test_misreport_iters=1000 hyp.test_misreport_inits=100 model.n_agents=1 model.n_items=2 model.n_hidden_layers=3 model.rho_incr_epoch_freq=2 model.lagr_update_iter_freq=100 model.rho=1.0 model.rho_incr_amount=10 model.regret_weight=5.0 model.train_misreport_lr=0.1 model.train_misreport_iters=25 model.train_misreport_inits=10

# 2x2 auction
python train_model.py experiment_name=train_2x2 load.train_experiment_name=null hyp.train_num_examples=640000 hyp.epochs=30 hyp.train_batch_size=128 hyp.lr=0.001 hyp.test_val_period=5 hyp.save_period=10 hyp.test_misreport_lr=0.1 hyp.test_misreport_iters=1000 hyp.test_misreport_inits=100 model.n_agents=2 model.n_items=2 model.n_hidden_layers=3 model.rho_incr_epoch_freq=2 model.lagr_update_iter_freq=100 model.rho=1.0 model.rho_incr_amount=5 model.regret_weight=5.0 model.train_misreport_lr=0.1 model.train_misreport_iters=25 model.train_misreport_inits=10

# 2x3 auction
python train_model.py experiment_name=train_2x3 load.train_experiment_name=null hyp.train_num_examples=640000 hyp.epochs=20 hyp.train_batch_size=128 hyp.lr=0.001 hyp.test_val_period=5 hyp.save_period=10 hyp.test_misreport_lr=0.1 hyp.test_misreport_iters=1000 hyp.test_misreport_inits=100 model.n_agents=2 model.n_items=3 model.n_hidden_layers=5 model.lagr_update_iter_freq=100 model.rho=1.0 model.rho_incr_epoch_freq=2 model.regret_weight=5.0 model.train_misreport_lr=0.1 model.train_misreport_iters=25 model.train_misreport_inits=10

# 3x2 auction
python train_model.py experiment_name=train_3x2 load.train_experiment_name=null hyp.train_num_examples=640000 hyp.epochs=20 hyp.train_batch_size=128 hyp.lr=0.001 hyp.test_val_period=5 hyp.save_period=10 hyp.test_misreport_lr=0.1 hyp.test_misreport_iters=1000 hyp.test_misreport_inits=100 model.n_agents=3 model.n_items=2 model.n_hidden_layers=5 model.lagr_update_iter_freq=100 model.rho=1.0 model.rho_incr_epoch_freq=2 model.regret_weight=5.0 model.train_misreport_lr=0.1 model.train_misreport_iters=25 model.train_misreport_inits=10

# 3x3 auction
python train_model.py experiment_name=train_3x3 load.train_experiment_name=null hyp.train_num_examples=640000 hyp.epochs=30 hyp.train_batch_size=128 hyp.lr=0.001 hyp.test_val_period=5 hyp.save_period=10 hyp.test_misreport_lr=0.1 hyp.test_misreport_iters=1000 hyp.test_misreport_inits=100 model.n_agents=3 model.n_items=3 model.n_hidden_layers=5 model.lagr_update_iter_freq=100 model.rho=1.0 model.regret_weight=5.0 model.train_misreport_lr=0.1 model.train_misreport_iters=25 model.rho_incr_epoch_freq=8 model.train_misreport_inits=10

