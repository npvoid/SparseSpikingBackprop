import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='fmnist', help='Dataset options: fmnist, nmnist, SHD ')
parser.add_argument('--seed', type=int, default=0, help='Seed')
parser.add_argument('--nb_trials', type=int, default=1, help='Number of trials')
parser.add_argument('--nb_hidden', type=int, nargs='+', default=[200], help='Number hidden neurons in layers 1 and 2')
parser.add_argument('--prefix', type=str, default='RUN', help='Name of the experiment')
parser.add_argument('--nb_epochs', type=int, default=100, help='Number of epochs')

parser.add_argument('--run_orig', action='store_true', help='Running pytorch original version')
parser.add_argument('--run_s3gd', action='store_true', help='Running sparse gradient implementation')
parser.add_argument('--time_freq', type=int, default=1, help='How often we time')

OPTIONS = parser.parse_args()

if __name__ == "__main__":

    # Run train and time
    assert OPTIONS.run_orig == OPTIONS.run_s3gd == True, "Should run both versions"

    from train_and_time import run
    dataset = OPTIONS.dataset
    hidden_list = OPTIONS.nb_hidden
    nb_trials = OPTIONS.nb_trials
    prs = {
        'PREFIX': OPTIONS.prefix,
        'nb_epochs': OPTIONS.nb_epochs,
        'seed': OPTIONS.seed,
        'time_freq': OPTIONS.time_freq,

        'run_orig': OPTIONS.run_orig,
        'run_s3gd': OPTIONS.run_s3gd,
        'algo': 'ORIG' if (OPTIONS.run_orig and OPTIONS.hpc) else ('S3GD' if OPTIONS.hpc else 'BOTH'),
    }
    run(dataset, hidden_list, nb_trials, hpc=OPTIONS.hpc, prs=prs)
