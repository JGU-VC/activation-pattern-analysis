
# ========================= #
# Experiment: entropy curve #
# ========================= #
# this experiment determines if generalization error and entropy are correlated


# parse cli args
. $SCRIPT_DIR/experiments/util/arg_dataset.sh
allowed="shallow deep" . $SCRIPT_DIR/experiments/util/arg_mode.sh;

# the parameters that are to be used for all experiments
seeds=25
every_epoch=0.1
if [ "$dataset" == cifar10 ]; then
    measure_points="--change 3 every_epoch 1000 50 every_epoch $every_epoch 51 every_epoch 1000 100 every_epoch $every_epoch 101 every_epoch 1000 199 every_epoch $every_epoch"
elif [ "$dataset" == tinyimagenet ]; then
    measure_points="--change 3 every_epoch 1000 69 every_epoch $every_epoch 71 every_epoch 1000 74 every_epoch $every_epoch"
fi

DEFAULT_PARAMS=(
    "$NOVAL"
    "--every_epoch $every_epoch"
    "$measure_points"
)
DEFAULT_MODULES=(
    sota$dataset
    DS-act-stats
    change-at
)

if [ "$mode" = "shallow" ]; then
    depths=$(seq 1 1 3)
else
    depths=$(seq 4 1 5)
fi

for net in resnet20 convnet20 toynet20; do
    for seed in $(seq $seeds); do
        for depth in $depths; do
            # args
            MODULES=($net)
            PARAMS=(
                "--resnet.num_blocks $depth $depth $depth"
                "--seed $seed"
            )

            # full_exp_name=$experiment-${net}_$logadd
            NAME="-${net}_depth=${depth}_seed=$seed"
            run_exp
        done
    done
done
