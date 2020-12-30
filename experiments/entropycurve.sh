
# ========================= #
# Experiment: entropy curve #
# ========================= #
# this experiment determines if generalization error and entropy are correlated

# parse cli args
# usage: ... dataset shallow|deep
. $SCRIPT_DIR/experiments/util/arg_dataset.sh
# allowed="shallow deep" . $SCRIPT_DIR/experiments/util/arg_mode.sh;


# -------- #
# settings # experiment settings
# -------- #
seeds=25
every_epoch=0.000001
never=1e6
stats_off="activation_stats.skip True"
stats_on="activation_stats.skip False"

# measure points
case $dataset in
    cifar10)
        measure_points_=(

            # debugging
            # "0.01 every_epoch $never"
            # "0.01 $stats_off"

            # correct measure points
            "0.01 every_epoch 0.01" # 4 - 40
            "0.1 $stats_off"
            "0.1 every_epoch $never"
            "99.99 $stats_on" # last batch before multistep drop (batchsize 128)
            "99.99 every_epoch $every_epoch"
            "100.007 $stats_off"
            "100.007 every_epoch $never"
            "199.990 $stats_on" # last batch before training end (batchsize 128)
            "199.990 every_epoch $every_epoch"
        )
    ;;
    tinyimagenet)
        measure_points_=(

            # debugging
            # "0.01 every_epoch $never"
            # "0.01 $stats_off"

            # correct measure points
            "0.01 every_epoch 0.01" # 4 - 40
            "0.1 $stats_off"
            "0.1 every_epoch $never"
            "69.99 $stats_on" # last batch before multistep drop (batchsize 128)
            "69.99 every_epoch $every_epoch"
            "70.007 $stats_off"
            "70.007 every_epoch $never"
            "79.990 $stats_on" # last batch before training end (batchsize 128)
            "79.990 every_epoch $every_epoch"
        )
    ;;
    *)
        echo -n "${C_RED}Error: This Experiment is not defined for the dataset '$dataset'${NO_FORMAT}"
        exit 1
esac
measure_points="--change ${measure_points_[@]}"

# set depths
depths=$(seq 1 1 5)


# set nets
nets="resnet20 convnet20 toynet20"

# -------- #
# defaults # the parameters that are to be used for all experiments
# -------- #
DEFAULT_MODULES=(
    sota$dataset.multistep
    DS-act-stats
    change-at
)
DEFAULT_PARAMS=(
    "$FINAL_PARAMS"
    "--every_epoch $every_epoch"
    "$measure_points"

    "--hashmap_size_factor 7e8"
    "--DS-act-stats.since_init false"
    "--DS-act-stats.since_last false"
)

for seed in $(seq $seeds); do
    for net in $nets; do
        for depth in $depths; do

            # args
            MODULES=($net)
            PARAMS=(
                "--resnet.num_blocks $depth $depth $depth"
                "--seed $seed"
            )

            # full_exp_name=$experiment-${net}_$logadd
            NAME="-${dataset}-${net}_depth=${depth}_seed=$seed"
            run_exp
        done
    done
done
