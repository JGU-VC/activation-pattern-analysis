
# ============================ #
# Experiment: measures methods #
# ============================ #
# this experiments measures activation measures over the full dataset for several methods


# parse cli args
. $SCRIPT_DIR/experiments/util/arg_dataset.sh
allowed="early full" . $SCRIPT_DIR/experiments/util/arg_mode.sh;

# the parameters that are to be used for all experiments


if [ "$mode" = "early" ]; then
    every_epoch=0.0000001
    epochs=10
else
    every_epoch=0.5
    epochs=@
fi

if [ "$dataset" == cifar10 ]; then
    specific="--hashmap_size_factor 3e8"
elif [ "$dataset" == tinyimagenet ]; then
    specific="--hashmap_size_factor 3e8"
    if [ "$mode" = "early" ]; then
        epochs=4
    fi
fi





DEFAULT_MODULES=(
    sota${dataset}.multistep
    DS-act-stats
)

DEFAULT_PARAMS=(
    "$NOVAL"
    "--every_epoch $every_epoch"
    "--epochs $epochs"
    "$specific"
)


for net in resnet20 convnet20 toynet20; do

    MODULES=($net)
    NAME="-${dataset}-${mode}_${net}_multistep_relu"
    run_exp

done
