

# ============================ #
# Experiment: measures methods #
# ============================ #
# this experiments measures activation measures over the full dataset for several methods


# parse cli args
. $SCRIPT_DIR/experiments/util/arg_dataset.sh
allowed="early full" . $SCRIPT_DIR/experiments/util/arg_mode.sh; len=$mode
allowed="save actswithend acts" . $SCRIPT_DIR/experiments/util/arg_mode.sh;

# the parameters that are to be used for all experiments
defaultnet=resnet
netsize=20

# modes
if [ "$len" = "early" ]; then
    every_epoch=0.0000001
    epochs=10
else
    every_epoch=1
    epochs=@
fi

# dataset specific settings
# if [ "$dataset" == cifar10 ]; then
#     specific=""
# elif [ "$dataset" == tinyimagenet ]; then
#     specific=""
# fi

if [ "$dataset" == cifar10 ]; then
    specific="--hashmap_size_factor 3e8"
elif [ "$dataset" == tinyimagenet ]; then
    specific="--hashmap_size_factor 3e8"
fi





if [ "$mode" == save ]; then
    DEFAULT_PARAMS=(
        "$FINAL_PARAMS"
        "--num_workers 0 --disable_multiprocessing"

        "--save.every_epoch -1"
        "--epochs $epochs"

        "--DS-act-stats.every_epoch $every_epoch"
        "$specific"
        "--since_init False"
        "--since_last False"
        "--dummy True"
    )
    DEFAULT_MODULES=(
        save
        DS-act-stats # in dummy-mode
    )
else
    DEFAULT_PARAMS=(
        "$FINAL_PARAMS"
        "--num_workers 0 --disable_multiprocessing"

        "--every_epoch $every_epoch"
        "--epochs $epochs"
        "$specific"
        "--since_init True"
        "--since_last True"
    )
    if [ "$mode" == actswithend ]; then
        DEFAULT_PARAMS[${#DEFAULT_PARAMS[@]}]="--since_final True"
    else
        DEFAULT_PARAMS[${#DEFAULT_PARAMS[@]}]="--since_final False"
    fi
    DEFAULT_MODULES=(
        DS-act-stats
    )
fi



# defaultnet="${defaultnet}$netsize"

# for defaultnet in convnet$netsize; do
for defaultnet in convnet$netsize resnet$netsize; do

    MODULES=(sota${dataset}.multistep $defaultnet)
    NAME="-${dataset}-${len}_${defaultnet}_multistep_relu"
    run_exp

    MODULES=(sota${dataset}.multistep $defaultnet prelu)
    NAME="-${dataset}-${len}_${defaultnet}_multistep_prelu"
    run_exp

    MODULES=(sota${dataset}.multistep $defaultnet lrelu)
    NAME="-${dataset}-${len}_${defaultnet}_multistep_lrelu"
    run_exp

    MODULES=(sota${dataset}.cycliclr $defaultnet)
    NAME="-${dataset}-${len}_${defaultnet}_cycliclr"
    run_exp

    # MODULES=(sota${dataset}.onecycle $defaultnet)
    # PARAMS=("--onecyclelr.anneal_strategy lin")
    # NAME="-${dataset}-${len}_${defaultnet}_onecyclelr-lin"
    # run_exp

    MODULES=(sota${dataset}.onecycle $defaultnet)
    NAME="-${dataset}-${len}_${defaultnet}_onecyclelr-cos"
    run_exp

done

MODULES=(sota${dataset}.fixup $defaultnet)
NAME="-${dataset}-${len}_${defaultnet}_fixup"
run_exp

MODULES=(sota${dataset}.pyramid pyramidnet$netsize)
NAME="-${dataset}-${len}_pyramid${netsize}"
run_exp

# MODULES=(sota${dataset}.multistep $net metainit)
# NAME="-${dataset}-${len}_${net}_multistep_metainit"
# run_exp

# MODULES=(sota${dataset} $net weightnorm)
# NAME="-${dataset}-${len}_${net}_weightnorm"
# run_exp
