
if [ ! -e $dataset ]; then
    dataset=${dataset}
else
    dataset=cifar10
fi

# complete
plots=(
    "% Hashmap Filled"
    "% max Entropy"
    # Entropy
    # "[sinceLast] JI(last,current)"
    # "[sincelast][>T] JI(last,current)"
    # "[sinceLast] JI(last,current)"
    # "[sinceInit] JI(init,current)"
    # "[sinceFinal] JI(final,current)"
    # "[sinceLast] wJI(last,current)"
    # "[sinceInit] wJI(init,current)"
    "[sinceFinal] wJI(final,current)"
    # "[sinceLast] % Patterns Changed"
    # "[sinceInit] JI(init,current)"
    # "% Num Patterns"
    # "% Count of most frequent Pattern"
)

files=(
    # logs/measures-methods-cifar10-full_convnet20_multistep_relu.json
    # logs/measures-methods-cifar10-full_convnet20_multistep_prelu.json
    # logs/measures-methods-cifar10-full_convnet20_multistep_lrelu.json
    # logs/measures-methods-cifar10-full_convnet20_cycliclr.json
    # # logs/measures-methods-cifar10-full_convnet20_onecyclelr-lin.json
    # logs/measures-methods-cifar10-full_convnet20_onecyclelr-cos.json
    # logs/measures-methods-cifar10-full_pyramid20_multistep_relu.json
    # logs/measures-methods-cifar10-full_resnet20_multistep_relu.json
    # logs/measures-methods-cifar10-full_resnet20_multistep_prelu.json
    # logs/measures-methods-cifar10-full_resnet20_multistep_lrelu.json
    # logs/measures-methods-cifar10-full_resnet20_cycliclr.json
    # logs/measures-methods-cifar10-full_resnet20_onecyclelr-lin.json
    # logs/measures-methods-cifar10-full_resnet20_onecyclelr-cos.json
    logs/measures-methods-cifar10-full_resnet20_fixup.json
)


echo Dataset: $dataset

for file in ${files[@]}; do
    for plot in "${plots[@]}"; do
        echo "FILE = $file";
        echo "======";
        echo -e "\t$plot"
        python ./plot.py 2dscalar "$file" "$plot" --save --label $dataset
    done;
done;
