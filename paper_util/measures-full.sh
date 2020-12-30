#!/bin/bash
pwd

if [ ! -e $dataset ]; then
    dataset=${dataset}
else
    dataset=cifar10
fi

# complete
# plots=(
#     "% Hashmap Filled"
#     "% max Entropy"
#     "Entropy"
#     "[sinceLast] JI(last,current)"
#     "[sincelast][>T] JI(last,current)"
#     "[sinceLast] % Patterns Changed"
#     "[sinceInit] JI(init,current)"
#     "Num Patterns"
#     "% Count of most frequent Pattern"
# )

plots=(
    "% Hashmap Filled"
    "% max Entropy"
    # Entropy
    # "[sinceLast] JI(last,current)"
    "[sinceInit] JI(init,current)"
    "[sinceInit] wJI(init,current)"
    "[sinceLast] % Patterns Changed"
    "% Num Patterns"
    "% Count of most frequent Pattern"
)




echo Dataset: $dataset

for net in resnet20 convnet20 toynet20; do
    # for mode in early full; do
    for mode in early; do # full; do
        file="./logs/measures-$dataset-${mode}_${net}_multistep_relu.json"
        echo "FILE = $file";
        echo "======";
        for plot in "${plots[@]}"; do
            echo -e "\t$plot"
            python ./plot.py 2dscalar "$file" "$plot" --save --label $dataset
        done;
    done;
done;
