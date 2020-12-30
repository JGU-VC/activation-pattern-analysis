
while :; do
    case $1 in
        -h|--help)
            echo """experiment usage: $experiment dataset
"""
            exit 0
        ;;
        *)
            dataset=$1
            shift || { echo -e "${C_RED}Error: No dataset parameter specified.${NO_FORMAT}" && exit 1; }
            break
    esac
    shift
done

valid_datasets=(
    "cf10"   "cifar10"
    "cf100"  "cifar100"
    "tiny"   "tinyimagenet"
)

found=false
for ((i=0;i<${#valid_datasets[@]};i=i+2)); do
    if [ "$dataset" == "${valid_datasets[i]}" ] || [ "$dataset" = "${valid_datasets[i+1]}" ]; then
        dataset="${valid_datasets[i+1]}"
        datasetshort="${valid_datasets[i]}"
        found=true
        break
    fi
done


# check if dataset specified
if [ "$dataset" == "" ]; then
    echo
    echo -e ${C_RED}Error: No nataset parameter specified.${NO_FORMAT}
    exit 1;
elif [ "$found" != "true" ]; then
    echo
    echo -e ${C_RED}Error: Dataset $dataset not known.${NO_FORMAT}
    exit 1;
fi
