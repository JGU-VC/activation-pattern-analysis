
if [ "$allowed" != "" ]; then
    found=false
    for a in $allowed; do
        if [ "$a" == "$1" ]; then
            found=true
        fi
    done

    if [ "$found" == "false" ]; then
        echo -e "${C_RED}Error: Mode parameter '$1' not known.${NO_FORMAT}"
        echo -e "${C_RED}\tAllowed are the following values $allowed.${NO_FORMAT}"
        exit 1
    fi
fi

# extract experiment mode
mode=$1
shift || { echo -e "${C_RED}Error: No mode parameter specified.${NO_FORMAT}" && exit 1; }

# reset for next call
allowed=""

echo $mode
