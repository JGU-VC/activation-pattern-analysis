#!/bin/sh
pdo="python do.py"

# color helpers
NO_FORMAT="\033[0m"
F_DIM="\033[2m"
F_BOLD="\033[1m"
C_BLUE="\033[38;5;12m"
C_RED="\033[38;5;9m"
C_GREEN="\033[38;5;2m"
C_YELLOW="\033[38;5;3m"

# defaults
GPU=${gpu:-0}
SEED=42
SHOW_STATS=false
SHOW_FILES=false
if [ -z $SLURM_JOB_ID ]; then
    DEBUG="run"
else
    DEBUG="off"
fi
FINAL=false
NEXT=false
CLEAR=false
DRYRUN=false
NOVAL=""
FINAL_PARAMS="--val_prop 0 --data_subset 0"

# helpers
SCRIPT_DIR="$( cd "$( dirname "$0" )" && pwd )"


# =================== #
# parse cli arguments #
# =================== #

while :; do
    case $1 in
        -h|--help)
            echo """usage: exp.sh [optional arguments] experiment-name [arguments to experiment]

optional arguments are:
    --stats			prints a table about all registered jobs and their status
    --files			prints the file names for a experiment
    -d/--debug		one of
						call	dry runs an experiment / shows only the commands that would be called
						run		runs the experiment, does not pipe output to a file
                    	off		(default)
    -g/--gpu INT	sets CUDA_VISIBLE_DEVICES to a specific id
    --seed INT		sets the seed for all experiments (default: 42)

experiments:
    appending "-final" to an experiment name sets the FINAL variable to true and also sets "--val_prop 0 --data_subset 0"
"""
            exit 0
        ;;
        --stats)
            SHOW_STATS=true
        ;;
        --files)
            SHOW_FILES=true
        ;;
        --next)
            NEXT=true
        ;;
        --clear)
            CLEAR=true
        ;;
        -d|--debug)
            case $2 in
                call|run|off) DEBUG=$2; shift ;;
                *) DEBUG="off"
            esac
        ;;
        -g|--gpu)
            case $2 in
                ''|*[!0-9]*) echo -e "${C_RED}${F_BOLD}Error:${NO_FORMAT} No number given for --gpu.\n\tGiven argument: ${C_RED}$1 $2;${NO_FORMAT}\n"; exit 1 ;;
                *) GPU=$2; shift
            esac
        ;;
        --seed)
            case $2 in
                ''|*[!0-9]*) echo -e "${C_RED}${F_BOLD}Error:${NO_FORMAT} No number given for --seed.\n\tGiven argument: ${C_RED}$1 $2;${NO_FORMAT}\n"; exit 1 ;;
                *) SEED=$2; shift
            esac
        ;;
        --dry)
            DRYRUN=true
        ;;
        --final)
            FINAL=true
            NOVAL="$FINAL_PARAMS"
        ;;
        --*)
            echo -e "${C_RED}${F_BOLD}Error:${NO_FORMAT} Argument $1 not known.${NO_FORMAT}\n"; exit 1
        ;;
        *)
            experiment=$1
            shift
            break
    esac
    shift
done





# ===================== #
# init & script helpers #
# ===================== #
DB=$SCRIPT_DIR/.db.expsh
DB_FILES=$(ls -1a $DB)

# this script uses the filesystem to determine what scripts need to be run next
mkdir -p $DB

# arrays to be filled by experiments
GLOBAL_PARAMS=("--seed $SEED" "--gpu 0")
GLOBAL_MODULES=("progressfile")
DEFAULT_PARAMS=()
DEFAULT_MODULES=()
PARAMS=()
MODULES=()

if [ "$DRYRUN" = "true" ]; then
    GLOBAL_MODULES[${#GLOBAL_MODULES[@]}]="dry-run"
    GLOBAL_PARAMS[${#GLOBAL_PARAMS[@]}]="--drymode full"
fi

# internal arrays of this script for all experiments to run
EXPSTATUS=()
EXPNAME=()
EXPMODULES_COMPLETE=()
EXPARG_COMPLETE=()
EXPMODULES=()
EXPARG=()
LEN_EXPNAME=0
LEN_EXPARG=0
LEN_EXPMODULES=0

# internal counters for this script
EXP_RUNNING=0
EXP_DONE=0
EXP_FAILED=0
EXP_TODO=0

function join_by { local d=$1; shift; local f=$1; shift; printf %s "$f" "${@/#/$d}"; }

function get_params {
    if [ "$1" = "complete" ]; then
        params="${GLOBAL_PARAMS[@]}";

        for ((i=0;i<${#DEFAULT_PARAMS[@]};++i)); do
            if [[ "${DEFAULT_PARAMS[i]}" != *@ ]]; then
                params="$params ${DEFAULT_PARAMS[i]}"
            fi
        done

    else
        params="$@";
    fi

    for ((i=0;i<${#PARAMS[@]};++i)); do
        if [[ "${PARAMS[i]}" != *@ ]]; then
            params="$params ${PARAMS[i]}"
        fi
    done

    echo "$params"
}

function get_modules {
    modules=("${GLOBAL_MODULES[@]}" "${DEFAULT_MODULES[@]}" "${MODULES[@]}")
    echo $(join_by "," "${modules[@]}")
}

function run_exp {
    params_complete=$(get_params complete)
    modules_complete=$(get_modules)
    NAME="$experiment$NAME"
    params=$(get_params)
    modules="${MODULES[@]}"
    file="$NAME"

    EXPNAME[${#EXPNAME[@]}]="$NAME"
    EXPARG[${#EXPARG[@]}]="$params"
    EXPMODULES[${#EXPMODULES[@]}]="$modules"
    EXPARG_COMPLETE[${#EXPARG_COMPLETE[@]}]="$params_complete"
    EXPMODULES_COMPLETE[${#EXPMODULES_COMPLETE[@]}]="$modules_complete"

    # get column lengths
    if [ "$SHOW_STATS" = "true" ]; then
        LEN_EXPNAME=$(( ${#NAME} > $LEN_EXPNAME ? ${#NAME} : $LEN_EXPNAME ))
        LEN_EXPMODULES=$(( ${#modules} > $LEN_EXPMODULES ? ${#modules} : $LEN_EXPMODULES ))
        LEN_EXPARG=$(( ${#params} > $LEN_EXPARG ? ${#params} : $LEN_EXPARG ))
    fi

    # in clear mode we remove all files and set to todo
    if [ "$CLEAR" = "true" ]; then
        rm -rf $DB/$file.{out,err,done,run}
        EXPSTATUS[${#EXPSTATUS[@]}]="todo"
        EXP_TODO=$((EXP_TODO+1))
        return
    fi

    # clear dangling files
    # if an experiment is marked as done, .out and .run are danglingi
    if [[ "$DB_FILES" == *"$file.done"* ]]; then
        if [[ "$DB_FILES" == *"$file.out"* ]] || [[ "$DB_FILES" == *"$file.run"* ]]; then
            rm -rf $DB/$file.{out,run}
            EXPSTATUS[${#EXPSTATUS[@]}]="done"
            EXP_DONE=$((EXP_DONE+1))
            continue
        fi
    fi

    if [[ "$DB_FILES" == *"$file.err"* ]]; then
        EXPSTATUS[${#EXPSTATUS[@]}]="failed"
        EXP_FAILED=$((EXP_FAILED+1))
    elif [[ "$DB_FILES" == *"$file.out"* ]]; then
        EXPSTATUS[${#EXPSTATUS[@]}]="running"
        EXP_RUNNING=$((EXP_RUNNING+1))
    elif [[ "$DB_FILES" == *"$file.done"* ]]; then
        EXPSTATUS[${#EXPSTATUS[@]}]="done"
        EXP_DONE=$((EXP_DONE+1))
    else
        EXPSTATUS[${#EXPSTATUS[@]}]="todo"
        EXP_TODO=$((EXP_TODO+1))
    fi

    PARAMS=()
    MODULES=()
    NAME=""
}




# ==================== #
# Start the Experiment #
# ==================== #

# Check if Experiment is registered
echo
if [ -f $SCRIPT_DIR/experiments/$experiment.sh ]; then
    echo -e "${F_BOLD}${C_GREEN}===================${NO_FORMAT}";
    echo -e "${F_BOLD}${C_GREEN}Loading Experiment:${NO_FORMAT} $experiment";
    echo -e "${F_BOLD}${C_GREEN}===================${NO_FORMAT}";
else
    echo -e "${F_BOLD}${C_RED}No experiment found with the name:${NO_FORMAT} $experiment.sh";
    echo -e "${C_RED}To register the experiment, create a file with the name:${NO_FORMAT}\n\t$SCRIPT_DIR/experiments/$experiment.sh";
    exit 1
fi

# load the experiment settings
. $SCRIPT_DIR/experiments/$experiment.sh

# entry message #
if [ "$DEBUG" != "off" ]; then
    echo -e "DEBUG-MODE: $DEBUG"
fi
echo -en "
Queried Experiment: ${C_BLUE}${F_BOLD}$experiment${NO_FORMAT}
-> with args ${C_BLUE}$@${NO_FORMAT}
-> on gpu ${C_BLUE}$GPU${NO_FORMAT}
-> with seed ${C_BLUE}$SEED${NO_FORMAT}
-> final ${C_BLUE}$FINAL${NO_FORMAT}
"
if [ -z $SLURM_JOB_ID ]; then
    echo -e "-> on local machine"
else
    echo -e "-> SLURM_JOB_ID: ${F_BOLD}$SLURM_JOB_ID${NO_FORMAT}"
fi
if [ $FINAL = "true" ]; then
    echo -e "-> is final experiment for paper"
else
    echo -e "-> ${C_RED}TODO${NO_FORMAT} rerun for paper in final mode"
fi

echo -en "
  ————————————————————
    Default Modules:   ${F_BOLD}${DEFAULT_MODULES[@]}${NO_FORMAT}
    Default Params:    ${F_BOLD}${DEFAULT_PARAMS[@]}${NO_FORMAT}
  ————————————————————

"

# statmode
# show how much are running / done / failed
if [ "$SHOW_STATS" = "true" ]; then

    C_RED=$(tput setaf 1)
    C_GREEN=$(tput setaf 2)
    C_YELLOW=$(tput setaf 3)
    C_BLUE=$(tput setaf 4)
    NO_FORMAT=$(tput sgr0)


    # helper variables
    total=${#EXPSTATUS[@]}
    running_perc=$(bc <<< "scale=1; 100 * $EXP_RUNNING / $total")
    failed_perc=$(bc <<< "scale=1; 100 * $EXP_FAILED / $total")
    done_perc=$(bc <<< "scale=1; 100 * $EXP_DONE / $total")
    todo_perc=$(bc <<< "scale=1; 100 * $EXP_TODO / $total")



    echo
    echo
    printf "  %-20s %-20s %-20s\n" STATUS NUMBER PERCENTAGE
    printf "  %-20s %-20s %-20s\n" ------ ------ ----------
    if [[ "$EXP_FAILED" -gt 0 ]]; then
        printf "  $C_RED%-20s %-20s %-20s$NO_FORMAT\n" Failed $EXP_FAILED $failed_perc%
    fi
    if [[ "$EXP_RUNNING" -gt 0 ]]; then
        printf "  $C_YELLOW%-20s %-20s %-20s$NO_FORMAT\n" Running $EXP_RUNNING $running_perc%
    fi
    if [[ "$EXP_DONE" -gt 0 ]]; then
        printf "  $C_GREEN%-20s %-20s %-20s$NO_FORMAT\n" Done $EXP_DONE $done_perc%
    fi
    if [[ "$EXP_TODO" -gt 0 ]]; then
        printf "  %-20s %-20s %-20s\n" "Queued" $EXP_TODO $todo_perc%
    fi
    printf "  $F_BOLD%-20s %-20s %-20s\n" Total ${#EXPSTATUS[@]} ""



    echo
    echo
    echo
    c1=26s
    c2=$((LEN_EXPNAME+3))s
    c3=$((LEN_EXPMODULES+3))s
    c4=${LEN_EXPARG}s
    printf "  %-$c1 %-$c2 %-$c3 %-$c4\n" STATUS EXPNAME MODULES ARGS
    printf "  %-$c1 %-$c2 %-$c3 %-$c4\n" ------ ------- ------- ----
    for ((i=0;i<${#EXPSTATUS[@]};++i)); do
        expstatus=${EXPSTATUS[i]}
        color=$NO_FORMAT
        _c1="28.26s"
        if [ "$expstatus" = "done" ]; then
            color=$C_GREEN
            expstatus="✔"
        elif [ "$expstatus" = "failed" ]; then
            color=$C_RED
            expstatus="✘ (failed)"
        elif [ "$expstatus" = "running" ]; then
            color=$C_YELLOW
            expstatus="↺ (starting)"
            runfile="${EXPNAME[i]}.run"
            if [[ "$DB_FILES" == *"$runfile"* ]]; then
                perc=$(head -n 1 "$DB/$runfile")
                remaining=$(tail -n 1 "$DB/$runfile")
		expstatus=$(printf "⌛%7s  ╱  ⏱ %8s" "$perc" "$remaining")
                _c1="31.29s"
            fi
        else
            expstatus="…"
        fi
        printf "$color  %-$_c1 %-$c2 %-$c3 %-$c4\n" "${expstatus}" "${EXPNAME[i]}" "${EXPMODULES[i]}" "${EXPARG[i]}"
    done

    echo
    echo
    echo
    printf "  %-20s %-20s %-20s\n" STATUS NUMBER PERCENTAGE
    printf "  %-20s %-20s %-20s\n" ------ ------ ----------
    if [[ "$EXP_FAILED" -gt 0 ]]; then
        printf "  $C_RED%-20s %-20s %-20s$NO_FORMAT\n" Failed $EXP_FAILED $failed_perc%
    fi
    if [[ "$EXP_RUNNING" -gt 0 ]]; then
        printf "  $C_YELLOW%-20s %-20s %-20s$NO_FORMAT\n" Running $EXP_RUNNING $running_perc%
    fi
    if [[ "$EXP_DONE" -gt 0 ]]; then
        printf "  $C_GREEN%-20s %-20s %-20s$NO_FORMAT\n" Done $EXP_DONE $done_perc%
    fi
    if [[ "$EXP_TODO" -gt 0 ]]; then
        printf "  %-20s %-20s %-20s\n" "Queued" $EXP_TODO $todo_perc%
    fi
    printf "  $F_BOLD%-20s %-20s %-20s$NO_FORMAT\n" Total "${#EXPSTATUS[@]}"

    exit 0
fi


if [ "$SHOW_FILES" = "true" ]; then

    FILES=""
    for ((i=0;i<${#EXPSTATUS[@]};++i)); do
        expstatus=${EXPSTATUS[i]}
        if [ "$expstatus" = "done" ]; then
            FILES="$FILES\n${C_GREEN}${EXPNAME[i]}.done${NO_FORMAT}"
        elif [ "$expstatus" = "failed" ]; then
            FILES="$FILES\n${C_RED}${EXPNAME[i]}.err${NO_FORMAT}"
        elif [ "$expstatus" = "running" ]; then
            FILES="$FILES\n${C_YELLOW}${EXPNAME[i]}.out${NO_FORMAT}"
        else
            FILES="$FILES\n${EXPNAME[i]}.marked"
        fi
    done
    echo
    echo
    echo -e "$FILES"
    exit 0

fi


# add file descriptor for flock
exec 9>$DB/global.lock || exit 1

# actually run the experiments
echo
echo
echo -e "${C_GREEN}====================${NO_FORMAT}"
echo -e "${C_GREEN}Starting Experiments${NO_FORMAT}"
echo -e "${C_GREEN}====================${NO_FORMAT}"
echo
for ((i=0;i<${#EXPSTATUS[@]};++i)); do

    expname="${EXPNAME[i]}"
    exparg="${EXPARG[i]}"
    expstatus="${EXPSTATUS[i]}"

    # show debug info about the current file
    if [ "$DEBUG" != "off" ]; then
        if [ "$expstatus" = "done" ]; then
            echo -e "${C_GREEN}Done, thus skipping $expname${NO_FORMAT}"
            continue
        elif [ "$expstatus" = "failed" ]; then
            echo -e "${C_RED}Failed, thus skipping $expname${NO_FORMAT}"
            continue
        elif [ "$expstatus" = "running" ]; then
            echo -e "${C_YELLOW}Runnig, thus skipping $expname${NO_FORMAT}"
            continue
        fi
    elif [ "$expstatus" != "todo" ]; then
        continue
    fi

    echo -e "${C_BLUE}Starting $expname${NO_FORMAT}"

    # construct call
    FULL_CALL="$pdo ${EXPMODULES_COMPLETE[i]} --log.tag $expname ${EXPARG_COMPLETE[i]} $@"
    CUDA_VISIBLE_DEVICES=$GPU

    # if debugging calls, just print the exact experiment call
    if [ "$DEBUG" = "call" ]; then
        echo -e "\t$FULL_CALL"
        if [ "$NEXT" = "true" ]; then
            break
        fi
        continue
    fi

    # note: only these calls to $DB need to be flocked
    # for all other calls, the order is chosen in such a way that no ambiguity can occur
    flock 9
        # check once more, if the experiment is currently used somewhere
        if compgen -G "${DB}/$expname.*" > /dev/null; then
            echo -e "${C_YELLOW}Touched by another process, thus skipping $expname${NO_FORMAT}"
            continue
        fi

        # mark file in DB
        touch $DB/$expname.out
        echo touch $DB/$expname.out
    flock -u 9

    # in debug-mode, the output will be shown for the callee
    if [ "$DEBUG" = "run" ]; then
        echo -e "\tmodules: ${EXPMODULES[i]}\n\targs: ${EXPARG[i]} $@"
    fi

    # create tmpfile & trap script
    trap 'rm -f "$PROGRESSFILE"' EXIT
    PROGRESSFILE="$DB/$expname.run"
    touch $PROGRESSFILE

    # run & check for return code
    if [ $DEBUG == "run" ] && CUDA_VISIBLE_DEVICES=$GPU PROGRESSFILE=$PROGRESSFILE $FULL_CALL || [ $DEBUG == "off" ] && CUDA_VISIBLE_DEVICES=$GPU PROGRESSFILE=$PROGRESSFILE $FULL_CALL > $DB/$expname.out 2>&1; then
        echo -e "${C_GREEN}Done $expname${NO_FORMAT}"

        # if done, we can mark this in the DB
        touch $DB/$expname.done
        \rm $DB/$expname.out

    else
        echo -e "${C_RED}Failed $expname${NO_FORMAT}"

        # if failed, we move log to err-file
        \cp $DB/$expname.out $DB/$expname.err
        \rm $DB/$expname.out
    fi

    # debug only this first run
    if [ "$NEXT" = "true" ]; then
        break
    fi

done
