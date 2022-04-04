#!/usr/bin/env bash

# There's a memory leak somewhere in the TensorRT runtime or the PyCUDA library
# or our code. As a workaround, we keep running the `run_prediction.py` script
# until it exits successfully (returns 0 exit code).
max_retry=5
counter=0

run() {
    echo "$@"
    python3 run_prediction.py "$@"
}

run_to_finish() {
    run "$@"
    local last_run_exit_code=$?
    while [ $last_run_exit_code -ne 0 ]; do

        if [ $last_run_exit_code -eq 4 ]; then
            # exit code 4 means finish one batch of models, not an error
            printf "\nLast exit code: %d (%s)\n\n" $last_run_exit_code "finished one ensemble but not all"
        elif [ $last_run_exit_code -eq 3 ]; then
            # Sometimes, somehow the symbolic links of the dynamic libs in
            # `/usr/lib/aarch64-linux-gnu` are messed up / modified by someone after
            # running the program. I haven't figured out how that happens and who
            # does it. When that happens, an OSError exception will be raised
            # because certain libs cannot be found. We will catch this exception in
            # python and exit with code 3. As a workaround, when we see the exit
            # code 3, a potential symbolic link issue, we run `ldconfig` to
            # reconcile the links.
            printf "\nLast exit code: %d (%s)\n\n" $last_run_exit_code "potential symbolic link issue"
            printf "Symbolic links broken. Trying to fix that with ldconfig.\n"
            ldconfig
            printf "Retry %s\n" $counter
        elif [ $last_run_exit_code -eq 137 ]; then
            printf "\nLast exit code: %d (%s)\n\n" $last_run_exit_code "potential OOM"
            printf "Retry %s\n" $counter
        else
            printf "\nLast exit code: %d (%s)\n\n" $last_run_exit_code ""
            printf "Retry %s\n" $counter
        fi

        run "$@"
        local last_run_exit_code=$?

	[[ $last_run_exit_code -eq 0 ]] && exit 0

        [[ $last_run_exit_code -ne 4 ]] && ((counter++))
        if [ $counter -eq $max_retry ]; then
            printf "Max retry reached. Stop trying. Last exit code: %d" $last_run_exit_code
            exit 0
        fi
    done
}

#
# Latency profiling
#
run_to_finish /data/ResNet50_ImageNet trt latency ensemble 6
