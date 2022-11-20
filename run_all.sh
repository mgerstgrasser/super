while read CLI_ARGS; do

[[ $CLI_ARGS =~ ^#.* ]] && continue

# Print parameters, and export them
# echo "CLI_ARGS: ${CLI_ARGS}"

echo ray job submit --no-wait --runtime-env=ray_runtime.yaml --working-dir=suPER/experiments -- python main.py ${CLI_ARGS} --experiment_project=super_allexperiments_ray_sereval --ray_plain_init --seed={1..3}
ray job submit --no-wait --runtime-env=ray_runtime.yaml --working-dir=suPER/experiments -- python main.py ${CLI_ARGS} --experiment_project=super_allexperiments_ray_sereval --ray_plain_init --seed=1
ray job submit --no-wait --runtime-env=ray_runtime.yaml --working-dir=suPER/experiments -- python main.py ${CLI_ARGS} --experiment_project=super_allexperiments_ray_sereval --ray_plain_init --seed=2
ray job submit --no-wait --runtime-env=ray_runtime.yaml --working-dir=suPER/experiments -- python main.py ${CLI_ARGS} --experiment_project=super_allexperiments_ray_sereval --ray_plain_init --seed=3


# sleep 1 # pause to be kind to the scheduler

done <suPER/experiments/configs.txt

