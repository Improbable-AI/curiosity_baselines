from monkeycli import MonkeyCLI

monkey = MonkeyCLI()

base_command = " docker run \
                --name trainer \
                -p 12345:12345 \
                --mount src=$(pwd),target=/curiosity_baselines,type=bind \
                --mount src=$(pwd)/rlpyt/envs/pycolab,target=/pycolab,type=bind \
                --mount src=$(pwd)/rlpyt/envs/mazeworld,target=/mazeworld,type=bind \
                --mount src=$(pwd)/rlpyt/envs/gym-super-mario-bros,target=/gym-super-mario-bros,type=bind \
                -w /curiosity_baselines \
                --entrypoint python3 --rm echen9898/curiosity_baselines:0.1.8 "

arguments = "  launch.py -env Deepmind5Room-v0 -alg ppo -curiosity_alg icm -record_freq 1 -num_cpus 4 -num_envs 1 -lstm -minibatches 1 -eval_envs 0 -timestep_limit 500 -log_interval 1000 -no_extrinsic -feature_encoding idf_maze -skip_input"

print(base_command + arguments)
monkey.run_command(base_command + arguments)
