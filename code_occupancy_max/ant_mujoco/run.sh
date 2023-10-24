# python main.py --test=True --env="Ant-v4" --model=mop --path=e_data/ant_z03_r0_a1/ant_z03_r0_a1_s0/pyt_save/model.pt
# python main.py --test=True --env="Ant-v4" --model=mop --path=e_data/ant_z03_r0_a1/ant_z03_r0_a1_s1/pyt_save/model.pt
# python main.py --test=True --env="Ant-v4" --model=mop --path=e_data/ant_z03_r0_a1/ant_z03_r0_a1_s2/pyt_save/model.pt
# python main.py --test=True --env="Ant-v4" --model=mop --path=e_data/ant_z03_r0_a1/ant_z03_r0_a1_s3/pyt_save/model.pt

# python main.py --test=True --env="Ant-v4" --model=e --epsilon=0.1 --path=e_data/ant_z03_e_greedy_r1_a0_eps_01/ant_z03_e_greedy_r1_a0_s0/pyt_save/model.pt 
# python main.py --test=True --env="Ant-v4" --model=e --epsilon=0.1 --path=e_data/ant_z03_e_greedy_r1_a0_eps_01/ant_z03_e_greedy_r1_a0_s1/pyt_save/model.pt 
# python main.py --test=True --env="Ant-v4" --model=e --epsilon=0.1 --path=e_data/ant_z03_e_greedy_r1_a0_eps_01/ant_z03_e_greedy_r1_a0_s2/pyt_save/model.pt 
# python main.py --test=True --env="Ant-v4" --model=e --epsilon=0.1 --path=e_data/ant_z03_e_greedy_r1_a0_eps_01/ant_z03_e_greedy_r1_a0_s3/pyt_save/model.pt 

# python main.py --test=True --env="Ant-v4" --model=e --epsilon=0.33 --path=e_data/y/ant_z03_e_greedy_r1_a0_s0/pyt_save/model.pt 
# python main.py --test=True --env="Ant-v4" --model=e --epsilon=0.33 --path=e_data/ant_z03_e_greedy_r1_a0_eps_033/ant_z03_e_greedy_r1_a0_s1/pyt_save/model.pt 
# python main.py --test=True --env="Ant-v4" --model=e --epsilon=0.33 --path=e_data/ant_z03_e_greedy_r1_a0_eps_033/ant_z03_e_greedy_r1_a0_s2/pyt_save/model.pt 
# python main.py --test=True --env="Ant-v4" --model=e --epsilon=0.33 --path=e_data/ant_z03_e_greedy_r1_a0_eps_033/ant_z03_e_greedy_r1_a0_s3/pyt_save/model.pt 


# (trap 'kill 0' SIGINT;
# python main.py --exp_name="ant_z03_e_greedy_r1_a0_eps_015" --env="Ant-v4" --epsilon=0.15 --apply_reward=True --reward=1 --alpha=0 --epochs=500 --seed=0 --model=e & 
# python main.py --exp_name="ant_z03_e_greedy_r1_a0_eps_015" --env="Ant-v4" --epsilon=0.15 --apply_reward=True --reward=1 --alpha=0 --epochs=500 --seed=1 --model=e &
# python main.py --exp_name="ant_z03_e_greedy_r1_a0_eps_015" --env="Ant-v4" --epsilon=0.15 --apply_reward=True --reward=1 --alpha=0 --epochs=500 --seed=2 --model=e &
# python main.py --exp_name="ant_z03_e_greedy_r1_a0_eps_015" --env="Ant-v4" --epsilon=0.15 --apply_reward=True --reward=1 --alpha=0 --epochs=500 --seed=3 --model=e &
# )


# python main.py --test=True --env="Ant-v4" --model=e --epsilon=0.15 --path=data/ant_z03_e_greedy_r1_a0_eps_015/ant_z03_e_greedy_r1_a0_eps_015_s0/pyt_save/model.pt
# python main.py --test=True --env="Ant-v4" --model=e --epsilon=0.15 --path=data/ant_z03_e_greedy_r1_a0_eps_015/ant_z03_e_greedy_r1_a0_eps_015_s1/pyt_save/model.pt
# python main.py --test=True --env="Ant-v4" --model=e --epsilon=0.15 --path=data/ant_z03_e_greedy_r1_a0_eps_015/ant_z03_e_greedy_r1_a0_eps_015_s2/pyt_save/model.pt
# python main.py --test=True --env="Ant-v4" --model=e --epsilon=0.15 --path=data/ant_z03_e_greedy_r1_a0_eps_015/ant_z03_e_greedy_r1_a0_eps_015_s3/pyt_save/model.pt


# (trap 'kill 0' SIGINT;
# python main.py --exp_name="en_ant_mop_r0_a1" --apply_reward=True --reward=0 --alpha=1 --epochs=1000 --seed=0 --model=mop & 
# python main.py --exp_name="en_ant_mop_r0_a1" --apply_reward=True --reward=0 --alpha=1 --epochs=1000 --seed=1 --model=mop & 
# python main.py --exp_name="en_ant_eg_r1_a0_eps_015" --epsilon=0.2 --apply_reward=True --reward=1 --alpha=0 --epochs=1000 --seed=0 --model=e & 
# python main.py --exp_name="en_ant_eg_r1_a0_eps_015" --epsilon=0.2 --apply_reward=True --reward=1 --alpha=0 --epochs=1000 --seed=1 --model=e &
# )

# fixed energy, bigger max entropy

# (trap 'kill 0' SIGINT;
# python main.py --test=True --model=mop --path=data/en_ant_mop_r0_a1/en_ant_mop_r0_a1_s0/pyt_save/model.pt 
# python main.py --test=True --model=mop --path=data/en_ant_mop_r0_a1/en_ant_mop_r0_a1_s1/pyt_save/model.pt
# python main.py --test=True --model=e --epsilon=0.05 --path=data/en_ant_eg_r1_a0_eps_005/en_ant_eg_r1_a0_eps_005_s0/pyt_save/model.pt
# python main.py --test=True --model=e --epsilon=0.05 --path=data/en_ant_eg_r1_a0_eps_005/en_ant_eg_r1_a0_eps_005_s1/pyt_save/model.pt
# python main.py --test=True --model=e --epsilon=0.03 --path=data/en_ant_eg_r1_a0_eps_003/en_ant_eg_r1_a0_eps_003_s0/pyt_save/model.pt
# python main.py --test=True --model=e --epsilon=0.03 --path=data/en_ant_eg_r1_a0_eps_003/en_ant_eg_r1_a0_eps_003_s1/pyt_save/model.pt
# python main.py --test=True --model=e --epsilon=0.07 --path=data/en_ant_eg_r1_a0_eps_007/en_ant_eg_r1_a0_eps_007_s0/pyt_save/model.pt
# python main.py --test=True --model=e --epsilon=0.07 --path=data/en_ant_eg_r1_a0_eps_007/en_ant_eg_r1_a0_eps_007_s1/pyt_save/model.pt
# )



# (trap 'kill 0' SIGINT;
# python main.py --exp_name="en_ran_init_ant_a_05_mop_r0_a1" --apply_reward=True --reward=0 --alpha=1 --epochs=1000 --seed=0 --model=mop &
# python main.py --exp_name="en_ran_init_ant_a_05_mop_r0_a1" --apply_reward=True --reward=0 --alpha=1 --epochs=1000 --seed=1 --model=mop &
# python main.py --exp_name="en_ran_init_ant_a_05_eg_r1_a0_eps_015" --epsilon=0.15 --apply_reward=True --reward=1 --alpha=0 --epochs=1000 --seed=0 --model=e & 
# python main.py --exp_name="en_ran_init_ant_a_05_eg_r1_a0_eps_015" --epsilon=0.15 --apply_reward=True --reward=1 --alpha=0 --epochs=1000 --seed=1 --model=e &
# )

# (trap 'kill 0' SIGINT;
# python main.py --exp_name="en_ran_init_ant_1f_a2_800_200_1500_mop_r0_a1"                       --apply_reward=True --reward=0 --alpha=1 --epochs=1000 --seed=0 --model=mop &
# python main.py --exp_name="en_ran_init_ant_1f_a2_800_200_1500_mop_r0_a1"                       --apply_reward=True --reward=0 --alpha=1 --epochs=1000 --seed=1 --model=mop &
# python main.py --exp_name="en_ran_init_ant_1f_a2_800_200_1500_eg_r1_a0_eps_015" --epsilon=0.15 --apply_reward=True --reward=1 --alpha=0 --epochs=1000 --seed=0 --model=e   & 
# python main.py --exp_name="en_ran_init_ant_1f_a2_800_200_1500_eg_r1_a0_eps_015" --epsilon=0.15 --apply_reward=True --reward=1 --alpha=0 --epochs=1000 --seed=1 --model=e   &
# )


# python main.py --test=True --model=mop --path=data/en_ran_init_ant_a_05_mop_r0_a1/en_ran_init_ant_a_05_mop_r0_a1_s0/pyt_save/model.pt
# python main.py --test=True --model=mop --path=data/en_ran_init_ant_a_05_mop_r0_a1/en_ran_init_ant_a_05_mop_r0_a1_s1/pyt_save/model.pt
# python main.py --test=True --model=e   --path=data/en_ran_init_ant_a_05_eg_r1_a0_eps_015/en_ran_init_ant_a_05_eg_r1_a0_eps_015_s0/pyt_save/model.pt --epsilon=0.15
# python main.py --test=True --model=e   --path=data/en_ran_init_ant_a_05_eg_r1_a0_eps_015/en_ran_init_ant_a_05_eg_r1_a0_eps_015_s1/pyt_save/model.pt --epsilon=0.15



# python main.py --test=True --model=mop              --path=data/en_ran_init_ant_1f_a2_800_200_1500_mop_r0_a1/en_ran_init_ant_1f_a2_800_200_1500_mop_r0_a1_s0/pyt_save/model.pt
# python main.py --test=True --model=mop              --path=data/en_ran_init_ant_1f_a2_800_200_1500_mop_r0_a1/en_ran_init_ant_1f_a2_800_200_1500_mop_r0_a1_s1/pyt_save/model.pt
# python main.py --test=True --model=e --epsilon=0.15 --path=data/en_ran_init_ant_1f_a2_800_200_1500_eg_r1_a0_eps_015/en_ran_init_ant_1f_a2_800_200_1500_eg_r1_a0_eps_015_s1/pyt_save/model.pt 
# python main.py --test=True --model=e --epsilon=0.15 --path=data/en_ran_init_ant_1f_a2_800_200_1500_eg_r1_a0_eps_015/en_ran_init_ant_1f_a2_800_200_1500_eg_r1_a0_eps_015_s0/pyt_save/model.pt 



# (trap 'kill 0' SIGINT;
# python main.py --exp_name="en_ran_init_ant_1f_a2_200_25_400_mop_r0_a1"                       --apply_reward=True --reward=0 --alpha=1 --epochs=300 --seed=0 --model=mop &
# python main.py --exp_name="en_ran_init_ant_1f_a2_200_25_400_mop_r0_a1"                       --apply_reward=True --reward=0 --alpha=1 --epochs=300 --seed=1 --model=mop &
# python main.py --exp_name="en_ran_init_ant_1f_a2_200_25_400_eg_r1_a0_eps_005" --epsilon=0.05 --apply_reward=True --reward=1 --alpha=0 --epochs=300 --seed=0 --model=e   & 
# python main.py --exp_name="en_ran_init_ant_1f_a2_200_25_400_eg_r1_a0_eps_005" --epsilon=0.05 --apply_reward=True --reward=1 --alpha=0 --epochs=300 --seed=1 --model=e   &
# python main.py --exp_name="en_ran_init_ant_1f_a2_200_25_400_eg_r1_a0_eps_010" --epsilon=0.10 --apply_reward=True --reward=1 --alpha=0 --epochs=300 --seed=0 --model=e   &
# python main.py --exp_name="en_ran_init_ant_1f_a2_200_25_400_eg_r1_a0_eps_010" --epsilon=0.10 --apply_reward=True --reward=1 --alpha=0 --epochs=300 --seed=1 --model=e   &
# python main.py --exp_name="en_ran_init_ant_1f_a2_200_25_400_eg_r1_a0_eps_015" --epsilon=0.15 --apply_reward=True --reward=1 --alpha=0 --epochs=300 --seed=0 --model=e   &
# python main.py --exp_name="en_ran_init_ant_1f_a2_200_25_400_eg_r1_a0_eps_015" --epsilon=0.15 --apply_reward=True --reward=1 --alpha=0 --epochs=300 --seed=1 --model=e   &
# )

# (trap 'kill 0' SIGINT;
# python main.py --test=True --model=e --epsilon=0.05 --path=data/en_ran_init_ant_1f_a2_200_25_400_eg_r1_a0_eps_005/en_ran_init_ant_1f_a2_200_25_400_eg_r1_a0_eps_005_s0/pyt_save/model.pt &
# python main.py --test=True --model=e --epsilon=0.05 --path=data/en_ran_init_ant_1f_a2_200_25_400_eg_r1_a0_eps_005/en_ran_init_ant_1f_a2_200_25_400_eg_r1_a0_eps_005_s1/pyt_save/model.pt &
# python main.py --test=True --model=e --epsilon=0.10 --path=data/en_ran_init_ant_1f_a2_200_25_400_eg_r1_a0_eps_010/en_ran_init_ant_1f_a2_200_25_400_eg_r1_a0_eps_010_s0/pyt_save/model.pt &
# python main.py --test=True --model=e --epsilon=0.10 --path=data/en_ran_init_ant_1f_a2_200_25_400_eg_r1_a0_eps_010/en_ran_init_ant_1f_a2_200_25_400_eg_r1_a0_eps_010_s1/pyt_save/model.pt &
# python main.py --test=True --model=e --epsilon=0.15 --path=data/en_ran_init_ant_1f_a2_200_25_400_eg_r1_a0_eps_015/en_ran_init_ant_1f_a2_200_25_400_eg_r1_a0_eps_015_s0/pyt_save/model.pt &
# python main.py --test=True --model=e --epsilon=0.15 --path=data/en_ran_init_ant_1f_a2_200_25_400_eg_r1_a0_eps_015/en_ran_init_ant_1f_a2_200_25_400_eg_r1_a0_eps_015_s1/pyt_save/model.pt &
# python main.py --test=True --model=mop              --path=data/en_ran_init_ant_1f_a2_200_25_400_mop_r0_a1/en_ran_init_ant_1f_a2_200_25_400_mop_r0_a1_s0/pyt_save/model.pt &
# python main.py --test=True --model=mop              --path=data/en_ran_init_ant_1f_a2_200_25_400_mop_r0_a1/en_ran_init_ant_1f_a2_200_25_400_mop_r0_a1_s1/pyt_save/model.pt &
# )



# (trap 'kill 0' SIGINT;
# python main.py --exp_name="ant_mop_r0_a1"        --env="Ant-v4"                --apply_reward=True --reward=0 --alpha=1 --epochs=300 --seed=2 --model=mop &
# python main.py --exp_name="ant_mop_r0_a1"        --env="Ant-v4"                --apply_reward=True --reward=0 --alpha=1 --epochs=300 --seed=3 --model=mop &
# python main.py --exp_name="ant_mop_r0_a1"        --env="Ant-v4"                --apply_reward=True --reward=0 --alpha=1 --epochs=300 --seed=4 --model=mop &
# python main.py --exp_name="ant_eg_r1_a0_eps_010" --env="Ant-v4" --epsilon=0.10 --apply_reward=True --reward=1 --alpha=0 --epochs=300 --seed=2 --model=e   &
# python main.py --exp_name="ant_eg_r1_a0_eps_010" --env="Ant-v4" --epsilon=0.10 --apply_reward=True --reward=1 --alpha=0 --epochs=300 --seed=3 --model=e   &
# python main.py --exp_name="ant_eg_r1_a0_eps_010" --env="Ant-v4" --epsilon=0.10 --apply_reward=True --reward=1 --alpha=0 --epochs=300 --seed=4 --model=e   &
# )


# (trap 'kill 0' SIGINT;
# python main.py --exp_name="en_ran_init_ant_1f_a2_200_25_400_mop_r0_a1"                       --apply_reward=True --reward=0 --alpha=1 --epochs=300 --seed=2 --model=mop &
# python main.py --exp_name="en_ran_init_ant_1f_a2_200_25_400_mop_r0_a1"                       --apply_reward=True --reward=0 --alpha=1 --epochs=300 --seed=3 --model=mop &
# python main.py --exp_name="en_ran_init_ant_1f_a2_200_25_400_mop_r0_a1"                       --apply_reward=True --reward=0 --alpha=1 --epochs=300 --seed=4 --model=mop &
# python main.py --exp_name="en_ran_init_ant_1f_a2_200_25_400_eg_r1_a0_eps_005" --epsilon=0.05 --apply_reward=True --reward=1 --alpha=0 --epochs=300 --seed=2 --model=e   & 
# python main.py --exp_name="en_ran_init_ant_1f_a2_200_25_400_eg_r1_a0_eps_005" --epsilon=0.05 --apply_reward=True --reward=1 --alpha=0 --epochs=300 --seed=3 --model=e   &
# python main.py --exp_name="en_ran_init_ant_1f_a2_200_25_400_eg_r1_a0_eps_005" --epsilon=0.05 --apply_reward=True --reward=1 --alpha=0 --epochs=300 --seed=4 --model=e   &
# )

# (trap 'kill 0' SIGINT;
# python main.py --test=True --model=e --epsilon=0.05 --path=data/en_ran_init_ant_1f_a2_200_25_400_eg_r1_a0_eps_005/en_ran_init_ant_1f_a2_200_25_400_eg_r1_a0_eps_005_s2/pyt_save/model.pt &
# python main.py --test=True --model=e --epsilon=0.05 --path=data/en_ran_init_ant_1f_a2_200_25_400_eg_r1_a0_eps_005/en_ran_init_ant_1f_a2_200_25_400_eg_r1_a0_eps_005_s3/pyt_save/model.pt &
# python main.py --test=True --model=e --epsilon=0.05 --path=data/en_ran_init_ant_1f_a2_200_25_400_eg_r1_a0_eps_005/en_ran_init_ant_1f_a2_200_25_400_eg_r1_a0_eps_005_s4/pyt_save/model.pt &
# python main.py --test=True --model=mop              --path=data/en_ran_init_ant_1f_a2_200_25_400_mop_r0_a1/en_ran_init_ant_1f_a2_200_25_400_mop_r0_a1_s2/pyt_save/model.pt &
# python main.py --test=True --model=mop              --path=data/en_ran_init_ant_1f_a2_200_25_400_mop_r0_a1/en_ran_init_ant_1f_a2_200_25_400_mop_r0_a1_s3/pyt_save/model.pt &
# python main.py --test=True --model=mop              --path=data/en_ran_init_ant_1f_a2_200_25_400_mop_r0_a1/en_ran_init_ant_1f_a2_200_25_400_mop_r0_a1_s4/pyt_save/model.pt &
# )



# (trap 'kill 0' SIGINT;
# python main.py --test=True --env="Ant-v4" --model=e --epsilon=0.05 --path=data/ant_eg_r1_a0_eps_005/ant_eg_r1_a0_eps_005_s0/pyt_save/model.pt &
# python main.py --test=True --env="Ant-v4" --model=e --epsilon=0.05 --path=data/ant_eg_r1_a0_eps_005/ant_eg_r1_a0_eps_005_s1/pyt_save/model.pt &
# python main.py --test=True --env="Ant-v4" --model=e --epsilon=0.10 --path=data/ant_eg_r1_a0_eps_010/ant_eg_r1_a0_eps_010_s0/pyt_save/model.pt &
# python main.py --test=True --env="Ant-v4" --model=e --epsilon=0.10 --path=data/ant_eg_r1_a0_eps_010/ant_eg_r1_a0_eps_010_s1/pyt_save/model.pt &
# python main.py --test=True --env="Ant-v4" --model=e --epsilon=0.15 --path=data/ant_eg_r1_a0_eps_015/ant_eg_r1_a0_eps_015_s0/pyt_save/model.pt &
# python main.py --test=True --env="Ant-v4" --model=e --epsilon=0.15 --path=data/ant_eg_r1_a0_eps_015/ant_eg_r1_a0_eps_015_s1/pyt_save/model.pt &
# python main.py --test=True --env="Ant-v4" --model=mop              --path=data/ant_mop_r0_a1/ant_mop_r0_a1_s0/pyt_save/model.pt &
# python main.py --test=True --env="Ant-v4" --model=mop              --path=data/ant_mop_r0_a1/ant_mop_r0_a1_s1/pyt_save/model.pt &
# )



# test

(trap 'kill 0' SIGINT;
python main.py --test=True --model=e --epsilon=0.05 --path=data/en_ran_init_ant_1f_a2_200_25_400_eg_r1_a0_eps_005/en_ran_init_ant_1f_a2_200_25_400_eg_r1_a0_eps_005_s2/pyt_save/model.pt &
python main.py --test=True --model=e --epsilon=0.05 --path=data/en_ran_init_ant_1f_a2_200_25_400_eg_r1_a0_eps_005/en_ran_init_ant_1f_a2_200_25_400_eg_r1_a0_eps_005_s3/pyt_save/model.pt &
python main.py --test=True --model=e --epsilon=0.05 --path=data/en_ran_init_ant_1f_a2_200_25_400_eg_r1_a0_eps_005/en_ran_init_ant_1f_a2_200_25_400_eg_r1_a0_eps_005_s4/pyt_save/model.pt &
python main.py --test=True --model=mop              --path=data/en_ran_init_ant_1f_a2_200_25_400_mop_r0_a1/en_ran_init_ant_1f_a2_200_25_400_mop_r0_a1_s2/pyt_save/model.pt &
python main.py --test=True --model=mop              --path=data/en_ran_init_ant_1f_a2_200_25_400_mop_r0_a1/en_ran_init_ant_1f_a2_200_25_400_mop_r0_a1_s3/pyt_save/model.pt &
python main.py --test=True --model=mop              --path=data/en_ran_init_ant_1f_a2_200_25_400_mop_r0_a1/en_ran_init_ant_1f_a2_200_25_400_mop_r0_a1_s4/pyt_save/model.pt &
)


(trap 'kill 0' SIGINT;
python main.py --test=True --env="Ant-v4" --model=e --epsilon=0.10 --path=data/ant_eg_r1_a0_eps_010/ant_eg_r1_a0_eps_010_s2/pyt_save/model.pt &
python main.py --test=True --env="Ant-v4" --model=e --epsilon=0.10 --path=data/ant_eg_r1_a0_eps_010/ant_eg_r1_a0_eps_010_s3/pyt_save/model.pt &
python main.py --test=True --env="Ant-v4" --model=e --epsilon=0.10 --path=data/ant_eg_r1_a0_eps_010/ant_eg_r1_a0_eps_010_s4/pyt_save/model.pt &
python main.py --test=True --env="Ant-v4" --model=mop              --path=data/ant_mop_r0_a1/ant_mop_r0_a1_s2/pyt_save/model.pt &
python main.py --test=True --env="Ant-v4" --model=mop              --path=data/ant_mop_r0_a1/ant_mop_r0_a1_s3/pyt_save/model.pt &
python main.py --test=True --env="Ant-v4" --model=mop              --path=data/ant_mop_r0_a1/ant_mop_r0_a1_s4/pyt_save/model.pt &
)

python main.py --energy=True --test=True --model=e   --epsilon=0.05 --path=data/main_data/en_ran_init_ant_1f_a2_200_25_400_eg_r1_a0_eps_005/en_ran_init_ant_1f_a2_200_25_400_eg_r1_a0_eps_005_s4/pyt_save/model.pt --env="Ant-v4"
python main.py               --test=True --model=mop                --path=data/main_data/ant_mop_r0_a1/ant_mop_r0_a1_s4/pyt_save/model.pt   --env="Ant-v4"
python main.py               --test=True --model=e --epsilon=0.10   --path=data/main_data/ant_eg_r1_a0_eps_010/ant_eg_r1_a0_eps_010_s4/pyt_save/model.pt --env="Ant-v4"