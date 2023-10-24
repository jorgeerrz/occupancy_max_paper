from agent import *
from envs import *
import argparse

def get_env_fn(energy):
    if energy:
        return lambda : CustomAntEnv(healthy_z_range=(0.3, 1.0), width=1280, height=720)
    else:
        return lambda : gym.make(args.env, healthy_z_range=(0.3, 1.0), width=1280, height=720)


def get_test_env(energy, render_mode=None):
    if energy:
        return CustomAntEnv(healthy_z_range=(0.3, 1.0), render_mode=render_mode, width=1280, height=720)
    else:
        return CustomAntEnvNoEnergy(healthy_z_range=(0.3, 1.0), render_mode=render_mode, width=1280, height=720) #gym.make(args.env, xml_file="/home/yamen/data/study/phd/code/MOP/ant.xml", healthy_z_range=(0.3, 1.0), render_mode=render_mode)


if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Ant-v4')
    parser.add_argument('--energy', type=bool, default=False)
    parser.add_argument('--path', type=str)
    parser.add_argument('--exp_name', type=str, default='mop')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--test', type=bool, default=False)
    parser.add_argument('--apply_reward', type=bool, default=False)
    parser.add_argument('--reward', type=int, default=0)
    parser.add_argument('--epsilon', type=float, default=0.2)
    parser.add_argument('--model', type=str)
    args = parser.parse_args()
    
    torch.set_num_threads(torch.get_num_threads())
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)
    
    energy=args.energy
    print(energy)
    env_fn = get_env_fn(energy)
    if args.model == "mop":
        model = MOP(exp_name=args.exp_name, env_fn=env_fn, actor_critic=MLPActorCritic,
            ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), alpha=args.alpha,
            gamma=0.999, seed=args.seed, epochs=args.epochs, steps_per_epoch=10000, start_steps=2000*10, lr=args.lr,
            logger_kwargs=logger_kwargs, max_ep_len=5000, update_every=16)
        
    elif args.model == "e":
        model = EGready(exp_name=args.exp_name, env_fn=env_fn, actor_critic=MLPActorCritic,
            ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), alpha=args.alpha,
            gamma=0.999, seed=args.seed, epochs=args.epochs, steps_per_epoch=10000, start_steps=2000*10, lr=args.lr,
            logger_kwargs=logger_kwargs, max_ep_len=5000, update_every=16, epsilon=args.epsilon)
    else:
        raise "Model not known"
        
    
    if args.apply_reward:
        model.reward_function = lambda x: int(args.reward)
        
    if args.test:
        for i in range(1):
            # mode = "rgb_array"
            mode= "human"
            env = get_test_env(energy, mode) 
            model.ac = torch.load(args.path).to("cuda")
            test_no_log_file(env, model, args.path, render_mode=mode, max_eps_length=4000, deterministic=False)
            
            time.sleep(2)
    else:
        model.train()
        

