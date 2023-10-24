# MOP

---


Here you'll discover our implementation of MOP in continuous space, primarily focused on the `Ant-v4` environment from MuJoCo's gym.

## Parameters

Here are the parameters you can configure when running the main script:

- **--env**: Environment that the model will interact with. Default is `'Ant-v4'`.

- **--energy**: Boolean flag to decide if the simulation with energy should be enabled. Default is `False`.

- **--path**: Path for saving or loading models. There is no default value and it needs to be provided.

- **--exp_name**: Name of the experiment. Default is `'mop'`.

- **--hid**: Size of the hidden layer in the neural network. Default value is `256`.

- **--l**: Number of layers in the neural network. Default value is `2`.

- **--lr**: Learning rate for the optimizer. Default value is `3e-4`.

- **--gamma**: Discount factor for the reward. Default value is `0.99`.

- **--seed**: Seed for random number generator. This ensures reproducibility. Default value is `0`.

- **--epochs**: Number of training epochs. Default value is `50`.

- **--alpha**: Alpha value. `1` for MOP and `0` for E-greedy.

- **--test**: Boolean flag to decide if the model should be run in test mode. Default is `False`.

- **--apply_reward**: Boolean flag to decide if a custom reward should be applied. Default is `False`.

- **--reward**: Value of the custom reward. It's applied only if `--apply_reward` is set to `True`. Default value is `0`.

- **--epsilon**: Epsilon value for E-greedy strategy. Default value is `0.2`.

- **--model**: Name or path of the pre-trained model to be used in testing mode.

---

## How to Run

Certainly! Here's a suggested addition for your README:

---

### Docker Support

For ease of deployment and ensuring consistent environments, we've provided a `docker/Dockerfile`. You can build and run the project in a Docker container, eliminating potential issues with dependencies or system configurations.

### Training

```bash
# MOP example
python main.py --exp_name="exp_mop"  --apply_reward=True --reward=0 --alpha=1 --epochs=300 --seed=2 --model=mop

# E-greedy example
python main.py --exp_name="exp_eg" --epsilon=0.05 --apply_reward=True --reward=1 --alpha=0 --epochs=300 --seed=2 --model=e

```

### Testing

```bash
# MOP example
python main.py --test=True --env="Ant-v4" --model=mop  --path=path/to/model.pt

# E-greedy example
python main.py --test=True --env="Ant-v4" --model=e --epsilon=0.10 --path=path/to/model.pt
```

---

## Experimental Data

In the `data` directory, you can find the logged data a part of our experiments (with some testing samples), which are extensively discussed and analyzed in our paper. This data provides a comprehensive view of our results and serves as an empirical foundation for the conclusions drawn in our research.

---

## Acknowledgements

This work makes use of several foundational resources:

- [Spinning Up in Deep RL](https://spinningup.openai.com/en/latest/) by OpenAI: We are thankful for the extensive documentation and resources on deep reinforcement learning provided by the OpenAI team.
  
- [Gym](https://github.com/openai/gym/tree/master) and [MuJoCo-Py](https://github.com/openai/mujoco-py) repositories: Our experiments and implementations heavily rely on environments from the MuJoCo plugin for OpenAI Gym. We appreciate the effort put into creating and maintaining this valuable toolset for the RL community.
---
