import torch
from model.AC import AC


def main():
    ac_model = AC(
        learn_table_size=4000,
        epsilon_start=0.9,
        epsilon_end=0.1,
        batch_size=10,
        foresight=0.7,
        actor_lr=0.01,
        critic_lr=0.01,
        action_space=1,
        state_space=4,
        epochs=1000,
        action_size=4
    )

    # 模拟一些环境状态，每个状态有4个特征
    states = torch.rand(5, 4)

    # # 模拟使用AC模型进行决策
    for i in range(5):
        state = states[i]
        action = ac_model.act(state)
        print(f"State: {state}")
        print(f"Action: {action}")

    # 一个简单的学习循环
    for _ in range(10):
        states = torch.rand(10, 4)
        actions = torch.randint(0, 4, (10, 1))
        rewards = torch.rand(10, 1)
        # 封装一些训练数据
        for i in range(10):
            ac_model.get_experience(states[i], actions[i], rewards[i])
        result = ac_model.learn()
        if result is not None:
            a, b = result
            print(f"Actor loss: {a}, Critic loss: {b}")


if __name__ == "__main__":
    main()
