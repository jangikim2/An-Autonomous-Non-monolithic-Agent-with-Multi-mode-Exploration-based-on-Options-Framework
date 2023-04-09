"""
HIRO training process
"""
import os
##os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import datetime
from torch import Tensor
import copy
import torch
from torch.nn import functional
import numpy as np
import wandb
from utils import get_env, get_target_position, log_video_hrl, ParamDict, LoggerTrigger, TimeLogger, print_cmd_hint
from network import ActorLow, ActorHigh, CriticLow, CriticHigh
from experience_buffer import ExperienceBufferLow, ExperienceBufferHigh

##############################################
from ppo.PPO import PPO
from ppo.PPOM import PPOM
from ppo.config import get_params
from ppo.running_mean_std import RunningMeanStd, apply_normalizer
##############################################

def save_evaluate_utils(step, actor_l, actor_h, params, file_path=None, file_name=None):
    if file_name is None:
        time = datetime.datetime.now()
        file_name = "evalutils-hiro-{}_{}-it({})-[{}].tar".format(params.env_name.lower(), params.prefix, step, time)
    if file_path is None:
        file_path = os.path.join(".", "save", "model", file_name)
    print("\n    > saving evaluation utils...")
    torch.save({
        'step': step,
        'actor_l': actor_l.state_dict(),
        'actor_h': actor_h.state_dict(),
    }, file_path)
    print("    > saved evaluation utils to: {}\n".format(file_path))


def save_checkpoint(step, actor_l, critic_l, actor_optimizer_l, critic_optimizer_l, exp_l, actor_h, critic_h, actor_optimizer_h, critic_optimizer_h, exp_h, logger, params, file_path=None, file_name=None):
    pass
    '''
    if file_name is None:
        time = datetime.datetime.now()
        file_name = "checkpoint-hiro-{}_{}-it({})-[{}].tar".format(params.env_name.lower(), params.prefix, step, time)
    if file_path is None:
        file_path = os.path.join(".", "save", "model", file_name)
    print("\n    > saving training checkpoint...")
    torch.save({
        'step': step,
        'params': params,
        'logger': logger,
        'actor_l': actor_l.state_dict(),
        'critic_l': critic_l.state_dict(),
        'actor_optimizer_l': actor_optimizer_l.state_dict(),
        'critic_optimizer_l': critic_optimizer_l.state_dict(),
        'exp_l': exp_l,
        'actor_h': actor_h.state_dict(),
        'critic_h': critic_h.state_dict(),
        'actor_optimizer_h': actor_optimizer_h.state_dict(),
        'critic_optimizer_h': critic_optimizer_h.state_dict(),
        'exp_h': exp_h
    }, file_path)
    print("    > saved checkpoint to: {}\n".format(file_path))
    '''

def load_checkpoint(file_name):
    try:
        # load checkpoint file
        print("\n    > loading training checkpoint...")
        file_path = os.path.join(".", "save", "model", file_name)
        checkpoint = torch.load(file_path)
        print("\n    > checkpoint file loaded! parsing data...")
        params = checkpoint['params']
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if params.use_cuda else "cpu"
        # load utils
        policy_params = params.policy_params
        state_dim = params.state_dim
        goal_dim = params.goal_dim
        action_dim = params.action_dim
        max_action = policy_params.max_action
        max_goal = policy_params.max_goal
        # initialize rl components
        actor_eval_l = ActorLow(state_dim, goal_dim, action_dim, max_action).to(device)
        actor_optimizer_l = torch.optim.Adam(actor_eval_l.parameters(), lr=policy_params.actor_lr)
        critic_eval_l = CriticLow(state_dim, goal_dim, action_dim).to(device)
        critic_optimizer_l = torch.optim.Adam(critic_eval_l.parameters(), lr=policy_params.critic_lr)
        actor_eval_h = ActorHigh(state_dim, goal_dim, max_goal, device).to(device)
        actor_optimizer_h = torch.optim.Adam(actor_eval_h.parameters(), lr=policy_params.actor_lr)
        critic_eval_h = CriticHigh(state_dim, goal_dim).to(device)
        critic_optimizer_h = torch.optim.Adam(critic_eval_h.parameters(), lr=policy_params.critic_lr)
        # unpack checkpoint object
        step = checkpoint['step'] + 1
        logger = checkpoint['logger']
        #
        actor_eval_l.load_state_dict(checkpoint['actor_l'])
        critic_eval_l.load_state_dict(checkpoint['critic_l'])
        actor_optimizer_l.load_state_dict((checkpoint['actor_optimizer_l']))
        critic_optimizer_l.load_state_dict(checkpoint['critic_optimizer_l'])
        experience_buffer_l = checkpoint['exp_l']
        #
        actor_eval_h.load_state_dict(checkpoint['actor_h'])
        critic_eval_h.load_state_dict(checkpoint['critic_h'])
        actor_optimizer_h.load_state_dict((checkpoint['actor_optimizer_h']))
        critic_optimizer_h.load_state_dict(checkpoint['critic_optimizer_h'])
        experience_buffer_h = checkpoint['exp_h']
        #
        actor_target_l = copy.deepcopy(actor_eval_l).to(device)
        critic_target_l = copy.deepcopy(critic_eval_l).to(device)
        actor_target_h = copy.deepcopy(actor_eval_h).to(device)
        critic_target_h = copy.deepcopy(critic_eval_h).to(device)
        #
        actor_eval_l.train(), actor_target_l.train(), critic_eval_l.train(), critic_target_l.train()
        actor_eval_h.train(), actor_target_h.train(), critic_eval_h.train(), critic_target_h.train()
        print("    > checkpoint resume success!")
    except Exception as e:
        print(e)
    return [step, params, device, logger,
            actor_eval_l, actor_target_l, critic_eval_l, critic_target_l, actor_optimizer_l, critic_optimizer_l, experience_buffer_l,
            actor_eval_h, actor_target_h, critic_eval_h, critic_target_h, actor_optimizer_h, critic_optimizer_h, experience_buffer_h]


def initialize_params(params, device):
    policy_params = params.policy_params
    env_name = params.env_name
    max_goal = Tensor(policy_params.max_goal).to(device)
    action_dim = params.action_dim
    goal_dim = params.goal_dim
    max_action = policy_params.max_action
    expl_noise_std_l = policy_params.expl_noise_std_l
    expl_noise_std_h = policy_params.expl_noise_std_h
    c = policy_params.c
    episode_len = policy_params.episode_len
    max_timestep = policy_params.max_timestep
    start_timestep = policy_params.start_timestep
    batch_size = policy_params.batch_size
    log_interval = params.log_interval
    checkpoint_interval = params.checkpoint_interval
    evaluation_interval = params.evaluation_interval
    save_video = params.save_video
    video_interval = params.video_interval
    env = get_env(params.env_name)
    video_log_trigger = LoggerTrigger(start_ind=policy_params.start_timestep)
    state_print_trigger = LoggerTrigger(start_ind=policy_params.start_timestep)
    checkpoint_logger = LoggerTrigger(start_ind=policy_params.start_timestep, first_log=False)
    evalutil_logger = LoggerTrigger(start_ind=policy_params.start_timestep, first_log=False)
    time_logger = TimeLogger()
    return [policy_params, env_name, max_goal, action_dim, goal_dim, max_action, expl_noise_std_l, expl_noise_std_h,
            c, episode_len, max_timestep, start_timestep, batch_size,
            log_interval, checkpoint_interval, evaluation_interval, save_video, video_interval, env, video_log_trigger, state_print_trigger, checkpoint_logger, evalutil_logger, time_logger]


def initialize_params_checkpoint(params, device):
    policy_params = params.policy_params
    env_name = params.env_name
    max_goal = Tensor(policy_params.max_goal).to(device)
    action_dim = params.action_dim
    goal_dim = params.goal_dim
    max_action = policy_params.max_action
    expl_noise_std_l = policy_params.expl_noise_std_l
    expl_noise_std_h = policy_params.expl_noise_std_h
    c = policy_params.c
    episode_len = policy_params.episode_len
    max_timestep = policy_params.max_timestep
    start_timestep = policy_params.start_timestep
    batch_size = policy_params.batch_size
    log_interval = params.log_interval
    checkpoint_interval = params.checkpoint_interval
    save_video = params.save_video
    video_interval = params.video_interval
    env = get_env(params.env_name)
    return [policy_params, env_name, max_goal, action_dim, goal_dim, max_action, expl_noise_std_l, expl_noise_std_h,
            c, episode_len, max_timestep, start_timestep, batch_size,
            log_interval, checkpoint_interval, save_video, video_interval, env]


def record_logger(args, option, step):
    if option == "inter_loss":
        target_q_l, critic_loss_l, actor_loss_l, target_q_h, critic_loss_h, actor_loss_h = args[:]
        ##target_q_l, critic_loss_l, actor_loss_l = args[:]
        if target_q_l is not None: wandb.log({'target_q low': torch.mean(target_q_l).squeeze()}, step=step)
        if critic_loss_l is not None: wandb.log({'critic_loss low': torch.mean(critic_loss_l).squeeze()}, step=step)
        if actor_loss_l is not None: wandb.log({'actor_loss low': torch.mean(actor_loss_l).squeeze()}, step=step)
        if target_q_h is not None: wandb.log({'target_q high': torch.mean(target_q_h).squeeze()}, step=step)
        if critic_loss_h is not None: wandb.log({'critic_loss high': torch.mean(critic_loss_h).squeeze()}, step=step)
        if actor_loss_h is not None: wandb.log({'actor_loss high': torch.mean(actor_loss_h).squeeze()}, step=step)
    elif option == "reward":
        episode_reward_l, episode_reward_h = args[:]
        wandb.log({'episode reward low': episode_reward_l}, step=step)
        wandb.log({'episode reward high': episode_reward_h}, step=step)
    elif option == "ppo_high":
        loss, surr_loss, value_est_loss, entropy_loss = args[:]
        wandb.log({'ppo_high_total_loss': loss}, step=step)
        wandb.log({'ppo_high_surr_loss': surr_loss}, step=step)
        wandb.log({'ppo_high_value_est_loss': value_est_loss}, step=step)
        wandb.log({'ppo_high_entropy_loss': entropy_loss}, step=step)
    elif option == "ppom_high":
        ##loss, surr_loss, value_est_loss, entropy_loss, action_m = args[:]
        ###loss, surr_loss, value_est_loss, entropy_loss, on_policy_count, off_policy_count, random_policy_count = args[:]
        ##loss, surr_loss, value_est_loss, entropy_loss, off_policy_count, random_policy_count = args[:]
        off_policy_count, random_policy_count = args[:]
        #####wandb.log({'ppom_high_total_loss': loss}, step=step)
        #####wandb.log({'ppom_high_surr_loss': surr_loss}, step=step)
        #####wandb.log({'ppom_high_value_est_loss': value_est_loss}, step=step)
        #####wandb.log({'ppom_high_entropy_loss': entropy_loss}, step=step)
        ##wandb.log({'ppom_high_action_m': action_m}, step=step)
        #############wandb.log({'ppom_high_on_policy_count': on_policy_count}, step=step)
        wandb.log({'ppom_high_off_policy_count': off_policy_count}, step=step)
        wandb.log({'ppom_high_random_policy_count': random_policy_count}, step=step)
    elif option == "only_h_values_variance":
        value_h_promise_discrepancy = args[:]
        wandb.log({'value_h_promise_discrepancy': value_h_promise_discrepancy}, step=step)
    elif option == "values_variance":
        ##value_promise_discrepancy, value_m_promise_discrepancy, value_h_promise_discrepancy, value_l_promise_discrepancy, \
        ##state_values_var, state_m_values_var, q_target_h_var, q_target_l_var = args[:]
        value_h_promise_discrepancy, value_l_promise_discrepancy, \
        q_target_h_var, q_target_l_var = args[:]
        ##wandb.log({'value_promise_discrepancy': value_promise_discrepancy}, step=step)
        ##wandb.log({'value_m_promise_discrepancy': value_m_promise_discrepancy}, step=step)
        wandb.log({'value_h_promise_discrepancy': value_h_promise_discrepancy}, step=step)
        wandb.log({'value_l_promise_discrepancy': value_l_promise_discrepancy}, step=step)
        ##wandb.log({'state_values_var': state_values_var}, step=step)
        ##wandb.log({'state_m_values_var': state_m_values_var}, step=step)
        wandb.log({'q_target_h_var': q_target_h_var}, step=step)
        wandb.log({'q_target_l_var': q_target_l_var}, step=step)
    elif option == "success_rate":
        success_rate = args[0]
        wandb.log({'success rate': success_rate}, step=step)
    elif option == "higher_success_rate":
        higher_success_rate = args[0]
        wandb.log({'higher success rate': higher_success_rate}, step=step)
    elif option == "goal_diff_norm":
        goal_diff_norm = args[0]
        wandb.log({'goal diff norm': goal_diff_norm}, step=step)
    elif option == "pro_entropy_m":
        on_probability_mean, on_entropy_mean, on_state_value_mean, \
        off_probability_mean, off_entropy_mean, off_state_value_mean, \
        random_probability_mean, random_entropy_mean, random_state_value_mean = args[:]
        wandb.log({'on_probability_mean': on_probability_mean}, step=step)
        wandb.log({'on_entropy_mean': on_entropy_mean}, step=step)
        wandb.log({'on_state_value_mean': on_state_value_mean}, step=step)
        wandb.log({'off_probability_mean': off_probability_mean}, step=step)
        wandb.log({'off_entropy_mean': off_entropy_mean}, step=step)
        wandb.log({'off_state_value_mean': off_state_value_mean}, step=step)
        wandb.log({'random_probability_mean': random_probability_mean}, step=step)
        wandb.log({'random_entropy_mean': random_entropy_mean}, step=step)
        wandb.log({'random_state_value_mean': random_state_value_mean}, step=step)


def create_rl_components(params, device):
    # function local utils
    policy_params = params.policy_params
    state_dim, goal_dim, action_dim = params.state_dim, params.goal_dim, params.action_dim
    max_goal = Tensor(policy_params.max_goal)
    # low-level
    step, episode_num_h = 0, 0
    actor_eval_l = ActorLow(state_dim, goal_dim, action_dim, policy_params.max_action).to(device)
    actor_target_l = copy.deepcopy(actor_eval_l).to(device)
    actor_optimizer_l = torch.optim.Adam(actor_eval_l.parameters(), lr=policy_params.actor_lr)
    critic_eval_l = CriticLow(state_dim, goal_dim, action_dim).to(device)
    critic_target_l = copy.deepcopy(critic_eval_l).to(device)
    critic_optimizer_l = torch.optim.Adam(critic_eval_l.parameters(), lr=policy_params.critic_lr)
    experience_buffer_l = ExperienceBufferLow(policy_params.max_timestep, state_dim, goal_dim, action_dim, params.use_cuda)
    # high-level
    actor_eval_h = ActorHigh(state_dim, goal_dim, max_goal, device).to(device)
    actor_target_h = copy.deepcopy(actor_eval_h).to(device)
    actor_optimizer_h = torch.optim.Adam(actor_eval_h.parameters(), lr=policy_params.actor_lr)
    critic_eval_h = CriticHigh(state_dim, goal_dim).to(device)
    critic_target_h = copy.deepcopy(critic_eval_h).to(device)
    critic_optimizer_h = torch.optim.Adam(critic_eval_h.parameters(), lr=policy_params.critic_lr)
    experience_buffer_h = ExperienceBufferHigh(int(policy_params.max_timestep / policy_params.c) + 1, state_dim, goal_dim, params.use_cuda)

    return [step, episode_num_h,
            actor_eval_l, actor_target_l, actor_optimizer_l, critic_eval_l, critic_target_l, critic_optimizer_l, experience_buffer_l,
            actor_eval_h, actor_target_h, actor_optimizer_h, critic_eval_h, critic_target_h, critic_optimizer_h, experience_buffer_h]


def h_function(state, goal, next_state, goal_dim):
    # return next goal
    return state[:goal_dim] + goal - next_state[:goal_dim]


def intrinsic_reward(state, goal, next_state):
    # low-level dense reward (L2 norm), provided by high-level policy
    return -torch.pow(sum(torch.pow(state + goal - next_state, 2)), 1 / 2)


def intrinsic_reward_simple(state, goal, next_state, goal_dim):
    # low-level dense reward (L2 norm), provided by high-level policy
    return -torch.pow(sum(torch.pow(state[:goal_dim] + goal - next_state[:3], 2)), 1 / 2)


def dense_reward(state, goal_dim, target=Tensor([0, 19, 0.5])):
    device = state.device
    target = target.to(device)
    l2_norm = torch.pow(sum(torch.pow(state[:goal_dim] - target, 2)), 1 / 2)
    return -l2_norm


def done_judge_low(goal):
    # define low-level success: same as high-level success (L2 norm < 5, paper B.2.2)
    print("done_judge_low_goal : ", goal)
    l2_norm = torch.pow(sum(torch.pow(goal, 2)), 1 / 2)
    # done = (l2_norm <= 5.)
    done = (l2_norm <= 1.5)
    print("done_judge_low_done : ", done)
    return Tensor([done])


def success_judge(state, goal_dim, target=Tensor([0, 19, 0.5])):
    location = state[:goal_dim]
    l2_norm = torch.pow(sum(torch.pow(location - target, 2)), 1 / 2)
    done = (l2_norm <= 5.)
    return Tensor([done])


def off_policy_correction(actor, action_sequence, state_sequence, goal_dim, goal, end_state, max_goal, device):
    # initialize
    action_sequence = torch.stack(action_sequence).to(device)
    state_sequence = torch.stack(state_sequence).to(device)
    max_goal = max_goal.cpu()
    # prepare candidates
    mean = (end_state - state_sequence[0])[:goal_dim].cpu()
    std = 0.5 * max_goal
    candidates = [torch.min(torch.max(Tensor(np.random.normal(loc=mean, scale=std, size=goal_dim).astype(np.float32)), -max_goal), max_goal) for _ in range(8)]
    candidates.append(mean)
    candidates.append(goal.cpu())
    # select maximal
    candidates = torch.stack(candidates).to(device)
    surr_prob = [-functional.mse_loss(action_sequence, actor(state_sequence, state_sequence[0][:goal_dim] + candidate - state_sequence[:, :goal_dim])) for candidate in candidates]
    index = int(np.argmax(surr_prob))
    updated = (index != 9)
    goal_hat = candidates[index]
    return goal_hat.cpu(), updated


def step_update_l(experience_buffer, batch_size, total_it, actor_eval, actor_target, critic_eval, critic_target, critic_optimizer, actor_optimizer, params):
    # initialize
    policy_params = params.policy_params
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if params.use_cuda else "cpu"
    total_it[0] += 1
    # sample mini-batch transitions
    state, goal, action, reward, next_state, next_goal, done = experience_buffer.sample(batch_size)
    with torch.no_grad():
        # select action according to policy and add clipped noise
        policy_noise = Tensor(np.random.normal(loc=0, scale=policy_params.policy_noise_std, size=params.action_dim).astype(np.float32) * policy_params.policy_noise_scale) \
            .clamp(-policy_params.policy_noise_clip, policy_params.policy_noise_clip).to(device)
        next_action = (actor_target(next_state, next_goal) + policy_noise).clamp(-policy_params.max_action, policy_params.max_action)
        # clipped double Q-learning
        q_target_1, q_target_2 = critic_target(next_state, next_goal, next_action)
        q_target = torch.min(q_target_1, q_target_2)
        y = policy_params.reward_scal_l * reward + (1 - done) * policy_params.discount * q_target
    # update critic q_evaluate
    q_eval_1, q_eval_2 = critic_eval(state, goal, action)
    critic_loss = functional.mse_loss(q_eval_1, y) + functional.mse_loss(q_eval_2, y)
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()
    # delayed policy update
    actor_loss = None
    if total_it[0] % policy_params.policy_freq == 0:
        # compute actor loss
        actor_loss = -critic_eval.q1(state, goal, actor_eval(state, goal)).mean()
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()
        # soft update: critic q_target
        for param_eval, param_target in zip(critic_eval.parameters(), critic_target.parameters()):
            param_target.data.copy_(policy_params.tau * param_eval.data + (1 - policy_params.tau) * param_target.data)
        for param_eval, param_target in zip(actor_eval.parameters(), actor_target.parameters()):
            param_target.data.copy_(policy_params.tau * param_eval.data + (1 - policy_params.tau) * param_target.data)
        actor_loss = actor_loss.detach()
    return y.detach(), critic_loss.detach(), actor_loss


def step_update_h(experience_buffer, batch_size, total_it, actor_eval, actor_target, critic_eval, critic_target, critic_optimizer, actor_optimizer, params):
    policy_params = params.policy_params
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if params.use_cuda else "cpu"
    max_goal = Tensor(policy_params.max_goal).to(device)
    # sample mini-batch transitions
    state_start, goal, reward, state_end, done = experience_buffer.sample(batch_size)
    with torch.no_grad():
        # select action according to policy and add clipped noise
        ##policy_noise = Tensor(np.random.normal(loc=0, scale=policy_params.policy_noise_std, size=params.goal_dim).astype(np.float32) * policy_params.policy_noise_scale) \
        ##    .clamp(-policy_params.policy_noise_clip, policy_params.policy_noise_clip).to(device)
        ##next_goal = torch.min(torch.max(actor_target(state_end) + policy_noise, -max_goal), max_goal)
        next_goal = torch.min(torch.max(actor_target(state_end), -max_goal), max_goal)
        # clipped double Q-learning
        q_target_1, q_target_2 = critic_target(state_end, next_goal)
        q_target = torch.min(q_target_1, q_target_2)
        y = policy_params.reward_scal_h * reward + (1 - done) * policy_params.discount * q_target
    # update critic q_evaluate
    q_eval_1, q_eval_2 = critic_eval(state_start, goal)
    critic_loss = functional.mse_loss(q_eval_1, y) + functional.mse_loss(q_eval_2, y)
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()
    # delayed policy updates
    actor_loss = None
    if int(total_it[0] / policy_params.c) % policy_params.policy_freq == 0:
        # compute actor loss
        actor_loss = -critic_eval.q1(state_start, actor_eval(state_start)).mean()
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()
        # soft update: critic q_target
        for param_eval, param_target in zip(critic_eval.parameters(), critic_target.parameters()):
            param_target.data.copy_(policy_params.tau * param_eval.data + (1 - policy_params.tau) * param_target.data)
        for param_eval, param_target in zip(actor_eval.parameters(), actor_target.parameters()):
            param_target.data.copy_(policy_params.tau * param_eval.data + (1 - policy_params.tau) * param_target.data)
        actor_loss = actor_loss.detach()
    return y.detach(), critic_loss.detach(), actor_loss


def evaluate(actor_l, actor_h, params, target_pos, device):
    policy_params = params.policy_params
    print("\n    > evaluating policies...")
    success_number = 0
    env = get_env(params.env_name)
    goal_dim = params.goal_dim
    for i in range(10):
        env.seed(policy_params.seed + i)
        for j in range(5):
            t = 0
            episode_len = policy_params.episode_len
            obs, done = Tensor(env.reset()).to(device), False
            goal = Tensor(torch.randn(goal_dim)).to(device)
            while not done and t < episode_len:
                t += 1
                action = actor_l(obs, goal).to(device)
                obs, _, _, _ = env.step(action.detach().cpu())
                obs = Tensor(obs).to(device)
                done = success_judge(obs, goal_dim, target_pos)
                goal = actor_h(obs)
            if done:
                success_number += 1
        print("        > evaluated {} episodes".format(i * 5 + j + 1))
    success_rate = success_number / 50
    print("    > finished evaluation, success rate: {}\n".format(success_rate))
    return success_rate

def evaluate_higher_policy(actor_l, actor_h, params, target_pos, device, max_goal):
    policy_params = params.policy_params
    print("\n    > evaluating policies...")
    success_number = 0
    env = get_env(params.env_name)
    goal_dim = params.goal_dim
    for i in range(10):
        env.seed(policy_params.seed + i)
        for j in range(5):
            t = 0
            episode_len = policy_params.episode_len
            obs, done = Tensor(env.reset()).to(device), False
            goal = Tensor(torch.randn(goal_dim)).to(device)
            while not done and t < episode_len:
                t += 1
                action = actor_l(obs, goal).to(device)
                obs, _, _, _ = env.step(action.detach().cpu())
                obs = Tensor(obs).to(device)
                done = success_judge(obs, goal_dim, target_pos)
                ##goal = actor_h(obs)
                ##action_m = ppom_agent.select_action(np.concatenate([next_state.cpu().numpy(), target_pos.cpu().numpy()]))
                next_goal = actor_h.select_action_without_buffering(np.concatenate([obs.cpu().numpy(), target_pos.cpu().numpy()]))
                next_goal = Tensor(next_goal).to(device)
                goal = torch.min(torch.max(next_goal, -max_goal), max_goal)
            if done:
                success_number += 1
        print("        > evaluated {} episodes".format(i * 5 + j + 1))
    success_rate = success_number / 50
    print("    > finished evaluation, success rate: {}\n".format(success_rate))
    return success_rate



def homeostasis(x_t, time_step, x_bar, x_squared_bar, x_plus_bar, rho, device):
    print("x_t : ", x_t)
    Tau = np.minimum(time_step, 100/rho)
    print("Tau : ", Tau)
    x_bar = (1 - 1/Tau) * x_bar + 1/Tau * x_t
    print("x_bar : ", x_bar)
    x_squared_bar = (1 - 1/Tau) * x_squared_bar + 1/Tau * ((x_t - x_bar) ** 2)
    print("x_squared_bar : ", x_squared_bar)
    x_plus = np.exp((x_t - x_bar)/ np.square(x_squared_bar))
    print("x_plus : ", x_plus)
    x_plus_bar = (1 - 1/Tau) * x_plus_bar + 1/Tau * x_plus
    ##y_t = torch.bernoulli(np.minimum(1, rho*x_plus/x_plus_bar))
    print("rho*x_plus/x_plus_bar : ", rho*x_plus/x_plus_bar)
    y_t_input = np.minimum(1, rho*x_plus/x_plus_bar)
    print("y_t_input : ", y_t_input)
    y_t = torch.bernoulli(torch.tensor(y_t_input, dtype=torch.float32).to(device))
    print("homeostasis_y_t : ", y_t.cpu().numpy())
    return x_bar, x_squared_bar, x_plus_bar, y_t


'''
class RunningMeanStd:
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    # -> It's indeed batch normalization. :D
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)


def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count
'''

def train(params):
    # 1. Initialization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if params.use_cuda else "cpu"
    if params.checkpoint is None:
        # > rl components
        [step, episode_num_h,
         actor_eval_l, actor_target_l, actor_optimizer_l, critic_eval_l, critic_target_l, critic_optimizer_l, experience_buffer_l,
         actor_eval_h, actor_target_h, actor_optimizer_h, critic_eval_h, critic_target_h, critic_optimizer_h, experience_buffer_h] = create_rl_components(params, device)
        # > running utils
        [policy_params, env_name, max_goal, action_dim, goal_dim, max_action, expl_noise_std_l, expl_noise_std_h,
         c, episode_len, max_timestep, start_timestep, batch_size,
         log_interval, checkpoint_interval, evaluation_interval, save_video, video_interval, env, video_log_trigger, state_print_trigger, checkpoint_logger, evalutil_logger, time_logger] = initialize_params(params, device)
    else:
        # > rl components
        prefix = params.prefix
        [step, params, device, [time_logger, state_print_trigger, video_log_trigger, checkpoint_logger, evalutil_logger, episode_num_h],
         actor_eval_l, actor_target_l, critic_eval_l, critic_target_l, actor_optimizer_l, critic_optimizer_l, experience_buffer_l,
         actor_eval_h, actor_target_h, critic_eval_h, critic_target_h, actor_optimizer_h, critic_optimizer_h, experience_buffer_h] = load_checkpoint(params.checkpoint)
        # > running utils
        [policy_params, env_name, max_goal, action_dim, goal_dim, max_action, expl_noise_std_l, expl_noise_std_h,
         c, episode_len, max_timestep, start_timestep, batch_size,
         log_interval, checkpoint_interval, evaluation_interval, save_video, video_interval, env] = initialize_params_checkpoint(params, device)
        params.prefix = prefix
    target_q_h, critic_loss_h, actor_loss_h = None, None, None
    target_pos = get_target_position(env_name).to(device)
    # 1.2 set seeds
    env.seed(policy_params.seed)
    torch.manual_seed(policy_params.seed)
    np.random.seed(policy_params.seed)

    #####################################################
    config = get_params()
    #####################################################
    ####### initialize environment hyperparameters ######
    has_continuous_action_space = True  # continuous action space; else discrete

    max_ep_len = 1000                   # max timesteps in one episode
    action_std = 0.6                    # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = 0.05        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = 0.1                # minimum action_std (stop decay after action_std <= min_action_std)
    action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num timesteps)
    #####################################################

    ################ PPO hyperparameters ################
    ##update_timestep = max_ep_len * 4      # update policy every n timesteps
    update_timestep = 30      # update policy every n timesteps
    ##K_epochs = 80               # update policy for K epochs in one PPO update
    K_epochs = 10               # update policy for K epochs in one PPO update

    eps_clip = 0.2          # clip parameter for PPO
    gamma = 0.99            # discount factor

    lr_actor = 0.0003       # learning rate for actor network
    lr_critic = 0.001       # learning rate for critic network

    random_seed = 0         # set random seed if required (0 = no random seed)
    #####################################################

    #################################################################################################################################
    # initialize a PPO agent
    ##ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)
    ppo_agent = PPO(params.state_dim, params.goal_dim, params.goal_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)

    #################################################################################################################################
    has_continuous_action_space_m = False  # discrete
    action_dim_m = 2
    c_m = 50
    ##update_timestep_m = 400      # update policy every n timesteps
    ##K_epochs = 80               # update policy for K epochs in one PPO update
    K_epochs_m = 10               # update policy for K epochs in one PPO update

    ppom_agent = PPOM(params.state_dim, action_dim_m, params.goal_dim, lr_actor, lr_critic, gamma, K_epochs_m, eps_clip, has_continuous_action_space_m, action_std)
    on_policy_count = 0
    off_policy_count = 0
    random_policy_count = 0
    temp_done_h = False

    ##action_m = 2
    action_m = 0

    temp_ppom_action = Tensor([5]).to(device)
    #################################################################################################################################

    #################################################################################################################################
    '''
    meta_ppo_value_rms = RunningMeanStd(shape=(1,))
    ppo_value_rms = RunningMeanStd(shape=(1,))
    value_rms = RunningMeanStd(shape=(1,))
    old_meta_ppo_value_rms = 0
    old_ppo_value_rms = 0
    old_value_rms = 0
    '''
    value_promise_discrepancy_normalizer = RunningMeanStd(shape=())
    value_m_promise_discrepancy_normalizer = RunningMeanStd(shape=())
    value_h_promise_discrepancy_normalizer = RunningMeanStd(shape=())
    value_l_promise_discrepancy_normalizer = RunningMeanStd(shape=())

    state_values_var_rms = RunningMeanStd(shape=())
    state_m_values_var_rms = RunningMeanStd(shape=())
    q_target_h_var_rms = RunningMeanStd(shape=())
    q_target_l_var_rms = RunningMeanStd(shape=())

    random_probability_rms = RunningMeanStd(shape=())
    on_probability_rms = RunningMeanStd(shape=())
    off_probability_rms = RunningMeanStd(shape=())

    random_entropy_rms = RunningMeanStd(shape=())
    on_entropy_rms = RunningMeanStd(shape=())
    off_entropy_rms = RunningMeanStd(shape=())

    random_state_value_rms = RunningMeanStd(shape=())
    on_state_value_rms = RunningMeanStd(shape=())
    off_state_value_rms = RunningMeanStd(shape=())

    state_values_list, state_m_values_list, q_target_h_list, q_target_l_list = [], [], [], []
    old_value, old_m_value, old_h_target_value, old_l_target_value = 0, 0, 0, 0
    value_h_reward, value_h_reward_m, value_l_reward = 0, 0, 0

    random_probability_list, on_probability_list, off_probability_list = [], [], []
    random_entropy_list, on_entropy_list, off_entropy_list = [], [], []
    random_state_value_list, on_state_value_list, off_state_value_list = [], [], []

    value_m_promise_discrepancy = 0
    episode_reward_h_m = 0
    episode_count_m = 0
    episode_done_count_m = 0
    m_flag = False
    ppo_flag = False
    goal_diff_norm = 0
    starting_random_ref = 250000
    #####starting_random_ref = 250

    value_h_promise_discrepancy = 0
    trigger_value_h_promise_discrepancy = 0
    previous_state_values, previous_state_m_values, previous_q_target_h, previous_q_target_l = 0, 0, 0, 0
    previous_episode_reward_h, previous_episode_reward_h_m = 0, 0
    gamma_tigeer = 1
    gamma_tigeer_m = 1
    gamma_tigeer_l = 1
    x_bar, x_squared_bar, x_plus_bar = 0, 1, 1
    target_rate_rho = 0.001  #### 0.1, 0.01, 0.001, 0.0001
    explore_fixed_steps = 10
    update_timestep_m = 40      # update policy every n timesteps
    explore_count = 0
    explore_flag = False
    exploit_count = 1
    #################################################################################################################################



    # 2. Training Algorithm (TD3)
    # 2.1 initialize
    print_cmd_hint(params=params, location='start_train')
    time_logger.time_spent()
    total_it = [0]
    higher_success_rate, success_rate, episode_reward_l, episode_reward_h, episode_reward, episode_num_l, episode_timestep_l, episode_timestep_h = 0, 0, 0, 0, 0, 0, 1, 1
    state = Tensor(env.reset()).to(device)
    goal = Tensor(torch.randn(goal_dim)).to(device)
    state_sequence, goal_sequence, action_sequence, intri_reward_sequence, reward_h_sequence = [], [], [], [], []
    # 2.2 training loop
    for t in range(step, max_timestep):
        # 2.2.1 sample action
        if t < start_timestep:
            action = env.action_space.sample()
        else:
            expl_noise_action = np.random.normal(loc=0, scale=expl_noise_std_l, size=action_dim).astype(np.float32)
            action = (actor_eval_l(state, goal).detach().cpu() + expl_noise_action).clamp(-max_action, max_action).squeeze()
        # 2.2.2 interact environment
        ##print("state : ", state)
        ##print("state : ", goal)
        ##print("low_action : ", action)
        next_state, _, _, info = env.step(action)
        # 2.2.3 compute step arguments
        reward_h = dense_reward(state, goal_dim, target=target_pos)
        done_h = success_judge(state, goal_dim, target_pos)
        next_state, action, reward_h, done_h = Tensor(next_state).to(device), Tensor(action), Tensor([reward_h]), Tensor([done_h])
        intri_reward = intrinsic_reward_simple(state, goal, next_state, goal_dim)
        next_goal = h_function(state, goal, next_state, goal_dim)
        ##print("goal_before : ", goal)
        done_l = done_judge_low(goal)
        # 2.2.4 collect low-level experience
        experience_buffer_l.add(state, goal, action, intri_reward, next_state, next_goal, done_l)
        # 2.2.5 record segment arguments
        state_sequence.append(state)
        action_sequence.append(action)
        intri_reward_sequence.append(intri_reward)
        goal_sequence.append(goal)
        reward_h_sequence.append(reward_h)
        # 2.2.6 update low-level segment reward
        episode_reward_l += intri_reward
        episode_reward_h += reward_h
        episode_reward += reward_h
        ############################################
        ##episode_reward_h_m += reward_h
        ##print("done_h.cpu().numpy() : ",done_h[0].cpu().numpy())
        ##if done_h[0].cpu().numpy() != 0:
        ##    episode_done_count_m += 1
        ##episode_count_m += 1
        ############################################
        # if continuous action space; then decay action std of ouput action distribution
        ##if has_continuous_action_space and time_step % action_std_decay_freq == 0:

        ##if has_continuous_action_space and t % action_std_decay_freq == 0:
        ##    ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

        # if continuous action space; then decay action std of ouput action distribution
        ##if has_continuous_action_space and time_step % action_std_decay_freq == 0:
        ##if has_continuous_action_space_m and t % action_std_decay_freq == 0:
        ##    ppom_agent.decay_action_std(action_std_decay_rate, min_action_std)

        if (t + 1) % c == 0 and t > 0:
            # 2.2.7 sample goal

            if t < start_timestep:
                next_goal = (torch.randn_like(goal) * max_goal)
                next_goal = torch.min(torch.max(next_goal, -max_goal), max_goal)
            else:
                t_step = t-start_timestep
                if t_step <= starting_random_ref:
                    print("t_step_1")
                    action_m = 0
                else:
                    print("t_step_2")
                    if explore_flag == True :
                        if explore_count % explore_fixed_steps == 0 :
                            print("if explore_fixed_steps % explore_count == 0 :")
                            explore_count = 0
                            x_bar, x_squared_bar, x_plus_bar = 0, 1, 1
                            explore_flag = False
                        else:
                            print("if explore_fixed_steps % explore_count == 0 _ else:")
                            explore_count += 1

                    if explore_flag == False:
                        print("if explore_flag == False:")
                        if exploit_count % update_timestep == 0:
                            next_goal_t = torch.min(torch.max(actor_target_h(next_state), -max_goal), max_goal)
                            q_target_1, q_target_2 = critic_target_h(next_state, next_goal_t)
                            q_target_h = torch.min(q_target_1, q_target_2)
                            q_target_h_n = q_target_h.detach().cpu().numpy()[0][0]

                            value_h_promise_discrepancy = old_h_target_value - value_h_reward - (gamma_tigeer * q_target_h_n)
                            abs_value_h_promise_discrepancy = np.absolute(value_h_promise_discrepancy)
                            ##value_h_promise_discrepancy = apply_normalizer(abs_value_h_promise_discrepancy, value_h_promise_discrepancy_normalizer)

                            ##record_logger(args=[value_h_promise_discrepancy], option='only_h_values_variance', step=t-start_timestep)
                            record_logger(args=[abs_value_h_promise_discrepancy], option='only_h_values_variance', step=t-start_timestep)
                            old_h_target_value = q_target_h_n
                            value_h_reward = 0
                            gamma_tigeer = 1

                            ##homeostasis(x_t, time_step, x_bar, x_squared_bar, x_plus_bar, rho, device)
                            x_bar, x_squared_bar, x_plus_bar, y_t = homeostasis(abs_value_h_promise_discrepancy, exploit_count, x_bar, x_squared_bar, x_plus_bar, target_rate_rho, device)
                            ##print("y_t : ", y_t)
                            y_t_n = y_t.item()
                            print("y_t_n : ", y_t_n)
                            ##if y_t_n == 1:
                            if y_t_n == 0:
                                print("if explore_flag == False: if y_t == 1: ")
                                action_m = 1
                                ##exploit_count += 1
                            else:
                                print("if explore_flag == False: if y_t == 1: else ")
                                action_m = 0
                                exploit_count = 1
                                explore_flag = True

                        else:
                            print("if explore_flag == False: if y_t == 1: else_222 ")
                            temp_episode_reward_h = episode_reward_h.cpu().numpy()[0]
                            temp_episode_reward_h = gamma_tigeer * temp_episode_reward_h
                            value_h_reward += temp_episode_reward_h
                            gamma_tigeer = gamma_tigeer * gamma

                            exploit_count += 1

                if action_m == 1:
                    print("if explore_flag == False: if y_t == 1: if action_m == 1: ")
                    next_goal = (actor_eval_h(next_state.to(device)).detach().cpu()).squeeze().to(device)
                    next_goal = torch.min(torch.max(next_goal, -max_goal), max_goal)
                    off_policy_count += 1
                else:
                    print("if explore_flag == False: if y_t == 1: if action_m == 1: else ")
                    next_goal = (torch.randn_like(goal) * max_goal)
                    next_goal = torch.min(torch.max(next_goal, -max_goal), max_goal)
                    random_policy_count += 1

                record_logger(args=[off_policy_count, random_policy_count], option='ppom_high', step=t-start_timestep)

            # 2.2.8 collect high-level experience
            goal_hat, updated = off_policy_correction(actor_target_l, action_sequence, state_sequence, goal_dim, goal_sequence[0], next_state, max_goal, device)
            ##print("goal_sequence[0] : ", goal_sequence[0])
            ##print("goal_hat : ", goal_hat.to(device))
            temp_goal_diff = goal_sequence[0] - goal_hat.to(device)
            goal_diff_norm = torch.pow(sum(torch.pow(temp_goal_diff, 2)), 1 / 2).cpu().numpy()
            ##print("temp_goal_diff : ", l2_norm)
            experience_buffer_h.add(state_sequence[0], goal_hat, episode_reward_h, next_state, done_h)
            # if state_print_trigger.good2log(t, 500): print_cmd_hint(params=[state_sequence, goal_sequence, action_sequence, intri_reward_sequence, updated, goal_hat, reward_h_sequence], location='training_state')
            # 2.2.9 reset segment arguments & log (reward)
            state_sequence, action_sequence, intri_reward_sequence, goal_sequence, reward_h_sequence = [], [], [], [], []
            print(f"    > Segment: Total T: {t + 1} Episode_L Num: {episode_num_l + 1} Episode_L T: {episode_timestep_l} Reward_L: {float(episode_reward_l):.3f} Reward_H: {float(episode_reward_h):.3f}")
            if t >= start_timestep: record_logger(args=[episode_reward_l, episode_reward_h], option='reward', step=t-start_timestep)
            episode_reward_l, episode_timestep_l = 0, 0
            episode_reward_h = 0
            episode_num_l += 1
        # 2.2.10 update observations
        state = next_state
        goal = next_goal

        # 2.2.11 update networks
        if t >= start_timestep:
            target_q_l, critic_loss_l, actor_loss_l = \
                step_update_l(experience_buffer_l, batch_size, total_it, actor_eval_l, actor_target_l, critic_eval_l, critic_target_l, critic_optimizer_l, actor_optimizer_l, params)

        ##if t >= start_timestep and (t + 1) % c == 0:
        if t >= (start_timestep + update_timestep) and (t + 1) % c == 0:
            target_q_h, critic_loss_h, actor_loss_h = \
                step_update_h(experience_buffer_h, batch_size, total_it, actor_eval_h, actor_target_h, critic_eval_h, critic_target_h, critic_optimizer_h, actor_optimizer_h, params)


        # 2.2.12 log training curve (inter_loss)
        if t >= start_timestep and t % log_interval == 0:
            record_logger(args=[target_q_l, critic_loss_l, actor_loss_l, target_q_h, critic_loss_h, actor_loss_h], option='inter_loss', step=t-start_timestep)
            ##record_logger(args=[target_q_l, critic_loss_l, actor_loss_l], option='inter_loss', step=t-start_timestep)
            record_logger([success_rate], 'success_rate', step=t - start_timestep)
            ##record_logger([higher_success_rate], 'higher_success_rate', step=t - start_timestep)
            record_logger([goal_diff_norm], 'goal_diff_norm', step=t - start_timestep)

        # 2.2.13 start new episode
        if episode_timestep_h >= episode_len:
            # > update loggers
            if t > start_timestep: episode_num_h += 1
            else: episode_num_h = 0
            print(f"    >>> Episode: Total T: {t + 1} Episode_H Num: {episode_num_h+1} Episode_H T: {episode_timestep_h} Reward_Episode: {float(episode_reward):.3f}\n")
            # > clear loggers
            episode_reward = 0
            state_sequence, action_sequence, intri_reward_sequence, goal_sequence, reward_h_sequence = [], [], [], [], []
            episode_reward_l, episode_timestep_l, episode_num_l = 0, 0, 0
            state, done_h = Tensor(env.reset()).to(device), Tensor([False])
            episode_reward_h, episode_timestep_h = 0, 0
            episode_reward_h_m, episode_count_m, episode_done_count_m = 0, 0, 0
        # 2.2.14 update training loop arguments
        episode_timestep_l += 1
        episode_timestep_h += 1
        # 2.2.15 save videos & checkpoints
        if save_video and video_log_trigger.good2log(t, video_interval):
            log_video_hrl(env_name, actor_target_l, actor_target_h, params)
            time_logger.sps(t)
            time_logger.time_spent()
            print("")
        if checkpoint_logger.good2log(t, checkpoint_interval):
            logger = [time_logger, state_print_trigger, video_log_trigger, checkpoint_logger, evalutil_logger, episode_num_h]
            save_checkpoint(t, actor_target_l, critic_target_l, actor_optimizer_l, critic_optimizer_l, experience_buffer_l,
                            actor_target_h, critic_target_h, actor_optimizer_h, critic_optimizer_h, experience_buffer_h,
                            logger, params)
        if t > start_timestep and evalutil_logger.good2log(t, evaluation_interval):
            success_rate = evaluate(actor_target_l, actor_target_h, params, target_pos, device)
            ##higher_success_rate = evaluate_higher_policy(actor_target_l, ppo_agent, params, target_pos, device, max_goal)
    # 2.3 final log (episode videos)
    logger = [time_logger, state_print_trigger, video_log_trigger, checkpoint_logger, evalutil_logger, episode_num_h]
    save_checkpoint(max_timestep, actor_target_l, critic_target_l, actor_optimizer_l, critic_optimizer_l, experience_buffer_l,
                    actor_target_h, critic_target_h, actor_optimizer_h, critic_optimizer_h, experience_buffer_h,
                    logger, params)
    for i in range(3):
        log_video_hrl(env_name, actor_target_l, actor_target_h, params)
    print_cmd_hint(params=params, location='end_train')


if __name__ == "__main__":
    env_name = "AntFall"
    env = get_env(env_name)
    state_dim = env.observation_space.shape[0]
    goal_dim = 3
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    max_goal = [10., 10., .5]
    policy_params = ParamDict(
        seed=54321,
        c=10,
        policy_noise_scale=0.2,
        policy_noise_std=1.,
        expl_noise_std_l=1.,
        expl_noise_std_h=1.,
        policy_noise_clip=0.5,
        max_action=max_action,
        max_goal=max_goal,
        discount=0.99,
        policy_freq=1,
        tau=5e-3,
        actor_lr=1e-4,
        critic_lr=1e-3,
        reward_scal_l=1.,
        reward_scal_h=.1,
        episode_len=1000,
        max_timestep=int(3e6),
        start_timestep=int(300),
        batch_size=100
    )
    params = ParamDict(
        policy_params=policy_params,
        env_name=env_name,
        state_dim=state_dim,
        action_dim=action_dim,
        goal_dim=3,
        video_interval=int(1e4),
        log_interval=5,
        checkpoint_interval=int(1e5),
        evaluation_interval=int(1e4),
        prefix="test_simple_origGoal_fixedIntriR_posER",
        ##save_video=True,
        save_video=False,
        use_cuda=True,
        # checkpoint="hiro-antpush_test_simple_origGoal_fixedIntriR_posER-it(2000000)-[2020-07-02 20:35:25.673267].tar"
        checkpoint=None
    )

    wandb.init(project="ziang-hiro-new")
    train(params=params)
