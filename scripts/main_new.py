import os
import sys
import time
import copy
import collections
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import random
import json
sys.path.append('./')
sys.path.append('..')
sys.path.append('PATH_TO_Help_Hans_rl_envs')
# example PATH_TO_Help_Hans_rl_envs: sys.path.append('/home/my computer/helping_hands_rl_envs')

from scripts.create_agent import createAgent
from utils.visualization_utils import plot_action
from utils.parameters import *
from storage.buffer import QLearningBufferExpert
from utils.logger import Logger
from utils.env_wrapper import EnvWrapper
from utils.torch_utils import augmentData2Buffer
np.seterr(invalid='ignore')

ExpertTransition = collections.namedtuple('ExpertTransition',
                                          'state obs action reward next_state next_obs done step_left expert')


from raven.dataset import Dataset
from raven.gripper_env import Environment
from raven.gripper_block_insertion import BlockInsertion
import matplotlib.pyplot as plt
from raven import utils_
from raven import cameras




def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def getCurrentObs(in_hand, obs):
    print("getCurrentObs in_hand shape",in_hand.shape)
    print("getCurrentObs obs shape",obs.shape)
    obss = []
    for i, o in enumerate(obs):
        obss.append((o.squeeze(), in_hand[i].squeeze()))
    return obss


def train_step(agent, replay_buffer, logger):
    """
    Training an SGD step
    """
    batch = replay_buffer.sample(sample_batch_size, onpolicydata=sample_onpolicydata, onlyfailure=onlyfailure)
    loss, td_error = agent.update(batch)
    logger.trainingBookkeeping(loss, td_error.mean().item())
    logger.num_training_steps += 1


def show_pics(hmap,q_map=None,left = 1):
    if q_map is not None:
        q_map1 = q_map[0]
        plt.subplot(1, 2, 1)
        q_map1 = q_map1.detach().numpy().squeeze(0)
        plt.imshow(q_map1, cmap='gray')
        plt.subplot(1, 2, 2)
        if left == 1:
            plt.imshow(hmap[:160,:], cmap='gray')
        else:
            plt.imshow(hmap[160:,:], cmap='gray')

    else:
        if left == 1:
            plt.imshow(hmap[:160,:], cmap='gray')
        elif left == 0:
            plt.imshow(hmap[160:,:], cmap='gray')
        else:
            plt.imshow(hmap, cmap='gray')
    plt.show()


def evaluate(envs, agent, logger):
    """
    Evaluate the agent with num_eval_episodes
    """
    eval_steps = 0
    eval_rewards = []
    if not no_bar:
        eval_bar = tqdm(total=num_eval_episodes)

    agent.eval()
    with torch.no_grad():
        while eval_steps < num_eval_episodes:
            obs__ = envs.reset(collection=False)
            cmap, hmap = utils_.get_fused_heightmap(
                    obs__, cameras.RealSenseD415.CONFIG, np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.28]]), 0.003125)
            obs = torch.tensor(hmap[:160,:])
            obs = torch.unsqueeze(obs, 0)  
            obs = torch.unsqueeze(obs, 1)  
            print("obs:",obs.shape)
            states = torch.tensor([0])
            in_hands = torch.zeros((1,1,32,32))
            # Boltzmann sampling an action with evaluation temperature test_tau
            q_value_maps, actions_star_idx, actions_star =\
                    agent.getBoltzmannActions(states, in_hands, obs, temperature=test_tau)
            # show_pics(hmap,q_value_maps)
            actions_star = torch.cat((actions_star, states.unsqueeze(1)), dim=1)
            x = actions_star[0][0]
            y = actions_star[0][1]
            angle = utils_.eulerXYZ_to_quatXYZW([0,0,actions_star[0][2]])
            actions_star[0][2] = actions_star[0][2]*2
            eulerXYZ = utils_.quatXYZW_to_eulerXYZ(angle)
            print("previous euler:",[0,0,actions_star[0][2]],"transformed euler:",eulerXYZ)
            act = dict()
            act["pose0"] = (np.array([x, y , 0.07]),np.array(angle))
            act["pose1"] = (np.array([x + 0.02, y , 0.07]),np.array([-0.       ,  0.       , -0.5      , -0.8660254]))
            obs__, reward, done, info = envs.step_new(act)
            cmap, hmap = utils_.get_fused_heightmap(
            obs__, cameras.RealSenseD415.CONFIG, np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.28]]), 0.003125)
            obs_ = torch.tensor(hmap[:160,:])
            # obs_ = torch.zeros((160,160))
            obs_ = torch.unsqueeze(obs_, 0)  # 添加一个维度，使其形状为 (1, 320, 160)
            obs_ = torch.unsqueeze(obs_, 1)  # 添加一个维度，使其形状为 (1, 1, 320, 160)
            rewards = torch.tensor([reward])
            in_hands_ = torch.zeros((1,1,32,32))
            if done:
                dones = torch.tensor([1])
            else:
                dones = torch.tensor([0])
            dones = torch.tensor([1])
            print("dones:",dones)
            print("rewards:",rewards)
            states_ = torch.tensor([0])
            
            # states_, in_hands_, obs_, rewards, dones = envs.step(act, auto_reset=True)
            print("obs:",obs)
            print("action:",actions_star)
            print("states_:",states_)
            rewards = rewards.numpy()
            dones = np.ones_like(rewards)
            states = copy.copy(states_)
            obs = copy.copy(obs_)
            eval_steps += int(np.sum(dones))
            eval_rewards.append(rewards)
            eval_bar.update(eval_steps - eval_bar.n)
            
    agent.train()
    eval_rewards = np.concatenate(eval_rewards)
    logger.eval_rewards.append(eval_rewards.mean())

    if not no_bar:
        eval_bar.close()


def saveModelAndInfo(logger, agent):
    """
    Saving the model parameters and the training information
    :param logger:
    :param agent:
    """
    logger.saveModel(logger.num_steps, env, agent)
    logger.saveLossCurve(100)
    logger.saveTdErrorCurve(100)
    logger.saveRewards()
    logger.saveSGDtime()
    logger.saveLosses()
    logger.saveTdErrors()
    logger.saveStepLeftCurve(1000)
    logger.saveExpertSampleCurve(100)
    logger.saveLearningCurve(learning_curve_avg_window)
    logger.saveEvalCurve()
    logger.saveEvalRewards()


def collect_data(envs,agent,mode):
    angle_check = []
    num_eval_episodes = 5
    data_dir = "./"
    task = "block-insertion"
    eval_steps = 0
    eval_rewards = []
    if not no_bar:
        eval_bar = tqdm(total=num_eval_episodes)
    dataset = Dataset(os.path.join(data_dir,f'{task}-{mode}'))
    seed = dataset.max_seed
    if seed < 0:
        seed = -1 if (mode == 'test') else -2

    agent.eval()
    with torch.no_grad():
        while eval_steps < num_eval_episodes:
            episode, total_reward = [], 0
            obs__ = envs.reset(collection=True)
            info = None
            reward = 0
            cmap, hmap = utils_.get_fused_heightmap(
                    obs__, cameras.RealSenseD415.CONFIG, np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.28]]), 0.003125)
            obs = torch.tensor(hmap[160:,:])
            obs = torch.unsqueeze(obs, 0)  
            obs = torch.unsqueeze(obs, 1)  
            print("obs:",obs.shape)
            states = torch.tensor([0])
            in_hands = torch.zeros((1,1,32,32))
            # Boltzmann sampling an action with evaluation temperature test_tau
            q_value_maps, actions_star_idx, actions_star =\
                    agent.getBoltzmannActions(states, in_hands, obs, temperature=test_tau)
            # show_pics(hmap,q_value_maps,left = 0)
            actions_star = torch.cat((actions_star, states.unsqueeze(1)), dim=1)
            x = actions_star[0][0]
            y = actions_star[0][1]
            print("actions_star[0][2] before eulerXYZ_to_quatXYZW",actions_star[0][2].item())
            angle = utils_.eulerXYZ_to_quatXYZW([0,0,actions_star[0][2].item()])
            print("actions_star[0][2] after eulerXYZ_to_quatXYZW",actions_star[0][2].item())
            act = dict()
            act_rev = dict()
            y = y + 0.5
            act["pose0"] = (np.array([x, y , 0.07]),np.array(angle))
            x_pose1 = random.uniform(0.35,0.65)
            y_pose1 = random.uniform(-0.4,-0.1)
            angle_pose1 = random.uniform(-3.14,3.14)
            print("angle_pose1 before eulerXYZ_to_quatXYZW",angle_pose1)
            angle_pose1_ = utils_.eulerXYZ_to_quatXYZW([0,0,angle_pose1])
            print("angle_pose1 after eulerXYZ_to_quatXYZW",angle_pose1)
            act["pose1"] = (np.array([x_pose1,y_pose1 , 0.07]),np.array(angle_pose1_))
            new_angle0 = actions_star[0][2].item() + angle_pose1
            new_angle1 = - angle_pose1
            act_rev["pose0"] = (act["pose1"][0],utils_.eulerXYZ_to_quatXYZW([0,0,new_angle0]))
            act_rev["pose1"] = (act["pose0"][0],utils_.eulerXYZ_to_quatXYZW([0,0,new_angle1]))
            obs__1, reward1, done1, info1 = envs.step_new(act)
            print("reward is ",reward1)
            cmap, hmap = utils_.get_fused_heightmap(
                    obs__1, cameras.RealSenseD415.CONFIG, np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.28]]), 0.003125)
            # show_pics(hmap,q_value_maps,left = 0)
            episode.append((obs__1, act_rev, reward, info))
            episode.append((obs__, None, reward1, info))
            # for check
            _, reward2, ___, ___ = envs.step(act_rev)

            if reward2 > 0.9:
                dataset.add(seed,episode)
                eval_steps += 1
                eval_bar.update(eval_steps - eval_bar.n)
                angle_check.append([actions_star[0][2].item(),angle_pose1])
            print("angle_check:",angle_check)
            
    agent.train()
    if not no_bar:
        eval_bar.close()


def train():
    
    params = {
    "lr": lr,
    "training_iters": training_iters,
    "patch_size": patch_size,
    "num_rotations": num_rotations
    }

    
    best_reward = 0
    in_hand_size = 32
    if seed is not None:
        set_seed(seed)
    # 初始化环境
    # set up the environment
    enc_cls = Environment
    env = enc_cls('./raven/assets/',
                    disp=True,
                    shared_memory=False,
                    hz=480)
    task = BlockInsertion(continuous=False)
    env.set_task(task)
    obs = env.reset()
    eval_envs = env
    # 初始化网络
    # setup the agent
    agent = createAgent()
    agent.train()
    if load_model_pre:
        agent.loadModel(load_model_pre)
    # 初始化日志
    # setup logging
    simulator_str = copy.copy(simulator)
    if simulator == 'pybullet':
        simulator_str += ('_' + robot)
    log_dir = os.path.join("./log/", '{}_{}_{}_{}_{}'.format(alg, model, simulator_str, num_objects, max_episode_steps))
    print("log_dirrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr:",log_dir)
    if note:
        log_dir += '_'
        log_dir += note
    logger = Logger(log_dir, env, 'train', num_processes, max_episode, log_sub)
    # 储存可调参数
    # save params
    with open(os.path.join(logger.base_dir, "params.json"),"w") as f:
        json.dump(params, f)
    hyper_parameters['model_shape'] = agent.getModelStr()
    logger.saveParameters(hyper_parameters)

    # 初始化buffer
    # setup buffer
    replay_buffer = QLearningBufferExpert(buffer_size)


    obs__ = env.reset()
    cmap, hmap = utils_.get_fused_heightmap(
            obs__, cameras.RealSenseD415.CONFIG, np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.28]]), 0.003125)
    obs = torch.tensor(hmap[:160,:])
    obs = torch.unsqueeze(obs, 0)  
    obs = torch.unsqueeze(obs, 1)  
    states = torch.tensor([0])
    in_hands = torch.zeros((1,1,in_hand_size,in_hand_size))

    if load_sub:
        logger.loadCheckPoint(os.path.join(log_dir, load_sub, 'checkpoint'), env, agent, replay_buffer)

    if not no_bar:
        pbar = tqdm(total=max_episode)
        pbar.set_description('Episodes:0; Reward:0.0; Explore:0.0; Loss:0.0; Time:0.0')
    timer_start = time.time()

    # the training loop
    while logger.num_episodes < max_episode:
        print("train once—————————————————————————————————————————————————————————— ")
        gpu_usage = torch.cuda.max_memory_allocated(device=torch.cuda.current_device())
        print(f"GPU usage: {gpu_usage / 1024 / 1024} MB")
        eps = init_eps if logger.num_episodes < step_eps else final_eps
        is_expert = 0
        q_value_maps, actions_star_idx, actions_star, in_hand_obs = \
            agent.getBoltzmannActions(states, in_hands, obs, temperature=train_tau, eps=eps, return_patch=True)
        print(44444444444444444444444444)
        print("q_value_maps.shape",q_value_maps[0].shape)
        q_map1 = q_value_maps[0]
        plt.subplot(1, 2, 1)
        q_map1 = q_map1.detach().numpy().squeeze(0)
        plt.imshow(q_map1, cmap='gray')
        buffer_obs = getCurrentObs(in_hands, obs)
        print("angle num-------------------------------------------",8*3.14/actions_star[0][2])
        # actions_star = torch.cat((actions_star, states.unsqueeze(1)), dim=1)
        x = actions_star[0][0]
        y = actions_star[0][1]
        angle = utils_.eulerXYZ_to_quatXYZW([0,0,actions_star[0][2]])
        actions_star[0][2] = actions_star[0][2]*2
        print("actions_star2",actions_star)
        act = dict()
        act["pose0"] = (np.array([x, y , 0.07]),np.array(angle))
        act["pose1"] = (np.array([0.33741313, 0.14475863, 0.07]),np.array([-0.       ,  0.       , -0.5      , -0.8660254]))
        obs__, reward, done, info = env.step_new(act)
        cmap, hmap = utils_.get_fused_heightmap(
            obs__, cameras.RealSenseD415.CONFIG, np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.28]]), 0.003125)
        # image = hmap.squeeze(0)
        # image = image.squeeze(0)
        # 显示图像
        plt.subplot(1, 2, 2)
        plt.imshow(hmap[:160,:], cmap='gray')
        # plt.show()

        obs_ = torch.tensor(hmap[:160,:])
        # obs_ = torch.zeros((160,160))
        obs_ = torch.unsqueeze(obs_, 0)  # 添加一个维度，使其形状为 (1, 320, 160)
        obs_ = torch.unsqueeze(obs_, 1)  # 添加一个维度，使其形状为 (1, 1, 320, 160)
        rewards = torch.tensor([reward])
        in_hands_ = torch.zeros((1,1,in_hand_size,in_hand_size))
        if done:
            dones = torch.tensor([1])
        else:
            dones = torch.tensor([0])
        dones = torch.tensor([1])
        print("dones:",dones)
        print("rewards:",rewards)
        states_ = torch.tensor([0])
        # states_, in_hands_, obs_, rewards, dones = env.step()
        steps_lefts = torch.tensor([1])

        print("obs:",obs_.shape)
        print("actions_star3:",actions_star)
        print("states_:",states_)
        
        # # 将张量转换为图像
        # image = obs.detach().numpy().squeeze(0)
        # image = image.squeeze(0)
        # # 显示图像
        # plt.imshow(image, cmap='gray')
        # plt.show()

        done_idxes = torch.nonzero(dones).squeeze(1)
        if done_idxes.shape[0] != 0:
            reset_obs_ = env.reset()
            cmap, hmap = utils_.get_fused_heightmap(
            reset_obs_, cameras.RealSenseD415.CONFIG, np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.28]]), 0.003125)
            reset_obs_ = torch.tensor(hmap[:160,:])
            # reset_obs_ = torch.zeros((160,160))
            reset_obs_ = torch.unsqueeze(reset_obs_, 0)  # 添加一个维度，使其形状为 (1, 320, 160)
            reset_obs_ = torch.unsqueeze(reset_obs_, 1)  # 添加一个维度，使其形状为 (1, 1, 320, 160)
            reset_states_ = torch.tensor([0])
            reset_in_hands_ = torch.zeros((1,1,in_hand_size,in_hand_size))
            for j, idx in enumerate(done_idxes):
                states_[idx] = reset_states_[j]
                in_hands_[idx] = reset_in_hands_[j]
                obs_[idx] = reset_obs_[j]

        # if render:
        # yw: no use for now
        # if render and not rewards.item():
        #     plot_action(obs, agent, actions_star, actions_star_idx, q_value_maps, num_rotations,
        #                 patch_size, rewards, in_hand_obs, action_sequence)

        buffer_obs_ = getCurrentObs(in_hands_, obs_)
        print("steps_lefts",steps_lefts)
        if not fixed_buffer:
            for i in range(num_processes):
                data = ExpertTransition(states[i], buffer_obs[i], actions_star_idx[i], rewards[i], states_[i],
                                            buffer_obs_[i], dones[i], steps_lefts[i], torch.tensor(is_expert))
                augmentData2Buffer(replay_buffer, data, agent.rzs,
                                   onpolicy_data_aug_n, onpolicy_data_aug_rotate, onpolicy_data_aug_flip)

        logger.stepBookkeeping(rewards.numpy(), steps_lefts.numpy(), dones.numpy())

        if logger.num_episodes >= training_offset:
            for training_iter in range(training_iters):
                SGD_start = time.time()
                train_step(agent, replay_buffer, logger)
                logger.SGD_time.append(time.time() - SGD_start)

        states = copy.copy(states_)
        obs = copy.copy(obs_)
        in_hands = copy.copy(in_hands_)

        if not no_bar:
            timer_final = time.time()
            description = 'Steps:{}; Reward:{:.03f}; Eval Reward:{:.03f}; Explore:{:.03f}; Loss:{:.03f}; Time:{:.03f}'.format(
                logger.num_steps, logger.getCurrentAvgReward(learning_curve_avg_window),
                logger.eval_rewards[-1] if len(logger.eval_rewards) > 0 else 0,
                eps, float(logger.getCurrentLoss()), timer_final - timer_start)
            pbar.set_description(description)
            timer_start = timer_final
            pbar.update(logger.num_episodes - pbar.n)
        logger.num_steps += num_processes

        if logger.num_training_steps > 0 and logger.num_episodes % eval_freq == eval_freq - 1:
            evaluate(eval_envs, agent, logger)
            agent.writer.add_scalar('evaluate_reward', logger.eval_rewards[-1], logger.num_episodes)

        
        if (logger.num_episodes + 1) % (max_episode // num_saves) == 0 and logger.eval_rewards[-1] > best_reward:
            best_reward = logger.eval_rewards[-1]
            saveModelAndInfo(logger, agent)

        
        if len(logger.eval_rewards) > 0:
        # stop training if successful rate is higher than a threshold 
            if logger.eval_rewards[-1] > 0.90 or logger.getCurrentAvgReward(learning_curve_avg_window) > 0.90:
                saveModelAndInfo(logger, agent)
                break
        
        agent.writer.add_scalar('window50_reward', logger.getCurrentAvgReward(learning_curve_avg_window), logger.num_episodes)

        
    
    # saveModelAndInfo(logger, agent)
    logger.saveCheckPoint(args, env, agent, replay_buffer)
    with open("./log/" + "params.json", "w") as f:
        json.dump(params, f)
    env.close()
    agent.writer.close()
    print("finishhhhhhhhhhhhhhhhhhhhhh")


if __name__ == '__main__':
    is_test =   False
    is_collection = False
    if is_collection:
        print("collecting time!")
        agent = createAgent()
        agent.train()
        agent.loadModel(load_model_pre)
        env_config['render'] = True 
        # eval_envs = EnvWrapper(eval_num_processes, simulator, env, env_config, planner_config)
        enc_cls = Environment 
        env = enc_cls('./raven/assets/',
                        disp=True,

                        shared_memory=False,
                        hz=480)
        task = BlockInsertion(continuous=False)
        env.set_task(task)
        obs = env.reset(collection=True)
        eval_envs = env
        # setup logging
        simulator_str = copy.copy(simulator)
        if simulator == 'pybullet':
            simulator_str += ('_' + robot)
        log_dir = os.path.join("./log/", '{}_{}_{}_{}_{}'.format(alg, model, simulator_str, num_objects, max_episode_steps))
        if note:
            log_dir += '_'
            log_dir += note
        logger = Logger(log_dir, eval_envs, 'test', num_processes, max_episode, log_sub)
        hyper_parameters['model_shape'] = agent.getModelStr()
        logger.saveParameters(hyper_parameters)

        # setup buffer
        replay_buffer = QLearningBufferExpert(buffer_size)
        collect_data(eval_envs, agent,"train") 
    elif not is_test:
        print('is_test',is_test)
        train() 
    elif is_test:
        print("test time!")
        agent = createAgent()
        agent.train()
        agent.loadModel(load_model_pre)
        env_config['render'] = True 
        # eval_envs = EnvWrapper(eval_num_processes, simulator, env, env_config, planner_config)
        enc_cls = Environment
        env = enc_cls('./raven/assets/',
                        disp=True,
                        shared_memory=False,
                        hz=480)
        task = BlockInsertion(continuous=False)
        env.set_task(task)
        obs = env.reset(collection=True)
        eval_envs = env
        # setup logging
        simulator_str = copy.copy(simulator)
        if simulator == 'pybullet':
            simulator_str += ('_' + robot)
        log_dir = os.path.join("./log/", '{}_{}_{}_{}_{}'.format(alg, model, simulator_str, num_objects, max_episode_steps))
        if note:
            log_dir += '_'
            log_dir += note
        logger = Logger(log_dir, eval_envs, 'test', num_processes, max_episode, log_sub)
        hyper_parameters['model_shape'] = agent.getModelStr()
        logger.saveParameters(hyper_parameters)

        # setup buffer
        replay_buffer = QLearningBufferExpert(buffer_size)
        evaluate(eval_envs, agent, logger) 