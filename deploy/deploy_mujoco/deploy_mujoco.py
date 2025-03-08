ASAP_ROOT_DIR = "/home/lab/wy/ASAP"

import time

import mujoco.viewer
import mujoco
import numpy as np
import torch
import yaml
import torch.nn.functional as F

def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd


class HistoryHandler:
    def __init__(self, history_length=4, batch_size=1):
        self.history = {}
        self.history_length = history_length
        self.batch_size = batch_size
        
        self.history_structure = [
            'actions',
            'base_ang_vel',
            'dof_pos',
            'dof_vel',
            'projected_gravity',
            'ref_motion_phase'
        ]
        
        for key in self.history_structure:
            self.history[key] = np.zeros(
                (batch_size, history_length, self._get_obs_dim(key)),
                dtype=np.float32
            )

    def _get_obs_dim(self, key):
        dim_map = {
            'actions': 23,
            'base_ang_vel': 3,
            'dof_pos': 23,
            'dof_vel': 23,
            'projected_gravity': 3,
            'ref_motion_phase': 1
        }
        return dim_map[key]


    def update(self, current_obs):
        # print('update history')
        # print('current_obs:', current_obs)
        # print('before update:get_history_actor:', self.get_history_actor())

        for key in self.history_structure:
            #move data to the right,the new data will be in the first column
            # print('before self.history[key:',self.history[key])
            self.history[key] = np.roll(self.history[key], shift=1, axis=1)
            # print('after self.history[key:',self.history[key])
            
            if current_obs[key].ndim == 1:
                current_obs[key] = current_obs[key].reshape(1, -1)
            
            #fill new data to the first column
            self.history[key][:, 0] = current_obs[key]
            # print('after self.history[key]2:',self.history[key])
        # print('after update:get_history_actor:', self.get_history_actor())
        # exit(-1)

    def get_history_actor(self):
        history_vectors = []
        
        for key in self.history_structure:
            hist_flat = self.history[key].reshape(self.batch_size, -1) 
            history_vectors.append(hist_flat)
        
        return np.concatenate(history_vectors, axis=1)

#########这里定义了些参数的范围
obs_scales = {
    'base_ang_vel': 0.25,
    'projected_gravity': 1.0,
    'dof_pos': 1.0,
    'dof_vel': 0.05,
    'actions': 1.0,
    'ref_motion_phase': 1.0,
    'history_actor': 1.0
}


def build_actor_obs(current_obs, history_handler):
    ordered_keys = sorted([
        'actions',
        'base_ang_vel', 
        'dof_pos',
        'dof_vel',
        'history_actor',
        'projected_gravity',
        'ref_motion_phase'
    ])
    # print('enter build_actor_obs')
    # print('current_obs:',current_obs)
    scaled_obs = []
    for key in ordered_keys:
        # print('key:',key)
        if key == 'history_actor':
            # print('before get_history_actor,scaled_obs:',scaled_obs)
            hist = history_handler.get_history_actor()
            # print('obs_scales[key]:','key',obs_scales[key])
            scaled_hist = hist * obs_scales[key]
            scaled_obs.append(scaled_hist)
        else:
            # print('key',key,'obs_scales.get(key, 1.0):',obs_scales.get(key, 1.0))
            scaled_value = current_obs[key] * obs_scales.get(key, 1.0)
            # print('scaled_value:',scaled_value)
            scaled_obs.append(scaled_value.reshape(1, -1)) 
            
    # return np.hstack(scaled_obs)
    ret  = np.concatenate(scaled_obs, axis=1)  # Ensure all arrays have 2 dimensions
    # print('ret:',ret)
    return ret



def quaternion_to_euler_array(quat):
    # Ensure quaternion is in the correct format [x, y, z, w]
    w, x, y, z = quat
    
    # Roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)
    
    # Pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)
    
    # Yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)
    
    # Returns roll, pitch, yaw in a NumPy array in radians
    return np.array([roll_x, pitch_y, yaw_z])

def pd_control(target_q, q, kp, target_dq, dq, kd):
    '''Calculates torques from position commands
    '''
    return (target_q - q) * kp + (target_dq - dq) * kd


def quat_rotate_inverse(q, v):
    shape = q.shape
    q_w = q[-1]
    q_vec = q[:3]
    a = (2.0 * (q_w ** 2) - 1.0) * v
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a - b + c


####获取观测值
def get_obs(robot,dofs_idx):
    motor_dof = robot.qpos[dofs_idx]                         # 获取关节角度（Position）
    motor_vel = robot.qvel[dofs_idx]                         # 获取关节速度（Velocity）
    # -----imu------#
    base_quat_w_first = robot.xquat                          # 获取基座的四元数（MuJoCo 使用 WXYZ 顺序）
    euler = quaternion_to_euler_array(base_quat_w_first[2])     # 转换为欧拉角
    base_angular = robot.qvel[3:6]

    gravity_vec = np.array([0, 0, -1])
    base_quat_w_last = np.array([0,0,0,0])     # 变换为 XYZW 格式
    base_quat=robot.qpos[23:27]
    projected_gravity = quat_rotate_inverse(base_quat, gravity_vec)       #计算投影重力
    

    return np.array(motor_dof), np.array(motor_vel), np.array(euler), np.array(base_angular),projected_gravity

# ------motor------- # 
jnt_names = ['left_ankle_pitch_joint', 
             'left_ankle_roll_joint', 
             'left_elbow_joint', 
             'left_hip_pitch_joint', 
             'left_hip_roll_joint', 
             'left_hip_yaw_joint', 
             'left_knee_joint', 
             'left_shoulder_pitch_joint', 
             'left_shoulder_roll_joint', 
             'left_shoulder_yaw_joint', 
             'left_wrist_roll_joint', 
             'right_ankle_pitch_joint', 
             'right_ankle_roll_joint', 
             'right_elbow_joint', 
             'right_hip_pitch_joint', 
             'right_hip_roll_joint', 
             'right_hip_yaw_joint', 
             'right_knee_joint', 
             'right_shoulder_pitch_joint', 
             'right_shoulder_roll_joint', 
             'right_shoulder_yaw_joint', 
             'right_wrist_roll_joint', 
             'waist_yaw_joint'
]






if __name__ == "__main__":
    config_file = "g1.yaml"
    with open(f"{ASAP_ROOT_DIR}/deploy/deploy_mujoco/configs/{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = config["policy_path"].replace("{ASAP_ROOT_DIR}", ASAP_ROOT_DIR)
        xml_path = config["xml_path"].replace("{ASAP_ROOT_DIR}", ASAP_ROOT_DIR)
        print("-------------xml_path--------:", xml_path)

        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)

        default_angles = np.array(config["default_angles"], dtype=np.float32)

        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        num_actions = config["num_actions"]
        num_obs = config["num_obs"]
        
        cmd = np.array(config["cmd_init"], dtype=np.float32)

    counter = 0

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)       #从xml文件加载机器人模型
    d = mujoco.MjData(m)          #创建仿真数据对象
    policy = torch.jit.load(policy_path)

    m.opt.timestep = simulation_dt    #设置仿真时间步长

    dofs_idx = [m.jnt_dofadr[m.joint(name).id] for name in jnt_names]  # 获取 DOF 索引
    joint_index = []


    # 设置 kp（刚度）
    kp_values = np.array([100., 100., 100., 200.,  20.,  20., 100., 100., 100., 200.,  20.,  20.,
        400., 400., 400.,  90.,  60.,  20.,  60.,  90.,  60.,  20.,  60.])

    # 设置 kv（阻尼）
    kv_values = np.array([2.5000, 2.5000, 2.5000, 5.0000, 0.2000, 0.1000, 2.5000, 2.5000, 2.5000,
            5.0000, 0.2000, 0.1000, 5.0000, 5.0000, 5.0000, 2.0000, 1.0000, 0.4000,
            1.0000, 2.0000, 1.0000, 0.4000, 1.0000])

    # 给所有 DOF 关节赋值
    for i, dof_id in enumerate(dofs_idx):
        joint_id = m.dof_jntid[dof_id]  # 获取 DOF 对应的关节索引
        joint_index.append(joint_id)
        m.jnt_stiffness[joint_id] = kp_values[i]  # 只修改有效的关节
        m.dof_damping[dof_id] = kv_values[i]  # 设置 kv（阻尼）

    #####--------------------控制 DOF 力矩范围---------------########
    # 你的力矩范围
    lower = -1 * np.array([88, 88, 88, 139, 50, 50, 88, 88, 88, 139, 50, 50, 88, 50, 50, 25, 25, 25, 25, 25, 25, 25, 25])
    upper = np.array([88, 88, 88, 139, 50, 50, 88, 88, 88, 139, 50, 50, 88, 50, 50, 25, 25, 25, 25, 25, 25, 25, 25])
    # 确保 actuator 数量匹配
    assert len(lower) == len(upper) == m.nu, f"actuator 数量不匹配: {m.nu} vs {len(lower)}"
    # 赋值控制力范围
    m.actuator_ctrlrange[:, 0] = lower  # 设置最小值
    m.actuator_ctrlrange[:, 1] = upper  # 设置最大值


    kp = 1.0*np.array([100., 100., 100., 200.,  20.,  20., 100., 100., 100., 200.,  20.,  20.,
        400., 400., 400.,  90.,  60.,  20.,  60.,  90.,  60.,  20.,  60.])
    kv = 1.0 * np.array([2.5000, 2.5000, 2.5000, 5.0000, 0.2000, 0.1000, 2.5000, 2.5000, 2.5000,
        5.0000, 0.2000, 0.1000, 5.0000, 5.0000, 5.0000, 2.0000, 1.0000, 0.4000,
        1.0000, 2.0000, 1.0000, 0.4000, 1.0000]) 

    tau_limit = np.array([88,88,88,139,50,50,88,88,88,139,50,50,88,50,50,25,25,25,25,25,25,25,25])

    default_dof_pos = np.array([-0.1000,  0.0000,  0.0000,  0.3000, -0.2000,  0.0000, -0.1000,  0.0000,
          0.0000,  0.3000, -0.2000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
          0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000])


    sum_time = 0
    count_ = 0
    target_q =  np.zeros((23))    #目标关节角度
    target_dq = np.zeros((23))    #目标关节角速度
    obs = np.zeros([1,380])


    action = np.zeros((23))
    max_motion_times = 60.0

    history_handler = HistoryHandler(history_length=4)                 ########定义一个历史的类
    time_t = 0
    ctrl_dt_s = 0.02
    with mujoco.viewer.launch_passive(m, d) as viewer:
        while True:
            start = time.time()
            time_t += ctrl_dt_s
            # print('time_t: ',time_t)

            # if time_t> max_motion_times:
            #     print('finish motion')
            #     break 
            
            #获取观测值
            motor_dofs, motor_vels,base_euler,base_ang_vel,project_grav = get_obs(d,dofs_idx)   

            ref_motion_phase = np.array([time_t / max_motion_times])
            # print('ref_motion_phase: ',ref_motion_phase)

            current_obs = {
                'actions': action.astype(np.float32),
                'base_ang_vel': base_ang_vel.astype(np.float32),
                'dof_pos': motor_dofs.astype(np.float32),
                'dof_vel': motor_vels.astype(np.float32),
                'projected_gravity': project_grav.astype(np.float32),
                'ref_motion_phase': ref_motion_phase
            }

            # print('will build actor obs')
            obs = build_actor_obs(current_obs, history_handler)
            obs = np.clip(obs, -100, 100)
            # obs_tensor = torch.from_numpy(obs).unsqueeze(0)
            action = policy(torch.tensor(obs, dtype=torch.float32))[0].detach().cpu().numpy()
            clip_action = np.clip(action, -100, 100)
            target_q = clip_action * 0.25 + default_dof_pos
            torque = pd_control(target_q,motor_dofs,kp,target_dq, motor_vels,kv)
            torque = np.clip(torque, -tau_limit, tau_limit)
            d.ctrl[:] = torque
            history_handler.update(current_obs)
            count_+=1
            end = time.time()
            sum_time += (end - start)
            print("Aaaaaaaaaa")

            mujoco.mj_step(m, d)
            viewer.sync()

    print(dofs_idx)






