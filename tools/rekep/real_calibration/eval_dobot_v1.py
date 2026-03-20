import os, sys
import numpy as np
import logging
import shlex
from multiprocessing import Value
from utils import DictConfig, configurable
from termcolor import cprint
import clip
import torch
import time
import threading
import cv2
from scipy.spatial.transform import Rotation as R
from dataclasses import dataclass

from dobot_control.robots.dobot import DobotRobot
from dobot_control.robots.robot import BimanualRobot
from dobot_control.env import RobotEnv
from dobot_control.robots.robot_node import ZMQClientRobot, ZMQServerRobot
from dobot_control.cameras.realsense_camera import RealSenseCameraV3
from dobot_control.cameras.realsense_camera import RealSenseOfflineProcessor
from scripts.manipulate_utils import robot_pose_init, set_light, load_ini_data_camera

img_list = np.array([np.zeros((480, 640, 3)), np.zeros((480, 640, 3)), np.zeros((480, 640, 3))])
deep_list = np.array([np.zeros((480, 640)), np.zeros((480, 640)), np.zeros((480, 640))])
intrinsics = [None, None, None]  # 相机内参
depth_scales = [None, None, None]  # 深度比例系数
rotation_angle = -90  # 顺时针旋转90度
z_translation = 0.6  # 向上平移0.6米

# 相机配置类
@dataclass
class CameraConfig:
    serial: str
    name: str
    id: int
    eye_in_hand: bool
    T: np.ndarray = None  # 位置向量 (3,)
    R: np.ndarray = None  # 旋转矩阵 (3,3)
    depth_scale: float = 0.001  # 深度比例系数
    
@dataclass
class Args:
    robot_port: int = 6001
    hostname: str = "127.0.0.1"
    show_img: bool = True
    save_video: bool = True

# 定义相机配置
CAMERA_CONFIGS = {
    "global1": CameraConfig(
        serial="313522072872",
        name="top",
        id=0,
        eye_in_hand=False,
        # [[ 0.03959048 -0.89031752 -0.45361582]
        #  [-0.99824988 -0.01528342 -0.05712784]
        #  [ 0.04392912  0.45508366 -0.88936443]]
        R=np.array([
            [0.03959048, -0.89031752, -0.45361582],
            [-0.99824988, -0.01528342, -0.05712784],
            [0.04392912, 0.45508366, -0.88936443]
        ]),
        # [[ 487.44040938]
        #  [-488.33690348]
        #  [ 962.44451886]]
        T=np.array([487.44040938, -488.33690348, 962.44451886])
    ),
    "global2": CameraConfig(
        serial="327122072931",
        name="left",
        id=1,
        eye_in_hand=False,
        # [[-0.0047256  -0.91413176  0.40538968]
        #  [-0.99788011  0.03062412  0.0574235 ]
        #  [-0.06490735 -0.40425894 -0.91233861]]
        R=np.array([
            [-0.0047256, -0.91413176, 0.40538968],
            [-0.99788011, 0.03062412, 0.0574235],
            [-0.06490735, -0.40425894, -0.91233861]
        ]),
        # [[-345.92268045]
        #  [-607.73641275]
        #  [ 798.20031254]]
        T=np.array([-345.92268045, -607.73641275, 798.20031254])
    ),
    "wrist": CameraConfig(
        serial="419122270614",
        name="right",
        id=2,
        eye_in_hand=True,
        R=np.array([
            [-0.00201554, 0.91562923, -0.40201871],
            [-0.99703855, 0.02906608, 0.07119898],
            [0.07687698, 0.40097166, 0.91285906]
        ]),
        T=np.array([55.57403651, 6.63696492, -117.29192883])
    ),
    "global3": CameraConfig(
        serial="338522301270",
        name="left",
        id=3,
        eye_in_hand=False,
        # 平均旋转矩阵:
        # [[ 0.78837965  0.26791604 -0.55378563]
        # [ 0.57138998 -0.65248034  0.49777796]
        # [-0.22797153 -0.70886557 -0.66748676]]

        # 平均位移向量:
        # [[  341.08990772]
        # [-1132.70197929]
        # [  885.40637981]]
        R=np.array([
            [0.78837965, 0.26791604, -0.55378563],
            [0.57138998, -0.65248034, 0.49777796],
            [-0.22797153, -0.70886557, -0.66748676]
        ]),
        T=np.array([341.08990772, -1132.70197929, 885.40637981])
    )
}

def run_thread_cam(rs_cam, which_cam):
    while thread_running:
        image_cam, deep_cam = rs_cam.read()
        # image_cam = image_cam[:, :, ::-1]
        img_list[which_cam] = image_cam
        deep_list[which_cam] = deep_cam

def save_video_thread(log_dir, task):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(os.path.join(log_dir, f'{task}_evaluation_video_{time.strftime("%Y%m%d_%H%M%S")}.mp4'), fourcc, 20, (640*3, 480))
    while thread_running:
        show_canvas = np.zeros((480, 640*3, 3), dtype=np.uint8)
        show_canvas[:, :640] = np.asarray(img_list[0][:, :, ::-1], dtype="uint8")
        show_canvas[:, 640:640 * 2] = np.asarray(img_list[1][:, :, ::-1], dtype="uint8")
        show_canvas[:, 640 * 2:640 * 3] = np.asarray(img_list[2][:, :, ::-1], dtype="uint8")
        video_writer.write(show_canvas)
        time.sleep(0.1)  # 控制写入频率
    video_writer.release()
    
def init_robots(robot_ip_left, robot_ip_right, port, hostname):
    """初始化双机械臂并启动服务器线程"""
    # _robot_l = DobotRobot(robot_ip=robot_ip_left, robot_number=2)
    _robot_r = DobotRobot(robot_ip=robot_ip_right, robot_number=1)
    # _robot_r.r_inter.ClearError()
    time.sleep(1)
    _robot_r.set_do_status([1, 0])
    _robot_r.set_do_status([2, 0])
    _robot_r.set_do_status([3, 0])
    

    return None, _robot_r, None

def init_robots_v1(robot_ip_left, robot_ip_right, port, hostname):
    """初始化双机械臂并启动服务器线程"""
    _robot_l = DobotRobot(robot_ip=robot_ip_left, robot_number=2)
    _robot_r = DobotRobot(robot_ip=robot_ip_right, robot_number=2)
    robot_dobot = BimanualRobot(_robot_l, _robot_r)
    server_dobot = ZMQServerRobot(robot_dobot, port=port, host=hostname)
    thread_dobot = threading.Thread(target=server_dobot.serve)
    thread_dobot.start()
    print(f"Starting robot server on port {port}")
    # 等待服务器启动，并检查机械臂连接
    time.sleep(1)
    max_retries = 10
    for i in range(max_retries):
        try:
            # 尝试读取机械臂状态以确认连接
            state_r = _robot_r.get_XYZrxryrz_state()
            state_l = _robot_l.get_XYZrxryrz_state()
            print("Robots connected successfully.")
            break
        except Exception as e:
            print(f"Waiting for robots to connect... (attempt {i+1}/{max_retries})")
            time.sleep(1)
    else:
        raise ConnectionError("Failed to connect to robots after multiple attempts.")
    return _robot_l, _robot_r, thread_dobot

def create_transform_matrix(R, T):
    if R is None or T is None:
        return None

    transform = np.eye(4)
    transform[:3, :3] = R
    transform[:3, 3] = T / 1000.0  # 毫米转米
    return transform

def preprocess_point_cloud(point_cloud):
    """
    预处理点云：将新坐标系的点云变换到原始坐标系
    输入: point_cloud (h, w, 3) 单位米
    输出: 变换后的点云 (h, w, 3) 单位米
    """
    # 创建旋转矩阵（绕z轴顺时针旋转90度）
    rotation = R.from_euler('z', rotation_angle, degrees=True)
    rotation_matrix = rotation.as_matrix()

    # 应用旋转
    rotated_points = np.dot(point_cloud.reshape(-1, 3), rotation_matrix.T)

    # 应用平移 (只在z轴方向平移)
    transformed_points = rotated_points + np.array([0, 0, z_translation])

    # 恢复原始形状
    return transformed_points.reshape(point_cloud.shape)

def postprocess_action(action):
    """
    后处理动作：将原始坐标系预测的动作变换回新坐标系
    输入: 动作 (x, y, z, qx, qy, qz, qw) - xyz单位米，四元数
    输出: 变换后的动作 (x, y, z, rx, ry, rz) - xyz单位毫米，欧拉角度
    """
    # 提取位置和四元数
    position_m = action[:3]
    quaternion = action[3:]

    # 创建逆旋转对象（绕z轴逆时针旋转90度）
    inverse_rotation = R.from_euler('z', -rotation_angle, degrees=True)

    # 应用逆平移（先在z轴方向反向平移）
    untranslated_position = position_m - np.array([0, 0, z_translation])

    # 应用逆旋转到位置
    original_position = inverse_rotation.apply(untranslated_position)

    # 处理旋转部分：应用逆旋转到四元数
    predicted_rotation = R.from_quat(quaternion)
    original_rotation = inverse_rotation * predicted_rotation

    # 转换为欧拉角
    euler_angles = original_rotation.as_euler('xyz', degrees=True)

    # 将位置从米转换为毫米
    position_mm = original_position * 1000.0

    # 合并位置和欧拉角
    return np.concatenate([position_mm, euler_angles])

@configurable()
def main(cfg: DictConfig):
    global thread_running
    thread_running = True

    logger = logging.getLogger(__name__)
    logger.info("bash> " + " ".join(map(shlex.quote, sys.argv)))
    log_dir = cfg.output_dir
    logger.info(f"Log dir: {log_dir}") 
    os.makedirs(log_dir, exist_ok=True)
    device = cfg.eval.device
    args = Args()

    py_module = cfg.py_module
    from importlib import import_module
    MOD = import_module(py_module)
    cprint(f"[Info]: Model: {py_module}", "red")
    if py_module== "camera_policy_plus" or py_module== "center_policy" or py_module== "camera_policy_plus_plus":
        Policy, PolicyNetwork = MOD.Policy, MOD.CameraPolicy
    else:
        Policy, PolicyNetwork = MOD.Policy, MOD.PolicyNetwork

    net = PolicyNetwork(cfg.model.hp, cfg.env, render_device=f"cuda:{device}").to(device)
    agent = Policy(net, cfg.model.hp, log_dir=log_dir)

    agent.build(training=False, device=device)
    agent.load(cfg.model.weights)
    agent.eval()
    if hasattr(agent, 'load_clip'):
        agent.load_clip()

    # start robot
    _robot_l, _robot_r, thread_dobot = init_robots(
        robot_ip_left="192.168.5.1",
        robot_ip_right="192.168.5.2",
        port=6001,
        hostname="127.0.0.1"
    )
    realsense_config_path = './realsense_config/'
    camera_dict = load_ini_data_camera()
    rs1 = RealSenseCameraV3(flip=False, device_id=camera_dict["global1"], calibration_dir=realsense_config_path)
    rs2 = RealSenseCameraV3(flip=False, device_id=camera_dict["global3"], calibration_dir=realsense_config_path)
    rs3 = RealSenseCameraV3(flip=False, device_id=camera_dict["right"], calibration_dir=realsense_config_path)
    intrinsics[0] = rs1.get_intrinsics()
    intrinsics[1] = rs2.get_intrinsics()
    intrinsics[2] = rs3.get_intrinsics()
    depth_scales[0] = rs1.get_depth_scale()
    depth_scales[1] = rs2.get_depth_scale()
    depth_scales[2] = rs3.get_depth_scale()

    print("Camera intrinsics and depth scales:")
    print(intrinsics)
    print(depth_scales)

    thread_cam_top = threading.Thread(target=run_thread_cam, args=(rs1, 0))
    thread_cam_left = threading.Thread(target=run_thread_cam, args=(rs2, 1))
    thread_cam_right = threading.Thread(target=run_thread_cam, args=(rs3, 2))
    thread_cam_top.start()
    thread_cam_left.start()
    thread_cam_right.start()
    time.sleep(2)
    print("camera thread init success...")

    # 机械臂运动到固定位置
    # _robot_l.moveJ([1.0030964, - 0.11022995, - 1.5607112, 0.08130229,  1.52524056,  1.71798272, 0.97942708])
    # time.sleep(5)
    _robot_r.moveJ([1.57106819, -0.00330064, 1.56805806, -0.00357336, -1.57398132, -1.57415522, 0.96307964])
    time.sleep(4)

    # print("Waiting to connect the robot...")
    # robot_client = ZMQClientRobot(port=args.robot_port, host=args.hostname)
    # print("If the robot fails to initialize successfully after 5 seconds,please check that the robot network is connected correctly and make sure TCP/IP mode is turned!")
    # env = RobotEnv(robot_client)
    # env.set_do_status([1, 0])
    # env.set_do_status([2, 0])
    # env.set_do_status([3, 0])
    # # robot_pose_init(env)

    # curr_action = env.get_obs()
    # curr_action = curr_action["joint_positions"]
    # target_action = np.array([1.0030964, - 0.11022995, - 1.5607112,   0.08130229,  1.52524056,  1.71798272,
    #                             0.97942708, 1.57106819, -0.00330064, 1.56805806, -0.00357336, -1.57398132,
    #                             -1.57415522, 0.96307964])
    # max_delta = (np.abs(curr_action - target_action)).max()
    # steps = min(int(max_delta / 0.001), 100)
    # for jnt in np.linspace(curr_action, target_action, steps):
    #     env.step(jnt, np.array([1, 1]))
    
    episode_max_length = 60

    assert thread_cam_top.is_alive(), "Error: please check the global1 camera!"
    assert thread_cam_left.is_alive(), "Error: please check the global3 camera!"
    assert thread_cam_right.is_alive(), "Error: please check the right wrist camera!"

    task = 'pick_grape_pcd'
    # task = 'manipulation_dataset_stack_bowls'
    # task = 'manipulation_dataset_pick_grape'
    # task = 'manipulation_dataset_collect_fruits'

    tasks_instruction = {
        'pick_grape_pcd': "Put the grapes into the green plate.",
        "manipulation_dataset_stack_bowls":"Stack a pink bowl and a white bowl on a green plate.",
        "manipulation_dataset_pick_grape":"Put the grapes into the green plate.",
        "manipulation_dataset_collect_fruits":"Put a carrot, a carambola, and an eggplant into the white basin."
    }

    lang_goal = tasks_instruction[task]
    lang_goal_tokens = torch.tensor(clip.tokenize([lang_goal])[0].numpy(), device='cuda')[None, ...].long()

    folder_to_camera_config = {
                'leftImgDeep': CAMERA_CONFIGS['global3'],
                'rightImgDeep': CAMERA_CONFIGS['wrist'],
                'topImgDeep': CAMERA_CONFIGS['global1']
            }
    
    
    top_processor = RealSenseOfflineProcessor(realsense_config_path + f"realsense_calibration_{folder_to_camera_config['topImgDeep'].serial}_lastest.json")
    left_processor = RealSenseOfflineProcessor(realsense_config_path + f"realsense_calibration_{folder_to_camera_config['leftImgDeep'].serial}_lastest.json")
    right_processor = RealSenseOfflineProcessor(realsense_config_path + f"realsense_calibration_{folder_to_camera_config['rightImgDeep'].serial}_lastest.json")
    show_canvas = np.zeros((480, 640*3, 3), dtype=np.uint8)
    processor_dict = {
        'topImgDeep': top_processor,
        'leftImgDeep': left_processor,
        'rightImgDeep': right_processor
    }
    # set_light(env, "green", 1)
    # 开启独立进程，将三视角图片实时保存，保存为mp4视频
    if args.save_video:
        thread_video_save = threading.Thread(target=save_video_thread, args=(log_dir, task))
        thread_video_save.start()
        time.sleep(1)
        print("video save thread started...")



    # if args.save_video:
    #     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #     video_writer = cv2.VideoWriter(os.path.join(log_dir, f'{task}_evaluation_video_{time.strftime("%Y%m%d_%H%M%S")}.mp4'), fourcc, 1, (640*3, 480))



    try:
        for i in range(episode_max_length):
            # 开始推理
            obs_rgb = {
                'topImg': img_list[0],
                'leftImg': img_list[1],
                'rightImg': img_list[2],
            }
            # if args.show_img:
            #     imgs = np.hstack((obs_rgb['leftImg'], obs_rgb['rightImg'], obs_rgb['topImg']))
            #     cv2.imshow("imgs", imgs)
            #     cv2.waitKey(1)
            obs_deep = {
                'topImgDeep': deep_list[0],
                'leftImgDeep': deep_list[1],
                'rightImgDeep': deep_list[2],
            }
            obs_pcd = {}
            low_dim_state = _robot_r._get_gripper_pos()
            for name, depth in obs_deep.items():
                if name == 'leftImgDeep':
                    config = folder_to_camera_config[name]
                    transform_matrix = create_transform_matrix(config.R, config.T)
                    point_cloud = processor_dict[name].get_pcd_optimized(depth, transform_matrix)
                    point_cloud = preprocess_point_cloud(point_cloud)
                    obs_pcd[name.replace('Deep', 'Pcd')] = point_cloud
            
            model_input = {
                'lang_goal_tokens': lang_goal_tokens,
                'low_dim_state': torch.tensor(low_dim_state, device=device).float().unsqueeze(0).unsqueeze(0),
                'left_shoulder_rgb': torch.tensor(obs_rgb['leftImg'], device=device).float().unsqueeze(0).permute(0, 3, 1, 2),
                'left_shoulder_point_cloud': torch.tensor(obs_pcd['leftImgPcd'], device=device).float().unsqueeze(0).permute(0, 3, 1, 2),
            }
            print(model_input['low_dim_state'].shape)
            
            start_time = time.time()
            act_result = agent.act(-1, model_input)
            
            action = act_result.action[:-1]
            gripper_state = action[-1]
            ctrl_action = postprocess_action(action[:-1])
            print(f"Step {i}: Predicted action (x,y,z,rx,ry,rz,gripper): {ctrl_action}, {gripper_state} Time: {time.time() - start_time:.2f}s")
            
            start_time = time.time()
            _robot_r.robot.MovL(
                ctrl_action[0],
                ctrl_action[1],
                ctrl_action[2],
                ctrl_action[3],
                ctrl_action[4],
                ctrl_action[5]
            )
            if _robot_r._use_gripper:
                gripper_pos = int(gripper_state * 255)
                _robot_r.gripper.move(gripper_pos, 100, 1)
            print(f"Step {i}: Action executed. Time: {time.time() - start_time:.2f}s")
            if args.show_img:
                show_canvas[:, :640] = np.asarray(img_list[0][:, :, ::-1], dtype="uint8")
                show_canvas[:, 640:640 * 2] = np.asarray(img_list[1][:, :, ::-1], dtype="uint8")
                show_canvas[:, 640 * 2:640 * 3] = np.asarray(img_list[2][:, :, ::-1], dtype="uint8")
                # if args.save_video:
                #     video_writer.write(show_canvas)

                cv2.imshow("0", show_canvas)
                cv2.waitKey(1)
            time.sleep(3)
    except KeyboardInterrupt:
        print("Program interrupted by user")
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        # if args.save_video:
        #     video_writer.release()
        thread_running = False
        thread_cam_top.join(timeout=1)
        thread_cam_left.join(timeout=1)
        thread_cam_right.join(timeout=1)
        if args.save_video:
            thread_video_save.join(timeout=1)
            print("video save thread closed.")
        rs1.close()
        rs2.close()
        rs3.close()
        cv2.destroyAllWindows()
        print("all camera thread closed.")
        # if i == episode_max_length - 1:
        #     set_light(env, "red", 1)
        #     print("Reach max episode length, task stop!")
    print("Task accomplished")

    # thread_running = False
    # # 关闭相机thread_cam_top
    # thread_cam_top.join(timeout=1)
    # thread_cam_left.join(timeout=1)
    # thread_cam_right.join(timeout=1)

    # rs1.close()
    # rs2.close()
    # rs3.close()

    # cv2.destroyAllWindows()
    
    print("Program finished.")

if __name__ == "__main__":
    main()