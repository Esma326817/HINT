from openpi_client import image_tools
from openpi_client import websocket_client_policy
from transforms3d.axangles import mat2axangle, axangle2mat
from transforms3d.quaternions import quat2mat
from easydict import EasyDict as edict
import numpy as np

class PolicyInference:
    def __init__(self, host="localhost", port=8000, prompt="transfer_ethernet_cable_from_hub_to_ipc"):
        self.client = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)
        self.prompt = prompt
        print(f"Policy client initialized: {host}:{port}")
    
    @staticmethod
    def axisangle_to_rotation6d(axis_angle):
        axis_angle = np.asarray(axis_angle, dtype=np.float64)
        angle = np.linalg.norm(axis_angle)
        
        if angle < 1e-12:
            R = np.eye(3)
        else:
            axis = axis_angle / angle
            R = axangle2mat(axis, angle)
        
        r6d = R[:, :2].flatten()
        return r6d

    @staticmethod
    def axisangle2quat(v):
        v = np.asarray(v, dtype=np.float64)
        angle = np.linalg.norm(v)
        
        if angle < 1e-12:
            return np.array([1.0, 0.0, 0.0, 0.0])
        
        axis = v / angle
        half_angle = angle / 2.0
        w = np.cos(half_angle)
        xyz = np.sin(half_angle) * axis
        return np.concatenate(([w], xyz))

    @staticmethod
    def quat2axisangle(quat):
        mat = quat2mat(quat)
        axis, angle = mat2axangle(mat)
        return axis * angle

    def infer(self, image, state):
        if len(state) == 7:
            pos = state[0:3]
            quat = state[3:7]
            axis_angle = self.quat2axisangle(quat)
            processed_state = np.concatenate([pos, axis_angle])
        elif len(state) == 6:
            processed_state = state
        else:
            raise ValueError(f"State dimension should be 6 or 7, got {len(state)}")
        
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        resized_image = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(image, 224, 224)
        )
        
        observation = {
            "observation/image": resized_image,
            "observation/state": processed_state,
            "prompt": self.prompt,
        }
        
        action_chunk = self.client.infer(observation)["actions"]
        
        converted_actions = []
        for i in range(action_chunk.shape[0]):
            action = action_chunk[i]
            pos = action[0:3]
            axis_angle = action[3:6]
            gripper = action[6:7]

            quat = self.axisangle2quat(axis_angle)
            full_action = np.concatenate([pos, quat, gripper])  # shape (10,)
            # Convert axis-angle to 6D rotation representation
            # rot6d = self.axisangle_to_rotation6d(axis_angle)  # shape (6,)
            # full_action = np.concatenate([pos, rot6d, gripper])  # shape (10,)
            converted_actions.append(full_action)  #shape (N, 8)

        converted_actions = np.array(converted_actions)
        tcp = converted_actions[:, :7]          # (N, 7)
        ee_command = converted_actions[:, 7:8]  # (N, 1)

        return edict({
            "tcp": tcp,
            "ee_command": ee_command
        })


def create_policy_inference(host="localhost", port=8000, prompt="transfer_ethernet_cable_from_hub_to_ipc"):
    return PolicyInference(host=host, port=port, prompt=prompt)

if __name__ == "__main__":
    policy = create_policy_inference()
    
    test_img = np.ones((480, 640, 3), dtype=np.uint8) * 255
    
    state_quat = np.array([0.42074126, 0.38972884, -0.31186837, -0.02869474, -0.11622322, 0.9927717, 0.00855146])
    
    actions = policy.infer(test_img, state_quat)
    print(actions.keys())  # dict_keys(['tcp', 'ee_command'])
    print(actions.tcp.shape)        # (50, 9)
    print(actions["ee_command"].shape) # (50, 1)
