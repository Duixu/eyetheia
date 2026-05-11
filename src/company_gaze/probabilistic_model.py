import torch
import numpy as np
import time
import torch.nn as nn
import torch.nn.functional as F
import os
import sys


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )
        self.out = nn.Linear(hidden_size, 1)

    def forward(self, x, h):
        out, h = self.rnn(x, h)
        prediction = self.out(h)
        return prediction, h

class biLSTM():
    def __init__(self, input_size, hidden_size):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )
        self.out = nn.Linear(hidden_size, 1)

    def forward(self, x, h):
        out, h = self.rnn(x, h)
        prediction = self.out(h)
        return prediction, h


class rnn_method():
    def __init__(self, buffer_size=5, device=None, weight_path=None):
        self.position_buffer = np.zeros((buffer_size, 2))
        self.velocity_buffer = np.zeros((buffer_size, 2))
        self.accelerate_buffer = np.zeros((buffer_size, 2))
        self.time_buffer = np.zeros(buffer_size)
        self.buffer_count = 0
        self.buffer_size = buffer_size

        self.state_buffer = np.zeros((buffer_size, 2, 2))
        self.state = -1
        self.duration_time = 0
        self.state_count = 0
        self.distance = 700

        self.face = [0, 0, 0]
        self.wait = [0, 0, 0]
        self.time = time.time()
        self.thres = 70

        self.device = device if device is not None else (torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.rnn = RNN(5, 10).to(self.device)
        self.gaze = [0,0]
        # 自动查找权重路径
        if weight_path is None:
            root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            ckpt_path = os.path.join(root_dir, 'checkpoints', 'rnn50.pt')
            model_path = os.path.join(root_dir, 'models', 'rnn50.pt')
            if os.path.exists(ckpt_path):
                weight_path = ckpt_path
            else:
                weight_path = None
        if weight_path is not None:
            self.load(weight_path)
        else:
            print('警告：未找到rnn50.pt权重文件，RNN模型未加载权重！')

    def store(self, gaze, face):

        if np.linalg.norm(np.array(face) - np.array(self.face)) > 200:
            if np.linalg.norm(np.array(face) - np.array(self.wait)) < 100:
                self.face = face
                self.wait = [0, 0, 0]
                self.position_buffer = np.zeros((self.buffer_size, 2))
                self.velocity_buffer = np.zeros((self.buffer_size, 2))
                self.accelerate_buffer = np.zeros((self.buffer_size, 2))
                self.time_buffer = np.zeros(self.buffer_size)
                self.buffer_count = 0

            else:
                self.wait = face
                return

        time_tmp = time.time()
        interval = (time_tmp - self.time)
        self.time = time_tmp
        self.wait = [0, 0, 0]
        self.gaze = gaze

        self.position_buffer[self.buffer_count % self.buffer_size, :] = gaze

        if self.buffer_count >= 6:

            gaze = np.mean(self.position_buffer, axis=0)
            self.position_buffer[(self.buffer_count - 1) % self.buffer_size, :] = self.gaze
            velocity = [(self.gaze[0] - self.position_buffer[(self.buffer_count - 2) % self.buffer_size][0]) / interval,
                        (self.gaze[1] - self.position_buffer[(self.buffer_count - 2) % self.buffer_size][1]) / interval]
            self.velocity_buffer[(self.buffer_count-1) % self.buffer_size, :] = velocity
            accelerate = [
                    (velocity[0] - self.velocity_buffer[(self.buffer_count - 2) % self.buffer_size][0]) / interval,
                    (velocity[1] - self.velocity_buffer[(self.buffer_count - 2) % self.buffer_size][1]) / interval]
            self.accelerate_buffer[(self.buffer_count-1) % self.buffer_size, :] = accelerate

            

        self.position_buffer[self.buffer_count % self.buffer_size, :] = gaze
        if self.buffer_count > 0:

            velocity = [(gaze[0] - self.position_buffer[(self.buffer_count - 1) % self.buffer_size][0]) / interval,
                        (gaze[1] - self.position_buffer[(self.buffer_count - 1) % self.buffer_size][1]) / interval]
            # print(velocity)
            self.velocity_buffer[self.buffer_count % self.buffer_size, :] = velocity
            self.time_buffer[self.buffer_count % self.buffer_size] = interval

            if self.buffer_count > 1:
                accelerate = [
                    (velocity[0] - self.velocity_buffer[(self.buffer_count - 1) % self.buffer_size][0]) / interval,
                    (velocity[1] - self.velocity_buffer[(self.buffer_count - 1) % self.buffer_size][1]) / interval]
                self.accelerate_buffer[self.buffer_count % self.buffer_size, :] = accelerate

        self.buffer_count += 1

    def analysis(self):
        if self.buffer_count < 6 or self.wait[0]!=0 or self.wait[1]!=0 or self.wait[2]!=0:
            return -1, 0, 0
        seq = np.zeros((1, 5, 5))
        index = self.buffer_count % self.buffer_size
        seq[0, 0:self.buffer_size - index, 0:2] = self.position_buffer[index:self.buffer_size, :]
        seq[0, self.buffer_size - index:self.buffer_size, 0:2] = self.position_buffer[0:index, :]
        seq[0, 0:self.buffer_size - index, 2:4] = self.velocity_buffer[index:self.buffer_size, :]
        seq[0, self.buffer_size - index:self.buffer_size, 2:4] = self.velocity_buffer[0:index, :]
        seq[0, 0:self.buffer_size - index, 4] = self.time_buffer[index:self.buffer_size]
        seq[0, self.buffer_size - index:self.buffer_size, 4] = self.time_buffer[0:index]
        with torch.no_grad():
            input = torch.from_numpy(seq).float().to(self.device)
            h = torch.zeros([1, 1, 10]).float().to(self.device)
            pred, h = self.rnn(input, h)
            pred = F.sigmoid(pred)
            prediction = pred.cpu().detach().numpy()

        return prediction[0][0][0] > 0.5, np.mean(self.position_buffer, axis=0), np.mean(self.velocity_buffer, axis=0)

    def load(self, file):
        self.rnn.load_state_dict(torch.load(file, map_location=self.device))
        self.rnn.eval()


class thres_method():
    def __init__(self, buffer_size=5):
        self.position_buffer = np.zeros((buffer_size, 2))  # 改为2列，与gaze数据匹配
        self.velocity_buffer = np.zeros((buffer_size, 2))
        self.accelerate_buffer = np.zeros((buffer_size, 2))
        self.buffer_count = 0
        self.buffer_size = buffer_size

        self.state_buffer = np.zeros((buffer_size, 2, 2))
        self.state = -1
        self.duration_time = 0
        self.state_count = 0

        self.face = [0, 0, 0]
        self.wait = [0, 0, 0]
        self.time = time.time()
        self.thres = 70

    def store(self, gaze, face):

        if np.linalg.norm(np.array(face) - np.array(self.face)) > 200:
            if np.linalg.norm(np.array(face) - np.array(self.wait)) < 100:
                self.face = face
                self.wait = [0, 0, 0]
                self.position_buffer = np.zeros((self.buffer_size, 2))
                self.velocity_buffer = np.zeros((self.buffer_size, 2))
                self.accelerate_buffer = np.zeros((self.buffer_size, 2))
                self.buffer_count = 0

            else:
                self.wait = face
                return
        time_tmp = time.time()
        interval = time_tmp - self.time
        self.time = time_tmp
        self.wait = [0, 0, 0]
        self.position_buffer[self.buffer_count % self.buffer_size, :] = gaze
        if self.buffer_count > 0:

            velocity = [(gaze[0] - self.position_buffer[(self.buffer_count - 1) % self.buffer_size][0]) / interval,
                        (gaze[1] - self.position_buffer[(self.buffer_count - 1) % self.buffer_size][1]) / interval]

            self.velocity_buffer[self.buffer_count % self.buffer_size, :] = velocity

            if self.buffer_count > 1:
                accelerate = [
                    (velocity[0] - self.velocity_buffer[(self.buffer_count - 1) % self.buffer_size][0]) / interval,
                    (velocity[1] - self.velocity_buffer[(self.buffer_count - 1) % self.buffer_size][1]) / interval]
                self.accelerate_buffer[self.buffer_count % self.buffer_size, :] = accelerate

        self.buffer_count += 1

    def analysis(self):
        if self.buffer_count < 6 or self.wait[0]!=0 or self.wait[1]!=0 or self.wait[2]!=0:
            return -1, 0, 0
        #由速度分量计算总的速度，速度分量是x,y方向的速度分量， 返回值 0:注视，1:扫视
        total_velocity = np.sqrt(
            self.velocity_buffer[self.buffer_count % self.buffer_size, 0] ** 2 + self.velocity_buffer[
                self.buffer_count % self.buffer_size, 1] ** 2)
        if total_velocity < self.thres:
            return 0, np.mean(self.position_buffer, axis=0), np.mean(self.velocity_buffer, axis=0)
        else:
            return 1, np.mean(self.position_buffer, axis=0), np.mean(self.velocity_buffer, axis=0)
