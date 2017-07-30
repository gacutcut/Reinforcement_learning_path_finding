import datetime
import itertools
from tkinter import *
from tkinter import messagebox
from tkinter import ttk

import numpy as np
import paho.mqtt.client as mqtt
import pandas as pd

from Path_finding_simulation.variable import *

client = mqtt.Client()
root = Tk()
DEFAULT_SIZE_ROW = 5
DEFAULT_SIZE_COLUMN = 5
MAX_SIZE = 10
GRID_SIZE = 30
PADDING = 5

row_num = DEFAULT_SIZE_ROW
col_num = DEFAULT_SIZE_COLUMN

canvas_list = []
storage_value = []

NOT_USE = -1.0
OBSTACLE = 0.0
EMPTY = 1.0
TARGET = 0.75
START = 0.5

COLOR_NOT_USE = "grey90"
COLOR_OBSTACLE = "red2"
COLOR_EMPTY = "grey70"
COLOR_TARGET = "RoyalBlue1"
COLOR_START = "green"

target_count = 0
start_count = 0

MAX_EPISODES = 4000
LOAD_TRAINED_MODEL_PATH = ""
SAVE_FILE_PATH = "./save/self_drive_master.h5"

DEBUG = False
EPSILON_REDUCE = True
RANDOM_MODE = False

ACTION_LEFT = 0
ACTION_UP = 1
ACTION_RIGHT = 2
ACTION_DOWN = 3

STATE_START = 'start'
STATE_WIN = 'win'
STATE_LOSE = 'lose'
STATE_BLOCKED = 'blocked'
STATE_VALID = 'valid'
STATE_INVALID = 'invalid'

best_path = ""

detected_collision = False


class QLearningTable:
    def __init__(self, env):
        self.actions = [0, 1, 2, 3]  # a list
        self.lr = 0.01
        self.gamma = 0.9
        self.epsilon = 0.9
        self.q_table = pd.DataFrame(columns=self.actions)

    def choose_action(self, current_state, valid_actions):
        self.check_state_exist(current_state)
        # action selection
        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.q_table.ix[np.array_str(current_state), :]
            temp = np.array(state_action.get_values())
            temp = temp.argsort()[-4:][::-1]
            index = -1
            #
            for i in range(len(valid_actions)):
                if temp[i] in valid_actions:
                    index = i
                    break
            action = temp[index]
        else:
            # choose random action
            action = np.random.choice(valid_actions)
        return action

    def learn(self, current_state, action, reward, next_state, game_over):
        self.check_state_exist(np.array_str(next_state))
        q_predict = self.q_table.ix[np.array_str(current_state), action]
        if not game_over:
            q_target = reward + self.gamma * self.q_table.ix[np.array_str(next_state),
                                             :].max()  # next state is not terminal
        else:
            q_target = reward  # next state is terminal
        self.q_table.ix[np.array_str(current_state), action] += self.lr * (q_target - q_predict)  # update

    # check whether the state is exist or not
    def check_state_exist(self, next_state):
        if str(next_state) not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0] * len(self.actions),
                    index=self.q_table.columns,
                    name=str(next_state),
                )
            )


class Environment:
    def __init__(self, row_x, col_y):
        self.row_number = row_x
        self.col_number = col_y
        self.action_size = 4  # 1:UP,2:RIGHT,3:LEFT,4:RIGHT
        self.observation_space = row_x * col_y
        self._map = self._create_map(row_x, col_y)
        self.ready = False

    def _create_map(self, row_number, col_number):
        map = np.ones(shape=(row_number, col_number))
        return map

    def set_target(self, row_x, col_y):
        self.target = (row_x, col_y)
        self._map[row_x, col_y] = TARGET

    def set_collision(self, row_x, col_y):
        self._map[row_x, col_y] = OBSTACLE

    def set_start_point(self, row_x, col_y):
        self.start = (row_x, col_y)
        self.current_state = (row_x, col_y, STATE_START)
        self._map[row_x, col_y] = START

    def set_empty_point(self, row_x, col_y):
        self._map[row_x, col_y] = EMPTY

    def create_random_environment(self):
        self.ready = True
        self._map = self._create_map(self.row_number, self.col_number)
        # There are n number of object included: 1 start point, 1 target and n -2 colision.
        n = min(self.row_number, self.col_number) + 1
        count = 0
        random_set = np.empty(shape=(n, 2))
        while count < n:
            x = np.random.randint(self.row_number)
            y = np.random.randint(self.col_number)
            if ([x, y] in random_set.tolist()):
                continue
            random_set[count, 0] = x
            random_set[count, 1] = y
            count += 1
        self.set_start_point(int(random_set[0, 0]), int(random_set[0, 1]))
        self.set_target(int(random_set[1, 0]), int(random_set[1, 1]))
        for i in range(2, n):
            self.set_collision(int(random_set[i, 0]), int(random_set[i, 1]))

    # reset environment the default state
    def reset(self):
        if RANDOM_MODE:
            self.create_random_environment()
        row_x, col_y = self.start
        self.current_state = (row_x, col_y, STATE_START)
        self.visited = set()
        self.min_reward = - 1.5 * self.observation_space
        self.free_cells = [(r, c) for r in range(self.row_number) for c in range(self.col_number) if
                           self._map[r, c] == 1.0]
        # self.free_cells.remove(self.target)
        self.total_reward = 0
        self.map = np.copy(self._map)
        for row, col in itertools.product(range(self.row_number), range(self.col_number)):
            storage_value[row, col] = self._map[row, col]
        updateCanvas()

    #return the valid action for each state
    def valid_actions(self, cell=None):
        if cell is None:
            row, col, mode = self.current_state
        else:
            row, col = cell
        actions = [0, 1, 2, 3]
        nrows, ncols = self.map.shape
        if row == 0:
            actions.remove(1)
        elif row == nrows - 1:
            actions.remove(3)

        if col == 0:
            actions.remove(0)
        elif col == ncols - 1:
            actions.remove(2)

        if row > 0 and self.map[row - 1, col] == 0.0:
            actions.remove(1)
        if row < nrows - 1 and self.map[row + 1, col] == 0.0:
            actions.remove(3)

        if col > 0 and self.map[row, col - 1] == 0.0:
            actions.remove(0)
        if col < ncols - 1 and self.map[row, col + 1] == 0.0:
            actions.remove(2)
        return actions

    def update_state(self, action):
        nrow, ncol, nmode = current_row, current_col, mode = self.current_state

        if self.map[current_row, current_col] > 0.0:
            self.visited.add((current_row, current_col))

        valid_actions = self.valid_actions()

        if not valid_actions:
            nmode = STATE_BLOCKED
        elif action in valid_actions:
            nmode = STATE_VALID
            storage_value[nrow, ncol] = EMPTY
            if action == ACTION_LEFT:
                storage_value[nrow, ncol - 1] = START
                ncol -= 1
            elif action == ACTION_UP:
                storage_value[nrow - 1, ncol] = START
                nrow -= 1
            if action == ACTION_RIGHT:
                storage_value[nrow, ncol + 1] = START
                ncol += 1
            elif action == ACTION_DOWN:
                storage_value[nrow + 1, ncol] = START
                nrow += 1
        else:
            # invalid action. don't move the agent
            nmode = STATE_INVALID
        self.current_state = (nrow, ncol, nmode)

    # Action define:
    # 0: LEFT
    # 1: UP
    # 2: RIGHT
    # 3: DOWN
    def act(self, act):
        self.update_state(act)
        updateCanvas()
        reward = self.get_reward()
        self.total_reward += reward
        status = self.game_status()
        current_state = self.observe()
        return current_state, reward, status

    def observe(self):
        canvas = self.draw_env()
        envstate = canvas.reshape((1, -1))
        return envstate

    def draw_env(self):
        canvas = np.copy(self.map)
        nrows, ncols = self.map.shape
        # clear all visual marks
        for r in range(nrows):
            for c in range(ncols):
                if canvas[r, c] > 0.0:
                    canvas[r, c] = 1.0
        row, col, valid = self.current_state
        canvas[row, col] = 0.5  # set current position
        return canvas

    #return status fo game
    def game_status(self):
        if self.total_reward < self.min_reward:
            return STATE_LOSE
        current_row, current_col, mode = self.current_state
        target_row, target_col = self.target
        if current_row == target_row and current_col == target_col:
            return STATE_WIN
        return STATE_VALID

    #reinforcement learning reward function
    # return 1 if find the target
    # return min_reward-1 if obstacles block the way
    # return -0.75 if state invalid
    # return -0.25 if visited
    # return -0.05 for one possible state
    def get_reward(self):
        current_row, current_col, mode = self.current_state
        target_row, target_col = self.target
        if current_row == target_row and current_col == target_col:
            return 1.0
        if mode == STATE_BLOCKED:
            return self.min_reward - 1
        if mode == STATE_INVALID:
            return -0.75
        if (current_row, current_col) in self.visited:
            return -0.25
        if mode == STATE_VALID:
            return -0.05


# Format time string to minutes
def format_time(seconds):
    m = seconds / 60.0
    return "%.2f minutes" % (m,)


def check_best_path(memory):
    global best_path
    global detected_collision
    min = 100
    index = -1
    for i in range(len(memory)):
        if (len(memory[i]) < min):
            min = len(memory[i])
            index = i
    best_path = ''.join(str(x) for x in memory[index])
    btnSendQuery.configure(state=NORMAL)
    if detected_collision:
        send_message("Path" + best_path)
        detected_collision = False
    print(best_path)


def deepQLearning(model, env, randomMode=False, **opt):
    global saved_env
    episodes = opt.get('n_epoch', MAX_EPISODES)
    win_history = []
    memory = []
    win_rate = 0.0
    start_time = datetime.datetime.now()
    history_size = env.observation_space
    for episode in range(episodes):
        env.reset()
        game_over = False
        # number of step for each episode
        n_step = 0
        list_action = []
        next_state = env.map.reshape((1, -1))
        while not game_over:
            valid_actions = env.valid_actions()
            if not valid_actions:
                game_over = True
                print(env.map)
                continue

            current_state = next_state
            # Get next action
            action = model.choose_action(current_state, valid_actions)
            # print("**action = {}".format(action))
            # Apply action, get reward and new envstate
            next_state, reward, game_status = env.act(action)
            # print("action = {}, valid_actions = {}".format(action, valid_actions))
            if game_status == STATE_WIN:
                x, y, _ = env.current_state
                storage_value[x, y] = TARGET
                win_history.append(1)
                game_over = True
            elif game_status == STATE_LOSE:
                x, y, _ = env.current_state
                storage_value[x, y] = EMPTY
                win_history.append(0)
                game_over = True
            else:
                game_over = False

            if DEBUG:
                print("--------------------------------------")
                print(np.reshape(current_state, newshape=(4, 4)))
                print("action = {},valid_action = {},reward = {}, game_over = {}".format(action, valid_actions,
                                                                                         reward, game_over))
                print(np.reshape(next_state, newshape=(4, 4)))
            list_action.append(action)
            # Store episode (experience)
            model.learn(current_state, action, reward, next_state, game_over)
            n_step += 1
            print(env.map)
        if not EPSILON_REDUCE:
            if win_rate > 0.9:
                model.epsilon = 0.05
        if len(win_history) > history_size:
            win_rate = sum(win_history[-history_size:]) / history_size
        dt = datetime.datetime.now() - start_time
        t = format_time(dt.total_seconds())
        x_target, y_target = env.target
        # env.map[x_target, y_target] = 2
        # print(env.map)
        # print(list_action)
        memory.append(list_action)
        template = "Episodes: {:03d}/{:d} | Total_reward: {:3.4f} | Episodes: {:d} | Total win: {:d} | Win rate: {:.3f} | time: {}"
        print(template.format(episode, MAX_EPISODES - 1, env.total_reward, n_step,
                              sum(win_history),
                              win_rate, t))
        if sum(win_history[-history_size:]) == history_size:
            env.reset()
            saved_env = env
            print("Reached 100%% win rate at episode: %d" % (episode,))
            messagebox.showinfo("Result for Q Learning",
                                "Have 100% win rate at episode {:d}. \nWin {:d}/{:d} game during {}".format(episode,
                                                                                                            sum(
                                                                                                                win_history),
                                                                                                            episode, t))
            check_best_path(memory)
            break


def buttonClick():
    global target_count
    global start_count
    row_num = int(e1.get())
    col_num = int(e2.get())
    for row, col in itertools.product(range(10), range(10)):
        target_count = 0
        start_count = 0
        storage_value[row, col] = EMPTY
        canvas.itemconfigure(canvas_list[row, col], fill="grey90")
    for row, col in itertools.product(range(row_num), range(col_num)):
        canvas.itemconfigure(canvas_list[row, col], fill=COLOR_EMPTY)
        storage_value[row, col] = EMPTY


def getPosition(widget_num):
    row = np.math.floor(widget_num / MAX_SIZE)
    col = widget_num % MAX_SIZE - 1
    return row, col


def printMatrx():
    row_num = int(e1.get())
    col_num = int(e2.get())
    print(storage_value[0:row_num, 0:col_num])


def updateCanvas():
    row_num = int(e1.get())
    col_num = int(e2.get())
    for row, col in itertools.product(range(row_num), range(col_num)):
        if storage_value[row, col] == NOT_USE:
            canvas.itemconfigure(canvas_list[row, col], fill=COLOR_NOT_USE)
        elif storage_value[row, col] == OBSTACLE:
            canvas.itemconfigure(canvas_list[row, col], fill=COLOR_OBSTACLE)
        elif storage_value[row, col] == EMPTY:
            canvas.itemconfigure(canvas_list[row, col], fill=COLOR_EMPTY)
        elif storage_value[row, col] == TARGET:
            canvas.itemconfigure(canvas_list[row, col], fill=COLOR_TARGET)
        elif storage_value[row, col] == START:
            canvas.itemconfigure(canvas_list[row, col], fill=COLOR_START)
    root.update()


def onObjectClick(event):
    global start_count
    global target_count
    widget_num = event.widget.find_closest(event.x, event.y)[0]
    row, col = getPosition(widget_num)
    row_num = int(e1.get())
    col_num = int(e2.get())
    if (row < row_num and col < col_num):
        current_value = storage_value[row, col]
        valid_value = np.array([EMPTY, START, TARGET, OBSTACLE], dtype=np.float)
        # print("Start count = {}, target_count = {}".format(start_count, target_count))
        if (start_count == 1):
            valid_value = np.delete(valid_value, np.argwhere(valid_value == START))
        if (target_count == 1):
            valid_value = np.delete(valid_value, np.argwhere(valid_value == TARGET))
        if (current_value == START or current_value == TARGET):
            index = -1
        else:
            index = np.where(valid_value == current_value)[0][0]
        if (index != (len(valid_value) - 1)):
            next_value = valid_value[index + 1]
        else:
            next_value = valid_value[0]
        if (next_value == START):
            start_count += 1
        elif (next_value == TARGET):
            target_count += 1
        if (current_value == START):
            start_count = 0
        elif (current_value == TARGET):
            target_count = 0
        storage_value[row, col] = next_value
        # print("current= {},next = {},index = {}, valid = {}".format(current_value, next_value, index, valid_value))
        if next_value == NOT_USE:
            canvas.itemconfigure(canvas_list[row, col], fill=COLOR_NOT_USE)
        elif next_value == OBSTACLE:
            canvas.itemconfigure(canvas_list[row, col], fill=COLOR_OBSTACLE)
        elif next_value == EMPTY:
            canvas.itemconfigure(canvas_list[row, col], fill=COLOR_EMPTY)
        elif next_value == TARGET:
            canvas.itemconfigure(canvas_list[row, col], fill=COLOR_TARGET)
        elif next_value == START:
            canvas.itemconfigure(canvas_list[row, col], fill=COLOR_START)


def create_environment():
    if (start_count == 0 or target_count == 0):
        messagebox.showinfo("Error", "Please set START and TARGET point")
        return None
    row_num = int(e1.get())
    col_num = int(e2.get())
    env = Environment(row_num, col_num)
    for row, col in itertools.product(range(row_num), range(col_num)):
        if storage_value[row, col] == START:
            env.set_start_point(row, col)
        elif storage_value[row, col] == TARGET:
            env.set_target(row, col)
        elif storage_value[row, col] == OBSTACLE:
            env.set_collision(row, col)
    return env


# Function to create reinforcement learning agent
def trainDQN():
    env = create_environment()
    if env is None:
        return
    model = QLearningTable(env)
    deepQLearning(model, env, save_file_path="gacutcut.h5")
    pass


def sendEV3Query():
    send_message("Path" + best_path)
    pass


def send_message(COMMAND):
    client.publish(TOPIC_SUBSCRIBE_TO_EV3, COMMAND)


# when receive message
def on_message(client, userdata, msg):
    global detected_collision

    # receive message from EV3: Only in case detect the obstacles
    # check the position of the obstacle and current position of robot
    if (msg.topic == TOPIC_SUBSCRIBE_FROM_EV3):
        pass_step = int(msg.payload.decode())
        print("get Message = {}".format(pass_step))
        detected_collision = True
        action_list = []
        for i in range(len(best_path)):
            action_list.append(int(best_path[i]))
        env = create_environment()
        start_row_x, start_col_y = env.start
        env.set_empty_point(start_row_x, start_col_y)
        for i in range(pass_step):
            if (action_list[i] == ACTION_LEFT):
                start_col_y -= 1
            elif (action_list[i] == ACTION_UP):
                start_row_x -= 1
            elif (action_list[i] == ACTION_RIGHT):
                start_col_y += 1
            elif (action_list[i] == ACTION_DOWN):
                start_row_x += 1
        # because it stop in colision so the next step will be collision
        collision_x = start_row_x
        collision_y = start_col_y
        if (action_list[pass_step] == ACTION_LEFT):
            collision_y -= 1
        elif (action_list[pass_step] == ACTION_UP):
            collision_x -= 1
        elif (action_list[pass_step] == ACTION_RIGHT):
            collision_y += 1
        elif (action_list[pass_step] == ACTION_DOWN):
            collision_x += 1

        env.set_start_point(start_row_x, start_col_y)
        env.set_collision(collision_x, collision_y)
        env.reset()
        messagebox.showinfo("Collision detected", "Have unknown collision. Continue to train model!!")


# Call back when connect to MQTT Broker
def on_connect(client, userdata, rc):
    if rc == 0:
        print("Connect successful to Broker!!!")
    else:
        print("Connected with result code " + str(rc))
    # subscribe to all topic
    client.subscribe(TOPIC_SUBSCRIBE_ALL)


# Start MQTT Client
def start_mqtt_subscribe(client):
    client.connect(MQTT_SERVER_ADDRESS, MQTT_SERVER_PORT, 10)
    client.on_connect = on_connect
    client.on_message = on_message
    client.loop()


def publish(client):
    client.disconect()
    print("Disconnected")


def resetEV3Query():
    send_message("reset")
    pass


if __name__ == "__main__":
    root.title("qlearning_path_finding")
    # start_mqtt_subscribe(client)
    # client.loop_start()
    w = 350  # width for the Tk root
    h = 600  # height for the Tk root

    # get screen width and height
    ws = root.winfo_screenwidth()  # width of the screen
    hs = root.winfo_screenheight()  # height of the screen

    # calculate x and y coordinates for the Tk root window
    x = (ws / 2) - (w / 2)
    y = (hs / 2) - (h / 2)
    ttk.Style().configure('green/black.TButton', foreground='green', background='black')
    # set the dimensions of the screen
    # and where it is placed
    root.geometry('%dx%d+%d+%d' % (w, h, x, y))
    frame1 = Frame(root)
    frame1.pack()
    frame2 = Frame(root)
    frame2.pack()
    frame3 = Frame(root)
    frame3.pack()
    frame4 = Frame(root)
    frame4.pack()
    array = []

    Label(frame1, text="Number of row").grid(row=0, sticky=W)
    Label(frame1, text="Number of column").grid(row=1, sticky=W)
    e1 = Entry(frame1, width="3")
    e2 = Entry(frame1, width="3")
    e1.insert(0, DEFAULT_SIZE_ROW)
    e2.insert(0, DEFAULT_SIZE_COLUMN)
    e1.grid(row=0, column=1)
    e2.grid(row=1, column=1)
    Button(frame1, text="Generate size", command=buttonClick) \
        .grid(row=0, column=2, columnspan=2, rowspan=2)
    Label(frame1, text="START", bg=COLOR_START).grid(row=2, column=0, sticky=W + E + N + S)
    Label(frame1, text="TARGET", bg=COLOR_TARGET).grid(row=2, column=1, sticky=W + E + N + S)
    Label(frame1, text="OBSTACLES", bg=COLOR_OBSTACLE).grid(row=2, column=2, sticky=W + E + N + S)
    canvas = Canvas(frame2, width=360, height=360)
    Button(frame3, text="Train with Q Learning", command=trainDQN).pack()
    Button(frame4, text="Reset EV3 direction", command=resetEV3Query).pack(side=LEFT)
    btnSendQuery = Button(frame4, text="Start run EV3", command=sendEV3Query, state=DISABLED)
    btnSendQuery.pack(side=RIGHT)

    for row, col in itertools.product(range(10), range(10)):
        x1 = col * (GRID_SIZE + PADDING)
        y1 = row * (GRID_SIZE + PADDING)
        x2 = x1 + GRID_SIZE
        y2 = y1 + GRID_SIZE
        canvas_item = canvas.create_rectangle(x1, y1, x2, y2, fill='grey90', width=0)
        canvas_list.append(canvas_item)
        canvas.tag_bind(canvas_item, '<ButtonPress-1>', onObjectClick)
        storage_value.append(NOT_USE)
    canvas_list = np.array(canvas_list).reshape(10, 10)
    storage_value = np.array(storage_value, dtype=np.float).reshape(10, 10)
    canvas.pack()
    root.call('wm', 'attributes', '.', '-topmost', '2')
    root.mainloop()

    # ==================================
    # Define graphic user interface - END
    # ==================================
