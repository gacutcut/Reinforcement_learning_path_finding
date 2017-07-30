#!/usr/bin/env python3
# so that script can be run from Brickman

import termios, tty, sys
from ev3dev.ev3 import *
import paho.mqtt.client as mqtt
from Variable import *
from time import sleep

SPEED = 700
THRESHOLD = 80
rate = 0.6
DEFAULT_DIRECTION = 3

DIRECTION_LEFT = 0
DIRECTION_UP = 1
DIRECTION_RIGHT = 2
DIRECTION_DOWN = 3

current_direction = DEFAULT_DIRECTION


# This is the Subscriber

def on_connect(client, userdata, flags, rc):
    print("Connected with result code " + str(rc))
    client.subscribe(TOPIC_SUBSCRIBE_TO_EV3)


def check_message(message):
    global SPEED
    global current_direction
    print(message)
    if (message == DRIVER_STATUS_FORWARD):
        forward()
    elif (message == DRIVER_STATUS_BACK):
        back()
    elif (message == DRIVER_STATUS_LEFT):
        left()
    elif (message == DRIVER_STATUS_RIGHT):
        right()
    elif (message == DRIVER_STATUS_TURN_OPPOSITE):
        turn_opposite()
    elif (message == DRIVER_STATUS_TURN_LEFT):
        turn_left()
    elif (message == DRIVER_STATUS_TURN_RIGHT):
        turn_right()
    elif (message == DRIVER_STATUS_STOP):
        stop()
    elif (message == "reset"):
        # reset to default direction
        current_direction = DEFAULT_DIRECTION
    elif (message == DRIVER_STATUS_EXIT):
        stop()
        exit()
    elif (message.isdigit()):
        SPEED = int(message)
    elif (message[:2] == "Go"):
        result = goStraight(int(message[2:]))
        print(result)
    elif (message[:4] == "Path"):
        command = message[4:]
        result = processCommand(command)
        print("Number of passed step is {}".format(result))
        if (result == len(command)):
            Sound.play('finish.wav')
        else:
            send_message(result)


def on_message(client, userdata, msg):
    check_message(msg.payload.decode())


def send_message(COMMAND):
    print("Send message = {}".format(COMMAND))
    client.publish(TOPIC_SUBSCRIBE_FROM_EV3, COMMAND)


# There are 4 direction : 0-LEFT, 1 - UP, 2 - RIGHT, 3-LEFT
# this function for decided which direction EV3 need to turn
def doCommand(current_step, count):
    global current_direction
    different = current_direction - current_step
    if (different == 2 or different == -2):
        turn_opposite()
    elif (different == 1 or different == -3):
        turn_left()
    elif (different == -1 or different == 3):
        turn_right()
    current_direction = current_step
    return goStraight(count)


def processCommand(command):
    print("command = {}".format(command))
    action_list = []
    for i in range(len(command)):
        action_list.append(int(command[i]))
    # append dump last value
    action_list.append(-1)
    count_step = 0
    i = 0
    while i < len(action_list):
        current_step = action_list[i]
        next_step = action_list[i + 1]
        count = 1
        while current_step == next_step:
            count += 1
            i += 1
            next_step = action_list[i + 1]
        i += 1
        n = doCommand(current_step, count)
        count_step += n
        print("current = {}, number = {}".format(current_step, count))
        # this is case have avoidance in opposite
        if (n != count or i == (len(action_list) - 1)):
            break
    return count_step


# attach large motors to ports B and C, medium motor to port A
print("Start get motor information")
motor_left = LargeMotor('outB')
motor_right = LargeMotor('outC')


# ==============================================

def getch():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(3)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


# ==============================================

def forward():
    print("Running forward")
    motor_left.run_forever(speed_sp=SPEED)
    motor_right.run_forever(speed_sp=SPEED)


# ==============================================

def back():
    print("Running back")
    motor_left.run_forever(speed_sp=-SPEED)
    motor_right.run_forever(speed_sp=-SPEED)


# ==============================================

def left():
    print("Turn left")
    motor_left.run_forever(speed_sp=SPEED)
    motor_right.run_forever(speed_sp=-SPEED)
    time.sleep(TURN_TIME)
    forward()


# ==============================================

def right():
    print("Turn right")
    motor_left.run_forever(speed_sp=-SPEED)
    motor_right.run_forever(speed_sp=SPEED)
    time.sleep(TURN_TIME)
    forward()


# ==============================================

def stop():
    print("Stop")
    motor_left.run_forever(speed_sp=0)
    motor_right.run_forever(speed_sp=0)


# ==============================================
def turn_opposite():
    # 450 for speed will turn EV3 almost 180 degree
    motor_left.run_forever(speed_sp=470)
    motor_right.run_forever(speed_sp=-470)
    time.sleep(0.8)
    stop()


# ==============================================
def turn_left():
    # 450 for speed will turn EV3 almost 180 degree
    motor_left.run_forever(speed_sp=-470)
    motor_right.run_forever(speed_sp=470)
    time.sleep(0.4)
    stop()


# ==============================================
def turn_right():
    # 450 for speed will turn EV3 almost 180 degree
    motor_left.run_forever(speed_sp=470)
    motor_right.run_forever(speed_sp=-470)
    time.sleep(0.4)
    stop()


# ==============================================

# ==============================================
def turn_motor(speedLeft, speedRight):
    # 450 for speed will turn EV3 almost 180 degree
    motor_left.run_forever(speed_sp=speedLeft)
    motor_right.run_forever(speed_sp=speedRight)


def getRate(clValue):
    return (100 - clValue) / 100 * float(rate) + 1


def isObstacles():
    if (ir.value() < 20):
        return True
    else:
        return False


# num is number of Intersection
def goStraight(num):
    t = 0
    count = 0
    last_t = -1
    while True:
        t = t + 1
        if btn.any():  # Checks if any button is pressed.
            stop()
            exit()
        clValue = cl.value()
        cl2Value = cl2.value() * DIFFER_SENSOR_RATE  # Due to reason 2 sensor doesn't return the same calibration
        differ = clValue - cl2Value
        if (abs(differ) < 15):
            if (clValue < 30):
                last_t = t
            turn_motor(SPEED * 0.5, SPEED * 0.5)
        else:
            turn_motor(((clValue - 50) / 100) * SPEED, ((cl2Value - 50) / 100) * SPEED)
        if ((t - last_t) == 1):
            count += 1
            Sound.beep()
            if (count == num):
                break
        if (isObstacles()):
            break
    stop()
    return count


cl = ColorSensor('in1')
cl2 = ColorSensor('in2')
ir = InfraredSensor()
btn = Button()

assert cl.connected, "Connect a color sensor to sensor port 1"
assert cl2.connected, "Connect a color sensor to sensor port 2"
assert ir.connected, "Connect a single infrared sensor to any sensor port"

cl.mode = 'COL-REFLECT'
cl2.mode = 'COL-REFLECT'
ir.mode = 'IR-PROX'

# ==============================================
client = mqtt.Client()
client.connect(MQTT_SERVER_ADDRESS, MQTT_SERVER_PORT, 60)

client.on_connect = on_connect
client.on_message = on_message

client.loop_forever()
