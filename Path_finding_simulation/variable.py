import time
class Data():
    received_data = ""

MQTT_SERVER_ADDRESS = "192.168.1.122"
MQTT_SERVER_PORT = 1883

DRIVER_STATUS_FORWARD = "turn_forward"
DRIVER_STATUS_BACK = "turn_back"
DRIVER_STATUS_LEFT = "turn_left"
DRIVER_STATUS_RIGHT = "turn_right"
DRIVER_STATUS_STOP = "stop"
DRIVER_STATUS_EXIT = "exit"
DRIVER_STATUS_TURN_OPPOSITE = "turn_opposite"

TOPIC_SUBSCRIBE_ALL = '#'
TOPIC_SUBSCRIBE = "turn_over"
TOPIC_SUBSCRIBE_TO_ANDROID = "MESSAGE_TO_ANDROID_CLIENT"
TOPIC_SUBSCRIBE_FROM_ANDROID = "MESSAGE_FROM_ANDROID_CLIENT"

TOPIC_SUBSCRIBE_TO_EV3 = "MESSAGE_TO_EV3_CLIENT"
TOPIC_SUBSCRIBE_FROM_EV3 = "MESSAGE_FROM_EV3_CLIENT"

INTERVAL_UPDATE_STATE = 0.05

BUFFER_SIZE = 100000

DEBUG = True


#stop - using when stop one epochs
ACTION_STOP = "STOP"
#forward
ACTION_FORWARD = "0"
#left
ACTION_LEFT = "1"
#right
ACTION_RIGHT = "2"

def waitForSecond(time):
    currentTime = getCurrentTime()
    while (getCurrentTime() - currentTime) < time:
        pass

def getCurrentTime():
    millis = int(round(time.time() * 1000))
    return millis