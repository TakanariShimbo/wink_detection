import socket


class Udp_client:

    def __init__(self, host_ip="127.0.0.1", port=5000):
        self.hostIP = host_ip
        self.port = port
        self.client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send_msg_face_detected(self, data):
        # check data
        if type(data) is not bool:
            return

        # create message
        msg = bytearray()
        msg.append(0x00)  # ID
        msg.append(0x01)  # Size
        msg.append(data)  # data

        self.client.sendto(msg, (self.hostIP, self.port))

    def send_msg_left_wink(self, data):
        # check data
        if type(data) is not bool:
            return

        # create message
        msg = bytearray()
        msg.append(0x01)  # ID
        msg.append(0x01)  # Size
        msg.append(data)  # data

        self.client.sendto(msg, (self.hostIP, self.port))

    def send_msg_right_wink(self, data):
        # check data
        if type(data) is not bool:
            return

        # create message
        msg = bytearray()
        msg.append(0x02)  # ID
        msg.append(0x01)  # Size
        msg.append(data)  # data

        self.client.sendto(msg, (self.hostIP, self.port))

