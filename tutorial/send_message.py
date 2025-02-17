from wxauto import WeChat


def send_message(who='文件传输助手', message='代码运行完毕'):
    wx = WeChat()
    wx.GetSessionList()
    wx.ChatWith(who)
    wx.SendMsg(message)


if __name__ == '__main__':
    send_message()