import smtplib
from email.mime.text import MIMEText
from email.header import Header

class MailSender():
    def __init__(self,receivers='327067530@qq.com' ):
        self.sender = 'woshiliyongzhi@yeah.net'
        self.receivers = [receivers] # 接收邮件，可设置为你的QQ邮箱或者其他邮箱
        self.mail_host='smtp.yeah.net'
        self.mail_user='15572382168@yeah.net'
        self.mail_pass='LYZliyongzhi!@#$'

    def send(self,mail_info,mail_head="Program ERROR"):
        # 三个参数：第一个为文本内容，第二个 plain 设置文本格式，第三个 utf-8 设置编码
        message = MIMEText(mail_info, 'plain', 'utf-8')
        message['From'] = Header("python program", 'utf-8')  # 发送者
        message['To'] = Header("测试", 'utf-8')  # 接收者

        subject = mail_head
        message['Subject'] = Header(subject, 'utf-8')

        try:
            smtpObj = smtplib.SMTP()
            smtpObj.connect(self.mail_host, 25)  # 25 为 SMTP 端口号
            smtpObj.login(self.mail_user, self.mail_pass)
            smtpObj.sendmail(self.sender, self.receivers, message.as_string())
            print("邮件发送成功")
        except smtplib.SMTPException:
            print("Error: 无法发送邮件")

sender=MailSender()
sender.send("sdfsdf")