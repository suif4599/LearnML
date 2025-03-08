import email.header
import email.message
import email.mime
import email.mime.multipart
import email.mime.text
import email.utils
import json
import smtplib
import imaplib
from email.header import Header
import email
import datetime
from typing import Callable
import traceback
import time
import re

class MailServer:
    def __init__(self, username: str, password: str, 
                 imap_ssl_port: int = 993, smtp_ssl_port: int = 465):
        self.username = username
        self.password = password
        self.smtp_ssl_port = smtp_ssl_port
        try:
            imap_server = imaplib.IMAP4_SSL(username.split("@")[1], imap_ssl_port)
            imap_server.login(username, password)
            imap_server.select("INBOX")
            self.imap = imap_server
        except Exception as e:
            self.send(username, "Error", traceback.format_exc())
            raise e
        self.username = username
        self.callback_map = {}
        self.callback_help = {}
        self.__replied = set()
    
    @property
    def smtp(self):
        smtp_server = smtplib.SMTP_SSL(self.username.split("@")[1], self.smtp_ssl_port)
        smtp_server.login(self.username, self.password)
        return smtp_server

    def send(self, subject: str = None, content: str = None, to: str = None):
        if to is None:
            to = self.username
        if subject is None:
            subject = "Python MailServer"
        if content is None:
            content = f"Sent at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        message = email.mime.text.MIMEText(content, "plain", "utf-8")
        message["From"] = Header(self.username, "utf-8")
        message["To"] = Header(to, "utf-8")
        message["Subject"] = Header(subject, "utf-8")
        self.smtp.sendmail(self.username, to, message.as_string())
    
    def register(self, command: str, callback: Callable, help: str = ""):
        "Register a command with a callback function, which should accept a string parameter and return None or str for reply.\n"
        "example: server.register('print', print)"
        self.callback_map[command] = callback
        self.callback_help[command] = help if help else "No help available."
    
    def reply(self, msg: email.message.Message, content: str):
        msg_id = msg["Message-ID"]
        if msg_id in self.__replied:
            return
        self.__replied.add(msg_id)
        original_from = email.utils.parseaddr(msg["From"])[1]
        subject = []
        for sub, enc in email.header.decode_header(msg["Subject"]):
            if enc is not None:
                sub = sub.decode(enc)
            subject.append(sub)
        subject = "".join(subject).strip()
        message = email.mime.text.MIMEText(content, "plain", "utf-8", policy=email.policy.default)
        message["From"] = Header(self.username, "utf-8")
        message["To"] = Header(original_from, "utf-8")
        message["Subject"] = f"Python reply: {subject}"
        message["In-Reply-To"] = msg_id
        message["References"] = msg_id
        self.smtp.sendmail(self.username, original_from, message.as_string())

    def mainloop(self, delay: float = 10., create_thread: bool = False):
        if create_thread:
            import threading
            threading.Thread(target=self.__mainloop, args=(delay,), daemon=True).start()
        else:
            self.__mainloop(delay)
    
    def __mainloop(self, delay):
        skip = set()
        _, data = self.imap.search(None, "ALL")
        for num in data[0].split():
            _, data = self.imap.fetch(num, "(RFC822)")
            if not data[0]:
                continue
            msg = email.message_from_bytes(data[0][1])
            if msg["In-Reply-To"]:
                self.__replied.add(msg["In-Reply-To"])
            if msg["References"]:
                self.__replied.add(msg["References"])
        while 1:
            _, data = self.imap.search(None, "ALL")
            for num in data[0].split():
                _, data = self.imap.fetch(num, "(RFC822)")
                if not data[0]:
                    continue
                msg = email.message_from_bytes(data[0][1])
                if msg["Message-ID"] in skip:
                    continue
                subject = []
                for sub, enc in email.header.decode_header(msg["Subject"]):
                    if enc is not None:
                        sub = sub.decode(enc)
                    else:
                        if isinstance(sub, bytes):
                            sub = sub.decode()
                    subject.append(sub)
                subject = "".join(subject).strip().lower()
                if subject != "command":
                    skip.add(msg["Message-ID"])
                    continue
                if msg["Message-ID"] in self.__replied:
                    continue
                content = msg.get_payload(decode=True).decode("utf-8").strip()
                command = re.search(r"(\w+)\((.*)\)", content)
                if command is None:
                    self.reply(msg, "Invalid command format.")
                    continue
                command = command.group(1).strip().lower()
                if command not in self.callback_map:
                    content = f"Unknown command: {command}\nRegistered commands: "
                    content += '\n'.join(f"{command}: {help}" for command, help in zip(self.callback_map, self.callback_help))
                    self.reply(msg, content)
                    continue
                try:
                    param = re.search(r"(.+)\((.*)\)", content).group(2)
                    t = time.time()
                    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    res = self.callback_map[command](param)
                    dt = time.time() - t
                    content = f"Command <{command}({param})> executed at {now}, time used: {dt:.2f}s\n"
                    if res is not None:
                        content += f"Result:\n{res}"
                    self.reply(msg, content)
                except Exception as e:
                    self.reply(msg, traceback.format_exc())
            time.sleep(delay)
            

# server = MailServer("xxx@xxx", "password")
# server.register("print", print)
# server.mainloop(delay=1, create_thread=True)
# input("Press Enter to exit.\n")
