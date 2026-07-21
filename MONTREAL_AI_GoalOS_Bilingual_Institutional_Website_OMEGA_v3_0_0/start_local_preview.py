#!/usr/bin/env python3
from __future__ import annotations
import http.server, pathlib, socketserver, threading, time, webbrowser

ROOT=pathlib.Path(__file__).resolve().parent
class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self,*args,**kwargs): super().__init__(*args,directory=str(ROOT),**kwargs)
    def log_message(self,fmt,*args): print(fmt%args)
class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address=True

with Server(('127.0.0.1',0),Handler) as server:
    port=server.server_address[1]
    url=f'http://127.0.0.1:{port}/index.html'
    print(f'MONTREAL.AI × GoalOS local preview: {url}')
    print('Press Ctrl+C to stop.')
    threading.Timer(0.8,lambda:webbrowser.open(url)).start()
    try: server.serve_forever()
    except KeyboardInterrupt: pass
