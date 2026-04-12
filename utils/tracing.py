# utils/tracing.py
from contextlib import contextmanager

class TraceSession_v0:
    def __init__(self):
        self.active = False
        self.events = []
        self.bind_items = []
        self.title = None
        self.level = 0

    def start_bind(self, title: str):
        self.active = True
        self.title = title
        self.bind_items = []

    def add_bind_item(self, path: str):
        if not self.active:
            return
        self.bind_items.append(path or "root")

    def end_bind(self):
        if not self.active:
            return

        print(f"🔗 {self.title}: подключение (bind)")
        for item in self.bind_items:
            print(f"  • {item}")
        #print()

        self.active = False
        self.bind_items = []
        self.title = None




class TraceSession:
    def __init__(self):
        self.stack = []
        self.active = False

    @contextmanager
    def bind_context(self, title: str):
        block = {"title": title, "items": []}
        self.stack.append(block)
        self.active = True
        try:
            yield self
        finally:
            self.stack.pop()
            self._render_block(block)
            self.active = bool(self.stack)

    def add_bind_item(self, path: str):
        if not self.active:
            return
        self.stack[-1]["items"].append(path)

    def _render_block(self, block):
        print(f"[bind] {block['title']} - подключение")

        items = block["items"]
        for i, item in enumerate(items):
            prefix = "└─" if i == len(items) - 1 else "├─"
            print(f"  {prefix} {item}")


TRACE = TraceSession()  # один экземпляр на всё приложение