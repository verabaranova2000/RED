# utils/tracing.py
from contextlib import contextmanager

class TraceSession_v0:
    """" Работал с bind_v1 """
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




class TraceSession_bad:
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



class TraceSession_v1:
    """
    Один трассировщик с “выключателем”

    В функциональных методах вообще не рисовать и не печатать, 
    а только при необходимости эмитить события в пустой по умолчанию приёмник. 
    А включать это всё — только через with ....

        - без with — обычная работа, без шума;
        - с with — сбор событий;
        - рендер — только в конце сессии;
        - bind, __setattr__, update_* не знают ничего про красивый вывод.
    """
    def __init__(self):
        self.enabled = False
        self.blocks = []
        self._current = None

    @contextmanager
    def session(self, title: str):
        prev = self.enabled
        self.enabled = True
        block = {"title": title, "events": []}
        self.blocks.append(block)
        self._current = block
        try:
            yield self
        finally:
            self._render(block)
            self.blocks.pop()
            self._current = self.blocks[-1] if self.blocks else None
            self.enabled = prev

    def emit(self, kind: str, text: str):
        if not self.enabled or self._current is None:
            return
        self._current["events"].append((kind, text))

    def _render(self, block):
        print(f"[{block['title']}]")
        for kind, text in block["events"]:
            print(f"  {kind}: {text}")
        #print()



# екущая версия уже почти подходит, но её лучше сделать 
# контекстной и стековой, чтобы она умела вложенные блоки
class TraceSession:
    def __init__(self):
        self.stack = []

    @contextmanager
    def session(self, title: str):
        block = {"title": title, "events": []}
        self.stack.append(block)
        try:
            yield self
        finally:
            finished = self.stack.pop()
            self._render(finished)

    def emit(self, kind: str, text: str):
        if not self.stack:
            return
        self.stack[-1]["events"].append((kind, text))

    def _render(self, block):
        print(f"[{block['title']}]")
        for kind, text in block["events"]:
            print(f"  {kind}: {text}")
        print()

TRACE = TraceSession()  # один экземпляр на всё приложение