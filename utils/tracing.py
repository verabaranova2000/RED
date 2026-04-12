# utils/tracing.py

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


def render_bind(title, items):
    print(f"[bind] {title} - подключение")

    for i, item in enumerate(items):
        prefix = "└─" if i == len(items) - 1 else "├─"
        print(f"  {prefix} {item}")


class TraceSession:
    def __init__(self):
        self.active = False
        self.title = None
        self.bind_items = []

    def bind_context(self, title: str):
        return self._BindContext(self, title)

    class _BindContext:
        def __init__(self, trace, title):
            self.trace = trace
            self.title = title

        def __enter__(self):
            self.trace.active = True
            self.trace.title = self.title
            self.trace.bind_items = []
            return self.trace

        def __exit__(self, exc_type, exc, tb):
            self.trace.active = False
            render_bind(self.title, self.trace.bind_items)

TRACE = TraceSession()  # один экземпляр на всё приложение