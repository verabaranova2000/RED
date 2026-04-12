# utils/tracing.py

class TraceSession:
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

        print(f"\n🔗 {self.title}")
        for item in self.bind_items:
            print(f"  • {item}")
        print()

        self.active = False
        self.bind_items = []
        self.title = None


TRACE = TraceSession()  # один экземпляр на всё приложение