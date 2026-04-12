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



class TraceSession:
    def __init__(self):
        self.active = False
        self.title = None
        self.root = None
        self.bind_items = []

    def start_bind(self, title: str, root: str):
        self.active = True
        self.title = title
        self.root = root
        self.bind_items = []

    def add_bind_item(self, path: str):
        if not self.active:
            return
        self.bind_items.append(path or "root")

    def end_bind(self):
        if not self.active:
            return

        self._print_bind()

        self.active = False
        self.title = None
        self.root = None
        self.bind_items = []

    def _print_bind(self):
        print(f"[bind] {self.title} - подключение")

        # root первым
        items = ["root"] + [i for i in self.bind_items if i != ""]

        for i, item in enumerate(items):
            if i == 0:
                print(f"  ├─ {item}")
            elif i == len(items) - 1:
                print(f"  └─ {item}")
            else:
                print(f"  ├─ {item}")

TRACE = TraceSession()  # один экземпляр на всё приложение