class TraceSession:
    def __init__(self):
        self.active = False
        self.events = []
        self.level = 0

    def start(self, root):
        if self.active:
            return
        self.active = True
        self.events = []
        self.level = 0
        self.log(f"START: {root}")

    def end(self):
        if not self.active:
            return
        self.log("END")
        self.print()
        self.active = False

    def log(self, msg):
        if not self.active:
            return
        indent = "  " * self.level
        self.events.append(f"{indent}{msg}")

    def enter(self, name):
        self.log(f"→ {name}")
        self.level += 1

    def exit(self):
        self.level = max(0, self.level - 1)

    def print(self):
        print("\n🧩 TRACE SESSION:")
        for e in self.events:
            print(e)
        print()



TRACE = TraceSession()  # один экземпляр на всё приложение