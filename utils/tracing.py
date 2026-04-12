class TraceCollector:
    def __init__(self):
        self.current = None

    def start_bind(self, source, target):
        self.current = {
            "type": "bind",
            "source": source,
            "target": target,
            "paths": []
        }

    def add_path(self, path):
        if self.current and self.current["type"] == "bind":
            self.current["paths"].append(path or "root")

    def end(self):
        if not self.current:
            return

        self._print_bind(self.current)
        self.current = None

    def _print_bind(self, data):
        print(f"\n🔗 {data['source']} → {data['target']}")
        for p in data["paths"]:
            print(f"  • {p}")



TRACE = TraceCollector()  # один экземпляр на всё приложение