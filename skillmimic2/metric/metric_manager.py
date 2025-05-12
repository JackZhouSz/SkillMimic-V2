# skillmimic/metrics/metric_manager.py
class MetricManager:
    def __init__(self, metrics):
        self.metrics = metrics

    def reset(self, env_ids):
        for metric in self.metrics:
            metric.reset(env_ids)

    def update(self, state):
        for metric in self.metrics:
            metric.update(state)

    def compute(self):
        results = {}
        for metric in self.metrics:
            results[type(metric).__name__] = metric.compute()
        return results
