from metaflow import resources, step  # type: ignore
from metaflow.flowspec import FlowSpec


class LinearFlow(FlowSpec):
    @resources(memory=1000, cpu=1)
    @step
    def start(self):
        self.my_var = "hello world"
        self.next(self.a)

    @step
    def a(self):
        print("the data artifact is: %s" % self.my_var)
        self.next(self.end)

    @step
    def end(self):
        print("the data artifact is still: %s" % self.my_var)


if __name__ == "__main__":
    LinearFlow()
