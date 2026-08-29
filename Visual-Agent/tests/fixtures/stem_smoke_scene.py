from manim import Circle, Create, Scene


class StemSmokeScene(Scene):
    def construct(self):
        circle = Circle(color="#6a9bcc")
        self.play(Create(circle), run_time=0.2)
        self.wait(0.1)
