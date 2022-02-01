import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import mpl_toolkits.axes_grid1
import matplotlib.widgets
from dataclasses import dataclass
from typing import Callable, List, Tuple
from matplotlib.patches import Circle
from matplotlib.lines import Line2D


@dataclass(frozen=True, eq=True)
class Position:
    x: int
    y: int

    def numpy(self):
        return np.asarray([self.y, self.x])

    @staticmethod
    def from_numpy(array: np.array) -> "Position":
        assert array.shape == (2,)
        return Position(array[1], array[0])

    def __iadd__(self, other: "Position") -> "Position":
        self.x += other.x
        self.y += other.y
        return self

    def __add__(self, other: "Position") -> "Position":
        return Position(self.x + other.x, self.y + other.y)

    def is_in_rectangle(self, height: int, width: int) -> bool:
        return width > self.x >= 0 and height > self.y >= 0

    def crop(self, height: int, width: int) -> "Position":
        if width <= self.x:
            x = width - 1
        elif self.x < 0:
            x = 0
        else:
            x = self.x

        if height <= self.y:
            y = height - 1
        elif self.y < 0:
            y = 0
        else:
            y = self.y
        return Position(x, y)

    def as_tuple(self):
        return self.x, self.y


@dataclass
class Frame:
    data: np.array
    marker: Position
    policy: np.array
    trace: List[Position]


def draw_arrow(x, y, delta, target, scaling=0.25):
    dx, dy = delta

    return target.arrow(x-dx*scaling, y-dy*scaling, dx*scaling, dy*scaling, fc="k",
                        ec="k", head_width=0.15, head_length=0.2)


text_args = {
    'horizontalalignment': 'center',
    'verticalalignment': 'center',
    'fontsize': 15
}


def draw_line(target, trace: List[Position], success: bool):
    verts = [p.as_tuple() for p in trace]

    color = 'green' if success else 'red'
    line = Line2D(*zip(*verts), color=color, lw=2)
    target.add_line(line)


def draw_frame(title: str, frame: Frame, start: Position, goal: Position, arrow_map: Callable[[int], Tuple[int, int]], success: bool):

    fig, ax = plt.subplots()

    img = ax.imshow(frame.data)
    ax.text(start.x, start.y, 'S', **text_args)
    ax.text(goal.x, goal.y, 'G', **text_args)
    ax.set_title(title)

    fig.colorbar(img)
    arrays = [draw_arrow(x, y, arrow_map(p), target=ax)
              for y, row in enumerate(frame.policy) for x, p in enumerate(row) if (x, y) != goal.as_tuple()]
    draw_line(target=ax, trace=frame.trace, success=success)
    plt.show()


class Player(FuncAnimation):
    def __init__(self, frames: List[Frame], start: Position, goal: Position, arrow_map: Callable[[int], Tuple[int, int]], init_func=None, fargs=None,
                 save_count=None, pos=(0.125, 0.92), **kwargs):
        self.i = 0
        self.min = 0
        self.max = len(frames) - 1
        self.frames = frames
        self.runs = True
        self.forwards = True
        self.arrow_map = arrow_map
        self.fig, self.ax = plt.subplots()

        self.img = self.ax.imshow(frames[0].data)
        self.ax.text(start.x, start.y, 'S', **text_args)
        self.ax.text(goal.x, goal.y, 'G', **text_args)

        # self.arrays = [draw_arrow(x, y, arrow_map(p), target=self.ax)
        #                for y, row in enumerate(frames[0].policy) for x, p in enumerate(row)]

        self.fig.colorbar(self.img)

        self.fig.canvas.mpl_connect('key_press_event', self._on_press)
        self.marker = Circle((3, 4), radius=0.2, color='red')
        self.ax.add_patch(self.marker)
        self.setup(pos)
        FuncAnimation.__init__(self, self.fig, self.update, frames=self.play(),
                               init_func=init_func, fargs=fargs,
                               save_count=save_count, **kwargs)
        self.stop()

    def _on_press(self, event):
        if event.key == 'left':
            self.onebackward()
        elif event.key == 'right':
            self.oneforward()
        else:
            print(f'unknown key {event.key}')

    def play(self):
        while self.runs:
            self.i = self.i+self.forwards-(not self.forwards)
            if self.i > self.min and self.i < self.max:
                yield self.i
            else:
                self.stop()
                yield self.i

    def stop(self, event=None):
        self.runs = False
        self.event_source.stop()

    def oneforward(self, event=None):
        self.forwards = True
        self.onestep()

    def onebackward(self, event=None):
        self.forwards = False
        self.onestep()

    def onestep(self):
        if self.i > self.min and self.i < self.max:
            self.i = self.i+self.forwards-(not self.forwards)
        elif self.i == self.min and self.forwards:
            self.i += 1
        elif self.i == self.max and not self.forwards:
            self.i -= 1
        self.set_pos(self.i)
        self.slider.set_val(self.i)
        self.fig.canvas.draw_idle()

    def setup(self, pos):
        playerax = self.fig.add_axes([pos[0], pos[1], 0.64, 0.04])
        divider = mpl_toolkits.axes_grid1.make_axes_locatable(playerax)
        ofax = divider.append_axes("right", size="100%", pad=0.05)
        sliderax = divider.append_axes("right", size="500%", pad=0.07)
        self.button_oneback = matplotlib.widgets.Button(
            playerax, label='$\u29CF$')
        self.button_oneforward = matplotlib.widgets.Button(
            ofax, label='$\u29D0$')
        self.button_oneback.on_clicked(self.onebackward)
        self.button_oneforward.on_clicked(self.oneforward)
        self.slider = matplotlib.widgets.Slider(sliderax, '',
                                                self.min, self.max, valinit=self.i, valstep=1)
        self.slider.on_changed(self.set_pos)

    def set_pos(self, i):
        i = int(i)
        self.i = i
        frame = self.frames[i]
        self.img.set_data(frame.data)
        x = frame.marker.x
        y = frame.marker.y
        self.marker.set(center=(x, y))

        if frame.policy is not None:
            for y, row in enumerate(frame.policy):
                for x, p in enumerate(row):
                    i = y*row.shape[0]+x
                    dx, dy = self.arrow_map(p)
                    scale = 0.25
                    dx = dx * scale+x
                    dy = dy * scale+x
                    posB = dx, dx
                # self.arrays[i].set_patch(posB=posB)
                # self.arrays[i].set(dx=scale*dx, dy=scale*dy)
        # self.arrays = [draw_arrow(x, y, arrow_map(p), target=self.ax)
        #                for y, row in enumerate(frames[0].policy) for x, p in enumerate(row)]

    def update(self, i):
        self.slider.set_val(i)


if __name__ == '__main__':
    # test
    rng = np.random.RandomState()

    frames = 10
    width = 10
    height = 7

    data = [Frame(data=rng.uniform(0., 1., size=(height, width)),
                  policy=rng.randint(0, 3, size=(height, width)),
                  trace=[],
                  marker=Position(x=i % width,
                                  y=i % height)) for i in range(frames)]

    data[0].data[0, 0] = 1.5
    data[1].data[0, -1] = 1.5
    data[2].data[-1, -1] = 1.5
    data[3].data[-1, 0] = 1.5

    def arrow_map(index: int):
        arrow_dict = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        return arrow_dict[index]

    start = Position(0, 3)
    goal = Position(7, 3)

    ani = Player(frames=data, start=start, goal=goal, arrow_map=arrow_map)

    plt.show()
