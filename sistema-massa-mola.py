from dataclasses import dataclass
from vpython import *
import numpy as np

@dataclass
class Parametros:
    k: float = 1.0
    masses: list = None
    amplitudes: list = None
    phases: list = None
    velo: float = 2.0
    slider_range_k: tuple = (0.1, 5.0)
    slider_range_mass: tuple = (1.0, 15.0)
    slider_range_amp: tuple = (0.2, 0.3)

    def __post_init__(self):
        if self.masses is None:
            self.masses = [1.0, 1.0]
        if self.amplitudes is None:
            self.amplitudes = [0.2] * len(self.masses)
        if self.phases is None:
            self.phases = [0.0] * len(self.masses)

class SistemaMassaMola:
    MAX_MASSES = 4

    def __init__(self, params: Parametros):
        self.params = params
        self.freqs = []
        self.modos = []
        self.spheres = []
        self.springs = []
        self.ui_sliders = []
        self.ui_texts = []
        self.equilibrium = []
        self._setup_scene()
        self._build_system()
        self._create_ui()
        self._update_modes()

    def calcular_frequencias_modos(self):
        n = len(self.params.masses)
        M = np.diag(self.params.masses)
        k = self.params.k
        K = np.zeros((n, n))
        for i in range(n):
            if i > 0:
                K[i, i] += k; K[i, i-1] -= k
            if i < n-1:
                K[i, i] += k; K[i, i+1] -= k
        K[0, 0] += k; K[-1, -1] += k
        w2, v = np.linalg.eig(np.linalg.solve(M, K))
        idx = np.argsort(np.sqrt(w2))
        self.freqs = np.sqrt(w2[idx])
        self.modos = v[:, idx]

    def _setup_scene(self):
        scene.title = "Nome: Lucas Florindo Souza  |  RA: 11202131388 |  Sistema Massa-Mola Dinâmico"
        scene.width = 900; scene.height = 450; scene.background = color.white
        self.left_wall = box(pos=vector(-2, 0, 0), size=vector(0.1, 0.5, 0.5), color=color.gray(0.6))
        self.right_wall = box(pos=vector(2, 0, 0), size=vector(0.1, 0.5, 0.5), color=color.gray(0.6))

    def _build_system(self):
        for obj in self.spheres + self.springs:
            obj.visible = False
        self.spheres.clear(); self.springs.clear()

        n = len(self.params.masses)
        left_x = self.left_wall.pos.x + 0.05
        right_x = self.right_wall.pos.x - 0.05
        total_length = right_x - left_x
        segment = total_length / (n + 1)
        self.equilibrium = [left_x + (i+1)*segment for i in range(n)]

        for i, mass in enumerate(self.params.masses):
            x = self.equilibrium[i]
            color_sphere = vector(i/n, 0, 1-i/n)
            s = sphere(pos=vector(x, 0, 0), radius=0.1 * mass**(1/3), color=color_sphere)
            self.spheres.append(s)
            start = self.left_wall.pos + vector(0.05, 0, 0) if i == 0 else self.spheres[i-1].pos
            spring = helix(pos=start, axis=s.pos - start, radius=0.05)
            self.springs.append(spring)
        last = self.spheres[-1]
        end_axis = self.right_wall.pos - vector(0.05, 0, 0) - last.pos
        self.springs.append(helix(pos=last.pos, axis=end_axis, radius=0.05))

    def _clear_ui(self):
        scene.caption = ''
        for sl, txt in zip(self.ui_sliders, self.ui_texts):
            sl.visible = False; txt.visible = False
        self.ui_sliders.clear(); self.ui_texts.clear()

    def _slider_scalar(self, label, color_val, attr, value, rng, desc):
        scene.append_to_caption(f'<b>{label}:</b> ')
        txt = wtext(text=f'{value:.2f}')
        step = (rng[1] - rng[0]) / 100
        def on_slide(s):
            setattr(self.params, attr, s.value)
            txt.text = f'{s.value:.2f}'
            self._build_system(); self._update_modes()
        sl = slider(min=rng[0], max=rng[1], value=value, step=step,
                    length=200, color=color_val, bind=on_slide)
        scene.append_to_caption(' ', sl, ' ', txt, '<br>')
        scene.append_to_caption(f'<i>{desc}</i><br><br>')
        self.ui_sliders.append(sl); self.ui_texts.append(txt)

    def _slider_factory(self, label, color_val, attr, index, desc, rng):
        col = self.spheres[index].color
        r, g, b = int(col.x*255), int(col.y*255), int(col.z*255)
        dot = f"<span style='display:inline-block;width:10px;height:10px;background:rgb({r},{g},{b});border-radius:50%;margin-right:4px;'></span>"
        scene.append_to_caption(dot + f'<b>{label} {index+1}:</b> ')
        txt = wtext(text=f'{self.params.__dict__[attr][index]:.2f}')
        step = (rng[1] - rng[0]) / 100
        def on_slide(s):
            val = s.value
            self.params.__dict__[attr][index] = val
            txt.text = f'{val:.2f}'
            if attr == 'masses': self._build_system()
            self._update_modes()
        sl = slider(min=rng[0], max=rng[1], value=self.params.__dict__[attr][index],
                    step=step, length=200, color=color_val, bind=on_slide)
        scene.append_to_caption(' ', sl, ' ', txt, '<br>')
        scene.append_to_caption(f'<i>{desc}</i><br><br>')
        self.ui_sliders.append(sl); self.ui_texts.append(txt)

    def _create_ui(self):
        self._clear_ui(); scene.append_to_caption('<hr>')
        # slider de k
        self._slider_scalar(
            '🟢 Constante da Mola k', color.green, 'k', self.params.k,
            self.params.slider_range_k, 'Ajusta a constante elástica k das molas.'
        )
        # painel por esfera
        for i in range(len(self.params.masses)):
            scene.append_to_caption(
                "<div style='border:1px solid #ccc;padding:8px;border-radius:6px;margin:6px 0;background:#f7f7f7;'>"
            )
            col = self.spheres[i].color; r, g, b = int(col.x*255), int(col.y*255), int(col.z*255)
            dot = f"<span style='width:12px;height:12px;background:rgb({r},{g},{b});border-radius:50%;display:inline-block;margin-right:6px;'></span>"
            scene.append_to_caption(dot + f'<b>Esfera {i+1}</b><br>')
            self._slider_factory('Massa', self.spheres[i].color, 'masses', i,
                                  f'Ajusta massa {i+1} (kg) e raio.', self.params.slider_range_mass)
            self._slider_factory('Amplitude', self.spheres[i].color, 'amplitudes', i,
                                  f'Ajusta amplitude do modo {i+1}.', self.params.slider_range_amp)
            scene.append_to_caption('</div>')
        if len(self.params.masses) < self.MAX_MASSES:
            button(text='➕ Adicionar Massa', bind=lambda _: self._add_mass())
        button(text='🔄 Resetar Sistema', bind=lambda _: self._reset_system())
        scene.append_to_caption('<br><hr><br><b>📊 Informações Físicas</b><br>')
        self.info_texto = wtext(text='')

    def _add_mass(self):
        if len(self.params.masses) >= self.MAX_MASSES: return
        self.params.masses.append(1.0); self.params.amplitudes.append(0.2); self.params.phases.append(0.0)
        self._build_system(); self._create_ui(); self._update_modes()

    def _reset_system(self):
        self.params.masses = [1.0, 1.0]; self.params.amplitudes = [0.2, 0.2]; self.params.phases = [0.0, 0.0]
        self._build_system(); self._create_ui(); self._update_modes()

    def _update_modes(self):
        self.calcular_frequencias_modos(); freqs, modos = self.freqs, self.modos
        txt = '<b>Frequências (rad/s):</b><br>' + ''.join(f'ω{i+1}={freq:.3f}<br>'
              for i, freq in enumerate(freqs))
        txt += '<br><b>Modos Normais:</b><br>' + ''.join(
            f'Modo {j+1}:[{" ,".join(f"{m:.2f}" for m in modos[:,j])}]<br>'
            for j in range(len(freqs)))
        self.info_texto.text = txt

    def animate(self):
        t = 0
        while True:
            rate(100)
            dt = 0.01 * self.params.velo
            disp = [self.params.amplitudes[j] * np.cos(self.freqs[j] * t + self.params.phases[j])
                    for j in range(len(self.spheres))]
            for i, s in enumerate(self.spheres):
                s.pos.x = self.equilibrium[i] + disp[i]
                start = self.left_wall.pos + vector(0.05, 0, 0) if i == 0 else self.spheres[i-1].pos
                sp = self.springs[i]; sp.pos = start; sp.axis = s.pos - start
            last = self.spheres[-1]; sp_last = self.springs[-1]
            sp_last.pos = last.pos; sp_last.axis = self.right_wall.pos - vector(0.05, 0, 0) - last.pos
            t += dt

if __name__ == '__main__':
    params = Parametros()
    sistema = SistemaMassaMola(params)
    sistema.animate()
