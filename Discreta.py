
from __future__ import annotations

import tkinter as tk
from dataclasses import dataclass
from tkinter import messagebox, ttk
from typing import Callable, Optional

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


PLOT_LIMIT = 12
ANIMATION_FRAMES = 60
ANIMATION_INTERVAL = 25

APP_BG = "#eef3f7"
SURFACE = "#ffffff"
SURFACE_ALT = "#f8fafc"
TEXT = "#1f2933"
MUTED = "#64707a"
BORDER = "#cfd8e3"
HEADER_BG = "#20262d"
GRID = "#9aa7b3"

ORIGINAL_COLOR = "#27323a"
ROTATION_COLOR = "#128c7e"
SCALE_COLOR = "#d97706"
REFLECTION_COLOR = "#4f46e5"


@dataclass(frozen=True)
class Transformacion:
    clave: str
    titulo: str
    parametro: str
    valor_inicial: str
    color: str
    funcion: Callable[[np.ndarray, float], tuple[np.ndarray, np.ndarray]]


FIGURAS_PREDEFINIDAS: dict[str, list[list[float]]] = {
    "Nave espacial": [[0, 5], [-2, 1], [-1, 2], [0, 0], [1, 2], [2, 1]],
    "Personaje": [
        [-1.5, 0],
        [-2.5, -4],
        [-1, -4],
        [0, -1.5],
        [1, -4],
        [2.5, -4],
        [1.5, 0],
        [1, 2.5],
        [0, 3.5],
        [-1, 2.5],
    ],
    "Enemigo": [[-3, 0], [-3, 2], [-2, 3.5], [0, 2.5], [2, 3.5], [3, 2], [3, 0], [0, -1.5]],
    "Escudo": [[-2.5, -3.5], [2.5, -3.5], [3.5, 0], [0, 4.5], [-3.5, 0]],
    "Estrella": [[0, 4.5], [-1, 1.5], [-4.5, 1.5], [-2, -0.5], [-3, -4], [0, -2], [3, -4], [2, -0.5], [4.5, 1.5], [1, 1.5]],
    "Flecha": [[-4, -1], [1, -1], [1, -3], [5, 0], [1, 3], [1, 1], [-4, 1]],
    "Triangulo": [[0, 0], [5, 0], [2.5, 4]],
    "Cuadrado": [[-3, -3], [3, -3], [3, 3], [-3, 3]],
    "Casa": [[-4, -3], [4, -3], [4, 2], [0, 5.5], [-4, 2]],
    "Trapecio": [[-4, -3], [4, -3], [2.5, 3], [-2.5, 3]],
}


def rotacion(puntos: np.ndarray, angulo: float) -> tuple[np.ndarray, np.ndarray]:
    rad = np.radians(angulo)
    matriz = np.array(
        [
            [np.cos(rad), -np.sin(rad)],
            [np.sin(rad), np.cos(rad)],
        ],
        dtype=float,
    )
    return puntos @ matriz.T, matriz


def escalamiento(puntos: np.ndarray, factor: float) -> tuple[np.ndarray, np.ndarray]:
    matriz = np.array([[factor, 0.0], [0.0, factor]], dtype=float)
    return puntos @ matriz.T, matriz


def reflexion_recta_y_mx(puntos: np.ndarray, pendiente: float) -> tuple[np.ndarray, np.ndarray]:
    theta = np.arctan(pendiente)
    matriz = np.array(
        [
            [np.cos(2 * theta), np.sin(2 * theta)],
            [np.sin(2 * theta), -np.cos(2 * theta)],
        ],
        dtype=float,
    )
    return puntos @ matriz.T, matriz


TRANSFORMACIONES: tuple[Transformacion, ...] = (
    Transformacion("rotacion", "Rotacion", "Angulo (grados)", "45", ROTATION_COLOR, rotacion),
    Transformacion("escala", "Escalamiento", "Factor k", "1.5", SCALE_COLOR, escalamiento),
    Transformacion("reflexion", "Reflexion", "Pendiente m", "1", REFLECTION_COLOR, reflexion_recta_y_mx),
)


def cerrar_figura(puntos: np.ndarray) -> np.ndarray:
    if len(puntos) == 0:
        return puntos
    return np.vstack([puntos, puntos[0]])


def ease_in_out(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def configurar_ejes(ax, titulo: str, grupos: Optional[list[np.ndarray]] = None) -> None:
    ax.set_title(titulo, fontsize=10, fontweight="bold", color=TEXT, pad=8)
    ax.set_xlabel("Eje X", fontsize=8, color=MUTED)
    ax.set_ylabel("Eje Y", fontsize=8, color=MUTED)
    ax.axhline(0, color=GRID, linewidth=0.9)
    ax.axvline(0, color=GRID, linewidth=0.9)
    ax.grid(True, color=GRID, linestyle="-", linewidth=0.5, alpha=0.25)
    ax.set_facecolor(SURFACE_ALT)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(labelsize=7, colors=MUTED)

    limite = PLOT_LIMIT
    if grupos:
        validos = [g for g in grupos if g is not None and len(g) > 0]
        if validos:
            apilados = np.vstack(validos)
            max_abs = float(np.nanmax(np.abs(apilados)))
            limite = max(PLOT_LIMIT, int(np.ceil(max_abs * 1.25 + 1)))

    ax.set_xlim(-limite, limite)
    ax.set_ylim(-limite, limite)


def dibujar_figura(ax, puntos: np.ndarray, color: str, label: str, alpha: float = 1.0) -> None:
    cerrada = cerrar_figura(puntos)
    ax.fill(cerrada[:, 0], cerrada[:, 1], alpha=0.15 * alpha, color=color)
    ax.plot(cerrada[:, 0], cerrada[:, 1], color=color, linewidth=2.2, label=label, alpha=alpha)
    ax.scatter(puntos[:, 0], puntos[:, 1], color=color, s=28, zorder=5, alpha=alpha)


class MotorTransformaciones2D:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Motor de Transformaciones 2D 70% - Algebra Lineal")
        self.root.configure(bg=APP_BG)
        self.root.geometry("1300x820")
        self.root.minsize(1100, 720)

        self._puntos_base = np.array(FIGURAS_PREDEFINIDAS["Nave espacial"], dtype=float)
        self._transform_actual = TRANSFORMACIONES[0]
        self._after_id: Optional[str] = None
        self._animation_token = 0

        self._construir_ui()
        self._actualizar_grafica_inicial()

    def _construir_ui(self) -> None:
        self._construir_header()
        contenedor = tk.Frame(self.root, bg=APP_BG)
        contenedor.pack(fill="both", expand=True, padx=10, pady=8)

        panel_izq = tk.Frame(contenedor, bg=APP_BG, width=290)
        panel_izq.pack(side="left", fill="y", padx=(0, 8))
        panel_izq.pack_propagate(False)

        panel_der = tk.Frame(contenedor, bg=APP_BG)
        panel_der.pack(side="left", fill="both", expand=True)

        self._construir_controles(panel_izq)
        self._construir_graficas(panel_der)

    def _construir_header(self) -> None:
        hdr = tk.Frame(self.root, bg=HEADER_BG, height=58)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        tk.Label(
            hdr,
            text="Motor de Transformaciones 2D",
            font=("Segoe UI", 15, "bold"),
            bg=HEADER_BG,
            fg="#ffffff",
        ).pack(side="left", padx=18, pady=10)

        tk.Label(
            hdr,
            text="Algebra Lineal - Transformaciones en R2 - Aplicacion: Videojuegos 2D",
            font=("Segoe UI", 9),
            bg=HEADER_BG,
            fg="#94a3b8",
        ).pack(side="left", padx=4, pady=18)

        tk.Label(
            hdr,
            text="VERSION 70%  |  base funcional",
            font=("Segoe UI", 8, "bold"),
            bg="#cbd5e1",
            fg="#1f2933",
            padx=10,
            pady=5,
        ).pack(side="right", padx=14, pady=14)

    def _construir_controles(self, parent: tk.Frame) -> None:
        parent = self._crear_panel_scroll(parent)

        sec_fig = self._crear_seccion(parent, "Figura / Sprite del juego")
        self._figura_var = tk.StringVar(value="Nave espacial")
        cb = ttk.Combobox(
            sec_fig,
            textvariable=self._figura_var,
            values=list(FIGURAS_PREDEFINIDAS.keys()),
            state="readonly",
            font=("Segoe UI", 10),
        )
        cb.pack(fill="x", padx=10, pady=(4, 10))
        cb.bind("<<ComboboxSelected>>", self._on_figura_cambia)

        sec_puntos = self._crear_seccion(parent, "Ingreso manual de puntos")
        fila_puntos = tk.Frame(sec_puntos, bg=SURFACE)
        fila_puntos.pack(fill="x", padx=10, pady=(6, 4))
        self._x_entry = tk.Entry(fila_puntos, width=8, justify="center", font=("Segoe UI", 9))
        self._x_entry.pack(side="left", fill="x", expand=True, padx=(0, 5))
        self._x_entry.insert(0, "0")
        self._y_entry = tk.Entry(fila_puntos, width=8, justify="center", font=("Segoe UI", 9))
        self._y_entry.pack(side="left", fill="x", expand=True, padx=(0, 5))
        self._y_entry.insert(0, "0")
        self._boton_compacto(fila_puntos, "+", self._agregar_punto_manual, "#0ea5e9")

        fila_edicion = tk.Frame(sec_puntos, bg=SURFACE)
        fila_edicion.pack(fill="x", padx=10, pady=(0, 5))
        self._boton_compacto(fila_edicion, "Quitar ultimo", self._quitar_ultimo_punto, "#64748b")
        self._boton_compacto(fila_edicion, "Limpiar", self._limpiar_puntos, "#ef4444")

        self._lista_puntos = tk.Listbox(
            sec_puntos,
            height=4,
            bg="#f8fafc",
            fg=TEXT,
            relief="flat",
            highlightthickness=1,
            highlightbackground=BORDER,
            font=("Consolas", 8),
        )
        self._lista_puntos.pack(fill="x", padx=10, pady=(0, 8))
        self._actualizar_lista_puntos()

        sec_trans = self._crear_seccion(parent, "Transformacion lineal")
        self._trans_var = tk.StringVar(value=TRANSFORMACIONES[0].clave)
        for transformacion in TRANSFORMACIONES:
            tk.Radiobutton(
                sec_trans,
                text=f"  {transformacion.titulo}",
                variable=self._trans_var,
                value=transformacion.clave,
                bg=SURFACE,
                fg=TEXT,
                activebackground=SURFACE,
                selectcolor=SURFACE,
                font=("Segoe UI", 10),
                indicatoron=True,
                command=self._on_trans_cambia,
            ).pack(anchor="w", padx=10, pady=3)

        sec_param = self._crear_seccion(parent, "Parametro de la transformacion")
        self._param_label = tk.Label(sec_param, text="Angulo (grados)", bg=SURFACE, fg=MUTED, font=("Segoe UI", 9))
        self._param_label.pack(anchor="w", padx=10, pady=(6, 1))

        self._param_entry = tk.Entry(sec_param, font=("Segoe UI", 13, "bold"), justify="center", bd=1, relief="solid")
        self._param_entry.insert(0, "45")
        self._param_entry.pack(fill="x", padx=10, pady=(1, 4))
        self._param_entry.bind("<Return>", lambda _event: self._sincronizar_slider_desde_entrada())
        self._param_entry.bind("<FocusOut>", lambda _event: self._sincronizar_slider_desde_entrada())

        self._slider_var = tk.DoubleVar(value=45.0)
        self._slider = tk.Scale(
            sec_param,
            from_=-180,
            to=360,
            orient="horizontal",
            variable=self._slider_var,
            bg=SURFACE,
            highlightthickness=0,
            resolution=1,
            command=self._on_slider_mueve,
        )
        self._slider.pack(fill="x", padx=10, pady=(0, 8))

        sec_acc = self._crear_seccion(parent, "Acciones")
        botones = [
            ("Aplicar Transformacion", self._aplicar, "#0ea5e9"),
            ("Animar progresion", self._animar, "#10b981"),
            ("Comparar figuras", self._comparar, "#8b5cf6"),
            ("Restablecer todo", self._reset, "#64748b"),
        ]
        for texto, cmd, color in botones:
            self._boton(sec_acc, texto, cmd, color)

        sec_mat = self._crear_seccion(parent, "Matriz de transformacion T")
        self._texto_matriz = tk.Text(
            sec_mat,
            height=6,
            font=("Courier New", 9),
            bg="#1e2228",
            fg="#a3e635",
            bd=0,
            relief="flat",
            state="disabled",
        )
        self._texto_matriz.pack(fill="x", padx=8, pady=(4, 8))

    def _construir_graficas(self, parent: tk.Frame) -> None:
        self._fig = Figure(figsize=(11, 5.8), facecolor=SURFACE)
        self._ax_orig, self._ax_trans = self._fig.subplots(1, 2)
        self._fig.subplots_adjust(wspace=0.32, left=0.07, right=0.97)

        self._canvas = FigureCanvasTkAgg(self._fig, master=parent)
        self._canvas.get_tk_widget().pack(fill="both", expand=True)

    def _on_figura_cambia(self, _event=None) -> None:
        self._detener_animacion()
        nombre = self._figura_var.get()
        self._puntos_base = np.array(FIGURAS_PREDEFINIDAS[nombre], dtype=float)
        self._actualizar_lista_puntos()
        self._actualizar_grafica_inicial()
        self._limpiar_matriz()

    def _actualizar_lista_puntos(self) -> None:
        if not hasattr(self, "_lista_puntos"):
            return
        self._lista_puntos.delete(0, tk.END)
        for indice, (x, y) in enumerate(self._puntos_base, start=1):
            self._lista_puntos.insert(tk.END, f"{indice:02d}: ({x:6.2f}, {y:6.2f})")

    def _leer_punto_manual(self) -> Optional[tuple[float, float]]:
        try:
            return float(self._x_entry.get()), float(self._y_entry.get())
        except ValueError:
            messagebox.showerror("Punto invalido", "Ingresa valores numericos validos para X e Y.")
            return None

    def _agregar_punto_manual(self) -> None:
        punto = self._leer_punto_manual()
        if punto is None:
            return

        self._detener_animacion()
        nuevo = np.array([[punto[0], punto[1]]], dtype=float)
        self._puntos_base = nuevo if len(self._puntos_base) == 0 else np.vstack([self._puntos_base, nuevo])
        self._actualizar_lista_puntos()
        self._actualizar_grafica_inicial()
        self._limpiar_matriz()

    def _quitar_ultimo_punto(self) -> None:
        if len(self._puntos_base) == 0:
            return
        self._detener_animacion()
        self._puntos_base = self._puntos_base[:-1]
        self._actualizar_lista_puntos()
        self._actualizar_grafica_inicial()
        self._limpiar_matriz()

    def _limpiar_puntos(self) -> None:
        self._detener_animacion()
        self._puntos_base = np.empty((0, 2), dtype=float)
        self._actualizar_lista_puntos()
        self._actualizar_grafica_inicial()
        self._limpiar_matriz()

    def _on_trans_cambia(self) -> None:
        self._detener_animacion()
        clave = self._trans_var.get()
        transformacion = next(t for t in TRANSFORMACIONES if t.clave == clave)
        self._transform_actual = transformacion
        self._param_label.config(text=transformacion.parametro)
        self._param_entry.delete(0, "end")
        self._param_entry.insert(0, transformacion.valor_inicial)

        rangos = {
            "rotacion": (-180, 360, 1),
            "escala": (0.1, 5, 0.1),
            "reflexion": (-5, 5, 0.1),
        }
        minimo, maximo, resolucion = rangos[transformacion.clave]
        self._slider.config(from_=minimo, to=maximo, resolution=resolucion)
        self._slider_var.set(float(transformacion.valor_inicial))
        self._actualizar_grafica_inicial()
        self._limpiar_matriz()

    def _on_slider_mueve(self, val: str) -> None:
        self._param_entry.delete(0, "end")
        self._param_entry.insert(0, f"{float(val):.2f}")

    def _sincronizar_slider_desde_entrada(self) -> None:
        valor = self._obtener_parametro(mostrar_error=False)
        if valor is None:
            return
        minimo = float(self._slider.cget("from"))
        maximo = float(self._slider.cget("to"))
        valor_limitado = min(max(valor, minimo), maximo)
        self._slider_var.set(valor_limitado)

    def _obtener_parametro(self, mostrar_error: bool = True) -> Optional[float]:
        try:
            valor = float(self._param_entry.get())
        except ValueError:
            if mostrar_error:
                messagebox.showerror(
                    "Parametro invalido",
                    "El parametro debe ser un numero. Ejemplos: 45, 1.5, -1, 0.5.",
                )
            return None

        if self._transform_actual.clave == "escala" and np.isclose(valor, 0.0):
            if mostrar_error:
                messagebox.showerror("Factor invalido", "Usa un factor de escala distinto de 0.")
            return None
        return valor

    def _aplicar(self) -> None:
        valor = self._obtener_parametro()
        if valor is None or not self._validar_figura():
            return

        self._detener_animacion()
        transformacion = self._transform_actual
        puntos_t, matriz = transformacion.funcion(self._puntos_base, valor)
        self._dibujar_par(self._puntos_base, puntos_t, transformacion, valor)
        self._mostrar_matriz(matriz, transformacion, valor)

    def _animar(self) -> None:
        valor = self._obtener_parametro()
        if valor is None or not self._validar_figura():
            return

        transformacion = self._transform_actual
        puntos_finales, _matriz_final = transformacion.funcion(self._puntos_base, valor)

        def dibujar_frame(frame: int) -> None:
            progreso = ease_in_out(frame / (ANIMATION_FRAMES - 1))
            puntos_actuales, matriz_actual = self._calcular_frame(transformacion, valor, progreso, puntos_finales)
            grupos = [self._puntos_base, puntos_finales, puntos_actuales]

            self._ax_orig.cla()
            self._ax_trans.cla()
            configurar_ejes(self._ax_orig, f"Sprite original: {self._figura_var.get()}", grupos)
            configurar_ejes(self._ax_trans, f"Animacion: {transformacion.titulo}", grupos)

            dibujar_figura(self._ax_orig, self._puntos_base, ORIGINAL_COLOR, "Original")
            self._ax_orig.legend(fontsize=8, loc="upper right")

            dibujar_figura(self._ax_trans, self._puntos_base, ORIGINAL_COLOR, "Original", alpha=0.30)
            if transformacion.clave == "reflexion":
                self._dibujar_recta_reflexion(self._ax_trans, valor)
            dibujar_figura(self._ax_trans, puntos_finales, transformacion.color, "Destino", alpha=0.22)
            dibujar_figura(self._ax_trans, puntos_actuales, transformacion.color, transformacion.titulo, alpha=0.95)
            self._ax_trans.text(
                0.03,
                0.04,
                f"Progreso: {progreso * 100:5.1f}%",
                transform=self._ax_trans.transAxes,
                color=MUTED,
                fontsize=8,
            )
            self._ax_trans.legend(fontsize=8, loc="upper right")
            self._mostrar_matriz(matriz_actual, transformacion, self._valor_visual_frame(transformacion, valor, progreso))

        self._run_animation(ANIMATION_FRAMES, dibujar_frame)

    def _calcular_frame(
        self,
        transformacion: Transformacion,
        valor_final: float,
        progreso: float,
        puntos_finales: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if transformacion.clave == "rotacion":
            return rotacion(self._puntos_base, valor_final * progreso)
        if transformacion.clave == "escala":
            factor = 1.0 + (valor_final - 1.0) * progreso
            return escalamiento(self._puntos_base, factor)

        matriz_interpolada = np.eye(2) + (reflexion_recta_y_mx(self._puntos_base, valor_final)[1] - np.eye(2)) * progreso
        puntos_interpolados = self._puntos_base + (puntos_finales - self._puntos_base) * progreso
        return puntos_interpolados, matriz_interpolada

    def _valor_visual_frame(self, transformacion: Transformacion, valor_final: float, progreso: float) -> float:
        if transformacion.clave == "escala":
            return 1.0 + (valor_final - 1.0) * progreso
        if transformacion.clave == "reflexion":
            return valor_final
        return valor_final * progreso

    def _run_animation(self, total_frames: int, draw_frame: Callable[[int], None]) -> None:
        self._detener_animacion()
        self._animation_token += 1
        token = self._animation_token

        def paso(frame: int = 0) -> None:
            if token != self._animation_token:
                return
            draw_frame(frame)
            self._canvas.draw_idle()
            if frame < total_frames - 1:
                self._after_id = self.root.after(ANIMATION_INTERVAL, lambda: paso(frame + 1))
            else:
                self._after_id = None

        paso()

    def _comparar(self) -> None:
        valor = self._obtener_parametro()
        if valor is None or not self._validar_figura():
            return

        self._detener_animacion()
        transformacion = self._transform_actual
        puntos_t, matriz = transformacion.funcion(self._puntos_base, valor)
        grupos = [self._puntos_base, puntos_t]

        self._ax_orig.cla()
        configurar_ejes(self._ax_orig, "Comparacion: original vs transformada", grupos)
        dibujar_figura(self._ax_orig, self._puntos_base, ORIGINAL_COLOR, "Original")
        dibujar_figura(self._ax_orig, puntos_t, transformacion.color, transformacion.titulo, alpha=0.85)
        if transformacion.clave == "reflexion":
            self._dibujar_recta_reflexion(self._ax_orig, valor)
        self._ax_orig.legend(fontsize=8, loc="upper right")

        self._ax_trans.cla()
        configurar_ejes(self._ax_trans, f"{transformacion.titulo} - parametro = {valor}", grupos)
        if transformacion.clave == "reflexion":
            self._dibujar_recta_reflexion(self._ax_trans, valor)
        dibujar_figura(self._ax_trans, puntos_t, transformacion.color, transformacion.titulo)
        self._ax_trans.legend(fontsize=8, loc="upper right")

        self._mostrar_matriz(matriz, transformacion, valor)
        self._canvas.draw_idle()

    def _reset(self) -> None:
        self._detener_animacion()
        self._figura_var.set("Nave espacial")
        self._puntos_base = np.array(FIGURAS_PREDEFINIDAS["Nave espacial"], dtype=float)
        self._actualizar_lista_puntos()
        self._trans_var.set("rotacion")
        self._transform_actual = TRANSFORMACIONES[0]
        self._param_label.config(text=self._transform_actual.parametro)
        self._param_entry.delete(0, "end")
        self._param_entry.insert(0, self._transform_actual.valor_inicial)
        self._slider.config(from_=-180, to=360, resolution=1)
        self._slider_var.set(float(self._transform_actual.valor_inicial))
        self._actualizar_grafica_inicial()
        self._limpiar_matriz()

    def _actualizar_grafica_inicial(self) -> None:
        self._ax_orig.cla()
        self._ax_trans.cla()

        nombre = self._figura_var.get()
        configurar_ejes(self._ax_orig, f"Sprite original: {nombre}", [self._puntos_base])
        configurar_ejes(self._ax_trans, "Selecciona una transformacion y presiona Aplicar", [self._puntos_base])

        if len(self._puntos_base) == 0:
            self._ax_orig.text(0.5, 0.5, "Sin puntos", transform=self._ax_orig.transAxes, ha="center", va="center", color=MUTED)
        else:
            dibujar_figura(self._ax_orig, self._puntos_base, ORIGINAL_COLOR, "Original")
            self._ax_orig.legend(fontsize=8, loc="upper right")
        self._canvas.draw_idle()

    def _validar_figura(self) -> bool:
        if len(self._puntos_base) < 3:
            messagebox.showerror("Figura incompleta", "Ingresa al menos 3 puntos para formar una figura plana.")
            return False
        return True

    def _dibujar_par(self, pts_orig: np.ndarray, pts_trans: np.ndarray, transformacion: Transformacion, valor: float) -> None:
        grupos = [pts_orig, pts_trans]
        self._ax_orig.cla()
        self._ax_trans.cla()

        configurar_ejes(self._ax_orig, "Sprite original", grupos)
        configurar_ejes(self._ax_trans, f"{transformacion.titulo} - {transformacion.parametro} = {valor}", grupos)

        dibujar_figura(self._ax_orig, pts_orig, ORIGINAL_COLOR, "Original")
        if transformacion.clave == "reflexion":
            self._dibujar_recta_reflexion(self._ax_trans, valor)
        dibujar_figura(self._ax_trans, pts_trans, transformacion.color, transformacion.titulo)

        self._ax_orig.legend(fontsize=8, loc="upper right")
        self._ax_trans.legend(fontsize=8, loc="upper right")
        self._canvas.draw_idle()

    def _dibujar_recta_reflexion(self, ax, pendiente: float) -> None:
        x0, x1 = ax.get_xlim()
        xs = np.array([x0, x1], dtype=float)
        ax.plot(xs, pendiente * xs, color=REFLECTION_COLOR, linewidth=1.5, linestyle="--", alpha=0.75, label="Recta y=mx")

    def _mostrar_matriz(self, matriz: np.ndarray, transformacion: Transformacion, valor: float) -> None:
        a, b = matriz[0, 0], matriz[0, 1]
        c, d = matriz[1, 0], matriz[1, 1]
        det = np.linalg.det(matriz)
        traza = np.trace(matriz)

        texto = (
            f"  {transformacion.titulo} ({transformacion.parametro} = {valor:.3f})\n"
            f"  A = [[{a:+.4f}, {b:+.4f}],\n"
            f"       [{c:+.4f}, {d:+.4f}]]\n"
            f"  det(A) = {det:.4f}\n"
            f"  tr(A)  = {traza:.4f}"
        )
        self._texto_matriz.config(state="normal")
        self._texto_matriz.delete("1.0", "end")
        self._texto_matriz.insert("1.0", texto)
        self._texto_matriz.config(state="disabled")

    def _limpiar_matriz(self) -> None:
        self._texto_matriz.config(state="normal")
        self._texto_matriz.delete("1.0", "end")
        self._texto_matriz.insert("1.0", "  Matriz pendiente.\n  Aplica o anima una transformacion.")
        self._texto_matriz.config(state="disabled")

    @staticmethod
    def _crear_panel_scroll(parent: tk.Frame) -> tk.Frame:
        canvas = tk.Canvas(parent, bg=APP_BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        contenido = tk.Frame(canvas, bg=APP_BG)
        ventana = canvas.create_window((0, 0), window=contenido, anchor="nw")

        contenido.bind("<Configure>", lambda _event: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda event: canvas.itemconfigure(ventana, width=event.width))
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        def mover_rueda(event) -> None:
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        contenido.bind("<Enter>", lambda _event: canvas.bind_all("<MouseWheel>", mover_rueda))
        contenido.bind("<Leave>", lambda _event: canvas.unbind_all("<MouseWheel>"))
        return contenido

    @staticmethod
    def _crear_seccion(parent: tk.Frame, titulo: str) -> tk.Frame:
        tk.Label(parent, text=titulo, bg=APP_BG, fg=TEXT, font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=4, pady=(10, 2))
        frame = tk.Frame(parent, bg=SURFACE, bd=1, relief="solid", highlightbackground=BORDER)
        frame.pack(fill="x", pady=(0, 2))
        return frame

    def _boton(self, parent: tk.Frame, texto: str, cmd: Callable[[], None], color: str) -> None:
        btn = tk.Button(
            parent,
            text=texto,
            command=cmd,
            bg=color,
            fg="white",
            font=("Segoe UI", 10, "bold"),
            relief="flat",
            cursor="hand2",
            pady=8,
        )
        btn.pack(fill="x", padx=10, pady=3)
        oscuro = self._oscurecer(color)
        btn.bind("<Enter>", lambda _event: btn.config(bg=oscuro))
        btn.bind("<Leave>", lambda _event: btn.config(bg=color))

    def _boton_compacto(self, parent: tk.Frame, texto: str, cmd: Callable[[], None], color: str) -> None:
        btn = tk.Button(
            parent,
            text=texto,
            command=cmd,
            bg=color,
            fg="white",
            font=("Segoe UI", 8, "bold"),
            relief="flat",
            cursor="hand2",
            padx=7,
            pady=4,
        )
        btn.pack(side="left", fill="x", expand=True, padx=(0, 4))

    def _detener_animacion(self) -> None:
        self._animation_token += 1
        if self._after_id is not None:
            try:
                self.root.after_cancel(self._after_id)
            except tk.TclError:
                pass
            self._after_id = None

    @staticmethod
    def _oscurecer(hex_color: str) -> str:
        r = max(int(hex_color[1:3], 16) - 28, 0)
        g = max(int(hex_color[3:5], 16) - 28, 0)
        b = max(int(hex_color[5:7], 16) - 28, 0)
        return f"#{r:02x}{g:02x}{b:02x}"


def main() -> None:
    root = tk.Tk()
    MotorTransformaciones2D(root)
    root.mainloop()


if __name__ == "__main__":
    main()

